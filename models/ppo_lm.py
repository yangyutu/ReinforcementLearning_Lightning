from typing import List, Tuple, Union, Dict
import time
import pytorch_lightning as pl

from data_utils.data import ExperienceSourceDataset, LMDataLoader

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from transformers import DataCollatorForLanguageModeling
from omegaconf import DictConfig
from datasets import Dataset
from models.utils import (
    WANDB_PADDING,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
    AdaptiveKLController,
    FixedKLController,
    batch_to_device,
)
from data_utils.lm_data import dict_collator, LengthSampler


class PPOLM(pl.LightningModule):
    """
    PyTorch Lightning implementation of `PPO
    <https://arxiv.org/abs/1707.06347>`_
    Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

    Example:
        model = PPO("CartPole-v0")
    Train:
        trainer = Trainer()
        trainer.fit(model)
    Note:
        This example is based on:
        https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
        https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/rl/reinforce_model.py

    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        input_dataloader,
        reward_model,
        is_encoder_decoder,
        gen_kwargs,
        output_min_length,
        output_max_length,
        config: DictConfig,
    ) -> None:

        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.input_dataloader = input_dataloader
        self.is_encoder_decoder = is_encoder_decoder

        self.gen_kwargs = gen_kwargs
        self.output_length_sampler = LengthSampler(output_min_length, output_max_length)

        self.config = config

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                self.config.init_kl_coef, self.config.target, self.config.horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # need to turn off the automatic optimization and perform manual
        # https://pytorch-lightning.readthedocs.io/en/stable/model/build_model_advanced.html
        self.automatic_optimization = False
        # Hyperparameters
        self.save_hyperparameters()

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_step_reward = 0

    def train_batch(self) -> Dict[str, torch.Tensor]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 16,
        }

        for epoch, batch in enumerate(self.input_dataloader):
            query_tensors = [query.to(self.device) for query in batch["input_ids"]]

            #### Get response from gpt2
            response_tensors = []
            for query in query_tensors:
                gen_len = self.output_length_sampler()
                self.gen_kwargs["max_new_tokens"] = gen_len
                response = self.model.generate(query[None, :], **self.gen_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
            batch["response"] = [
                self.tokenizer.decode(r.squeeze()) for r in response_tensors
            ]

            #### Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.reward_model(texts, **sent_kwargs)
            traj_rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

            model_inputs = self.prepare_model_inputs(query_tensors, response_tensors)
            model_inputs_names = list(model_inputs.keys())

            # no need to have gradient since all these are used as reference/bootstrapping targets
            with torch.no_grad():
                all_logprobs_old, _, values_ref, masks = self.batched_forward_pass(
                    self.model, query_tensors, response_tensors, model_inputs
                )
                ref_logprobs, _, _, _ = self.batched_forward_pass(
                    self.ref_model, query_tensors, response_tensors, model_inputs
                )

            rewards, non_score_reward = self.compute_rewards(
                traj_rewards, all_logprobs_old, ref_logprobs, masks
            )

            batch_dict = {
                "queries": query_tensors,
                "responses": response_tensors,
                "logprobs_old": all_logprobs_old,
                "logprobs_ref": ref_logprobs,
                "values": values_ref,
                "rewards": rewards,
                "non_score_reward": non_score_reward,
                "traj_rewards": traj_rewards,
                "masks": masks,
            }

            # mini_batch_dict.update({"model_inputs": model_inputs})

            yield batch_dict, model_inputs

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(
            scores, logprobs, ref_logprobs, masks
        ):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": q, "attention_mask": torch.ones_like(q)}
                    for q in queries
                ]
            )

            input_data = batch_to_device(input_data, self.device)
            decoder_inputs = self.data_collator(
                [
                    {"input_ids": r, "attention_mask": torch.ones_like(r)}
                    for r in responses
                ]
            )

            decoder_inputs = batch_to_device(decoder_inputs, self.device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]

            input_data.pop("labels", None)  # we don't want to compute LM losses

        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                    for ids in input_ids
                ]
            )
            input_data = batch_to_device(input_data, self.device)
        return input_data

    def batched_forward_pass(
        self,
        model,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs]
                for key, value in model_inputs.items()
            }
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            # exclude the last eos logits; apply log_softmax across vocab_size axis to get the logprob
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(fbs):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    # Decoder sentence starts with the end of the query
                    start = len(query_batch[j]) - 1
                    if (
                        attention_mask[j, 0] == 0
                    ):  # offset left padding, shift to the first non-zero
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                if len(logprobs[j, start:end]) < 2:
                    raise ValueError(
                        "Responses are too short. Make sure they are at least 4 tokens long."
                    )

                masks[j, :start] = 0
                masks[j, end:] = 0

            all_logits.append(logits)
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1],
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `hidden_dim`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(
                entropy=entropy,
                approxkl=approxkl,
                policykl=policykl,
                clipfrac=pg_clipfrac,
                advantages=advantages,
                advantages_mean=masked_mean(advantages, mask),
                ratio=ratio,
            ),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=masked_mean(vpreds, mask),
                error=masked_mean((vpreds - returns) ** 2, mask),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        """
        Carries out a single update to actor and critic network from a batch of replay buffer.

        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        # each training step will consist of several training epochs on the generated batch
        batch_dict, model_inputs_dict = batch
        batch_dict.update(model_inputs_dict)
        mini_batch_data = Dataset.from_dict(batch_dict)
        mini_batch_data.set_format("torch", device=self.device)
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
            collate_fn=dict_collator,
        )

        self.log(
            "traj_rewards",
            torch.Tensor(batch_dict["traj_rewards"]).mean().item(),
            batch_size=len(batch_dict["traj_rewards"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        opt = self.optimizers()
        t = time.time()
        timing = {}
        all_stats = []
        for _ in range(self.config.ppo_epochs):
            for minibatch in mini_batch_dataloader:
                model_inputs = {
                    k: torch.stack(minibatch[k]) for k in model_inputs_dict.keys()
                }
                logprobs, logits, vpreds, _ = self.batched_forward_pass(
                    self.model,
                    minibatch["queries"],
                    minibatch["responses"],
                    model_inputs,
                )

                old_logprobs = torch.stack(minibatch["logprobs_old"])
                values = torch.stack(minibatch["values"])
                masks = torch.stack(minibatch["masks"])
                rewards = torch.stack(minibatch["rewards"])

                loss_p, loss_v, train_stats = self.loss(
                    old_logprobs, values, rewards, logits, vpreds, logprobs, masks
                )
                loss = loss_p + loss_v

                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()

                all_stats.append(train_stats)

        timing["time/ppo/optimize_step"] = time.time() - t

        

        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=batch_dict["rewards"],
            logprobs=batch_dict["logprobs_old"],
            ref_logprobs=batch_dict["logprobs_ref"],
            non_score_reward=batch_dict["non_score_reward"],
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=batch_dict["masks"],
        )

        self.log_dict(
            stats,
            batch_size=self.config.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        mask = data.pop("masks")

        kl_list = ((data["logprobs"] - data["ref_logprobs"]) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(data["non_score_reward"], mask)

        stats = {
            "objective/kl": mean_kl,
            #"objective/kl_dist": kl_list,
            #"objective/logprobs": data["logprobs"],
            #"objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
        }

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        
        # remove some arrays
        stats.pop("ppo/policy/advantages", None)
        stats.pop("ppo/policy/ratio", None)
        
        
        return stats

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return optimizer

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        # dataset = ExperienceSourceDataset(self.train_batch)
        # dataloader = DataLoader(dataset=dataset, batch_size=self.config.batch_size)
        # dataloader = DataLoader(dataset=dataset, batch_size=1)
        dataloader = LMDataLoader(self.train_batch)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()


def test_generation():

    from lm_models.modeling_value_head import AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer, pipeline
    from data_utils.lm_data import dict_collator, build_dataset

    model_name = "lvwerra/gpt2-imdb"
    device = "cuda"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = build_dataset(model_name)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=dict_collator,
        shuffle=True,
        drop_last=True,
    )

    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    sentiment_pipe = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=0
    )

    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }

    for epoch, batch in enumerate(input_dataloader):
        query_tensors = batch["input_ids"]

        #### Get response from gpt2
        response_tensors = []
        for query in query_tensors:
            generation_kwargs["max_new_tokens"] = gen_len
            response = model.generate(query[None, :], **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_pipeline(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        print(texts)
        print(rewards)
        break


def test_model():
    from lm_models.modeling_value_head import AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer, pipeline
    from data_utils.lm_data import dict_collator, build_dataset

    model_name = "lvwerra/gpt2-imdb"
    device = "cuda"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(model_name)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=dict_collator,
        shuffle=True,
        drop_last=True,
    )

    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    sentiment_pipe = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=0
    )

    config = DictConfig({})
    config.learning_rate = 1e-5
    config.adap_kl_ctrl = True
    config.init_kl_coef = 0.2
    config.target = 6
    config.horizon = 10000
    config.gamma = 1.0
    config.lam = 0.95
    config.cliprange = 0.2
    config.cliprange_value = 0.2
    config.vf_coef = 0.1
    config.batch_size = 256
    config.forward_batch_size = None
    config.mini_batch_size = 1
    config.ppo_epochs = 4
    config.max_grad_norm = None
    config.seed = 0

    ppo_model = PPOLM(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        input_dataloader=dataloader,
        reward_model=sentiment_pipe,
        gen_kwargs=gen_kwargs,
        is_encoder_decoder=False,
        output_min_length=4,
        output_max_length=16,
        config=config,
    )
    count = 0
    for batch in iter(ppo_model.train_dataloader()):
        loss = ppo_model.training_step(batch, 0)
        count += 1
        if count > 1:
            break
    print(batch)
    print(loss)


if __name__ == "__main__":
    test_model()
