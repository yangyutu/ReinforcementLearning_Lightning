import argparse
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.nn.functional import log_softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from models.networks import ActorCategorical, create_mlp, ActorAgent
from data_utils.data import ExperienceSourceDataset
import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:
    _GYM_AVAILABLE = False
else:
    _GYM_AVAILABLE = True


class Reinforce(pl.LightningModule):
    r"""PyTorch Lightning implementation of REINFORCE_.

    Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:
        >>> from pl_bolts.models.rl.reinforce_model import Reinforce
        ...
        >>> model = Reinforce("CartPole-v0")

    Train::

        trainer = Trainer()
        trainer.fit(model)

    Note:
        This example is based on:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/02_cartpole_reinforce.py

    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`

    .. _REINFORCE:
        https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
    """

    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.01,
        batch_size: int = 32,
        n_steps: int = 10,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
        epoch_len: int = 1000,
        num_batch_episodes: int = 4,
        **kwargs
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            n_steps: number of stakes per discounted experience
            entropy_beta: entropy coefficient
            epoch_len: how many batches before pseudo epoch
            num_batch_episodes: how many episodes to rollout for each batch of training
            avg_reward_len: how many episodes to take into account when calculating the avg reward
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "This Module requires gym environment which is not installed yet."
            )

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.batches_per_epoch = self.batch_size * epoch_len
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.n_steps = n_steps
        self.num_batch_episodes = num_batch_episodes
        self.steps_per_epoch = epoch_len

        self.save_hyperparameters()

        # Model components
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        # initialize actor network
        self.actor_net = ActorCategorical(
            create_mlp(self.env.observation_space.shape, self.env.action_space.n)
        )
        self.agent = ActorAgent(self.actor_net)

        # Tracking metrics
        self.total_steps = 0
        self.total_rewards = [0]
        self.done_episodes = 0
        self.avg_rewards = 0
        self.reward_sum = 0.0
        self.batch_episodes = 0
        self.avg_reward_len = avg_reward_len

        self.batch_states = []
        self.batch_actions = []
        self.batch_reward_to_go = []
        self.cur_rewards = []

        self.state = torch.FloatTensor(self.env.reset()[0])

    def calc_reward_to_go(self, rewards: List[float]) -> List[float]:
        """Calculate the discounted rewards of all rewards in list.

        Args:
            rewards: list of rewards from latest batch

        Returns:
            list of discounted rewards
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * self.gamma) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def train_batch(
        self,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Yield:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """
        episode_step = 0
        while True:

            _, action, _ = self.agent(self.state, self.device)

            next_state, reward, done, _, _ = self.env.step(action.cpu().numpy())

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.cur_rewards.append(reward)

            self.state = torch.FloatTensor(next_state)
            self.total_steps += 1
            episode_step += 1
            
            epoch_end = episode_step == (self.steps_per_epoch - 1)
            

            if done:
                # add rewards among all steps in an finish epoch
                self.batch_reward_to_go.extend(self.calc_reward_to_go(self.cur_rewards))
                self.batch_episodes += 1
                self.done_episodes += 1
                self.total_rewards.append(sum(self.cur_rewards))
                self.avg_rewards = float(
                    np.mean(self.total_rewards[-self.avg_reward_len :])
                )
                self.cur_rewards = []
                self.state = torch.FloatTensor(self.env.reset()[0])
            
            # if epoch_end and not done:
            #     # if no getting final rewards, clear all steps so far and reset
            #     self.batch_states.clear()
            #     self.batch_actions.clear()
            #     self.cur_rewards.clear()
            #     self.batch_reward_to_go.clear()
            #     episode_step = 0
            #     self.state = torch.FloatTensor(self.env.reset()[0])
           
            
            if self.batch_episodes >= self.num_batch_episodes:
                for state, action, reward_to_go in zip(
                    self.batch_states, self.batch_actions, self.batch_reward_to_go
                ):
                    yield state, action, reward_to_go

                self.batch_episodes = 0

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_reward_to_go.clear()

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def loss(self, states, actions, rewards_to_go) -> Tensor:

        pi, _  = self.actor_net.forward(states)
        log_prob = self.actor_net.get_log_prob(pi, actions)

        log_prob_actions = rewards_to_go * log_prob
        loss = -log_prob_actions.mean()

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        states, actions, rewards_to_go = batch

        loss = self.loss(states, actions, rewards_to_go)

        log_dict = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
            "loss": loss,
        }

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        return optimizer

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    @staticmethod
    def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
        """Adds arguments for DQN model.

        Note:
            These params are fine tuned for Pong env.

        Args:
            arg_parser: the current argument parser to add to

        Returns:
            arg_parser with model specific cargs added
        """
        arg_parser.add_argument(
            "--batches_per_epoch",
            type=int,
            default=10000,
            help="number of batches in an epoch",
        )
        arg_parser.add_argument(
            "--batch_size", type=int, default=32, help="size of the batches"
        )
        arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

        arg_parser.add_argument(
            "--env", type=str, required=True, help="gym environment tag"
        )
        arg_parser.add_argument(
            "--gamma", type=float, default=0.99, help="discount factor"
        )

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )

        arg_parser.add_argument(
            "--entropy_beta",
            type=float,
            default=0.01,
            help="entropy value",
        )

        return arg_parser


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Reinforce.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Reinforce(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max", verbose=True
    )

    seed_everything(123)
    trainer = Trainer.from_argparse_args(
        args, deterministic=True, callbacks=checkpoint_callback
    )
    trainer.fit(model)


if __name__ == "__main__":
    import networks
    import data_utils.data as data
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers.wandb import WandbLogger
    from envs.stablizer import StablizerOneD
    project_name = "lightning_RL"
    env_name = "StablizerOneD"

    env = StablizerOneD(end_step=float('inf'))
    # wandb_logger = WandbLogger(
    #     project=project_name,  # group runs in "MNIST" project
    #     log_model=False,
    #     save_dir="experiments",
    #     tags=[env_name, "Reinforce"],
    # )

    model = Reinforce(env)
    # pl trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        gpus=1,
        # logger=wandb_logger
    )
    # fit
    res = trainer.fit(model)
