from models.ppo_lm import PPOLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

import random
import torch
import numpy as np
from lm_models.modeling_value_head import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, pipeline
from data_utils.lm_data import dict_collator, build_dataset
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar


def main():

    project_name = "lightning_RL_LM"
    dataset_name = "imdb"

    wandb_logger = WandbLogger(
        project=project_name,
        log_model=False,
        save_dir="experiments",
        tags=[dataset_name, "PPO"],
    )

    model_name = "lvwerra/gpt2-imdb"
    device = "cuda"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(model_name)

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
    
    config.batch_size = 12

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dict_collator,
        shuffle=True,
        drop_last=True,
    )

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

    checkpoint_callback = ModelCheckpoint(
        monitor="traj_rewards_step",
        save_top_k=3,
        mode="max",
        every_n_train_steps=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # pl trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        gpus=1,
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            lr_monitor,
        ],
    )
    # fit
    trainer.fit(ppo_model)


if __name__ == "__main__":
    main()
