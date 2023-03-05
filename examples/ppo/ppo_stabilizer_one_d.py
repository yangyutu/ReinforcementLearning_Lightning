from models.ppo import PPO
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from envs.stablizer import StablizerOneD
import random
import torch
import numpy as np


def main():

    project_name = "lightning_RL"
    env_name = "StablizerOneD"

    env = StablizerOneD()

    wandb_logger = WandbLogger(
        project=project_name,
        log_model=False,
        save_dir="experiments",
        tags=[env_name, "PPO"],
    )

    model = PPO(
        env,
        lr_actor=1e-3,
        lr_critic=1e-3,
    )
    # pl trainer
    trainer = Trainer(max_epochs=10, accelerator="auto", gpus=1, logger=wandb_logger)
    # fit
    trainer.fit(model)

    states = torch.Tensor(np.linspace(-5, 5, 100))
    actions = model.actor(states[:, None])
    print(actions)


if __name__ == "__main__":
    main()
