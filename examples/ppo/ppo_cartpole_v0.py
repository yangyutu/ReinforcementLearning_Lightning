from models.ppo import PPO
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger


def main():

    project_name = "lightning_RL"
    env_name = "CartPole-v0"
    wandb_logger = WandbLogger(
        project=project_name,  # group runs in "MNIST" project
        log_model=False,
        save_dir="experiments",
        tags=[env_name, "PPO"],
    )

    model = PPO(env_name)
    # pl trainer
    trainer = Trainer(max_epochs=10, accelerator="auto", gpus=1, logger=wandb_logger)
    # fit
    trainer.fit(model)


if __name__ == "__main__":
    main()
