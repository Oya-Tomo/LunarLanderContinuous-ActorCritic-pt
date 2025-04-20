import os
from dataclasses import dataclass

import torch
import wandb

from env import LunarLanderV3
from model import ActorCriticModel, ModelConfig
from dataset import ReplayDataset, DatasetConfig, create_dataloader, DataloaderConfig
from eval import eval


@dataclass
class TrainingConfig:
    epochs: int = 100000
    episodes_per_epoch: int = 20
    max_steps_per_episode: int = 10000
    soft_update_per_epoch: int = 10
    seed: int = 42
    random_start_steps: int = 60


def train():
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="LunarLanderContinuous-ActorCritic-pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_config = TrainingConfig()

    env = LunarLanderV3(
        gui=False,
        device=device,
    )

    agent = ActorCriticModel(
        ModelConfig(
            lr=0.001,
            gamma=0.95,
            tau=0.01,
        ),
        device=device,
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
    )
    dataset = ReplayDataset(
        DatasetConfig(
            max_size=100000,
        ),
    )

    print("Filling dataset with random samples...")
    while len(dataset) < 100000:
        state, _ = env.reset()
        done = False
        step = 0

        while not done and step < training_config.max_steps_per_episode:
            action = env.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            dataset.add(
                state,
                action,
                reward,
                next_state,
                float(not done),
            )

            state = next_state
            step += 1

    print("Starting training...")
    for epoch in range(training_config.epochs):
        total_reward_history = []

        for episode in range(training_config.episodes_per_epoch):
            state, _ = env.reset()
            done = False
            step = 0
            total_reward = 0.0

            while not done and step < training_config.max_steps_per_episode:
                if step < training_config.random_start_steps:
                    action = env.sample()
                else:
                    action = agent.select_action(torch.tensor(state).float().to(device))
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                dataset.add(
                    state,
                    action,
                    reward,
                    next_state,
                    float(not done),
                )

                state = next_state
                step += 1

            total_reward_history.append(total_reward)

        dataset.periodic_delete()
        dataloader = create_dataloader(
            dataset,
            DataloaderConfig(
                batch_size=64,
                shuffle=True,
                num_workers=0,
            ),
        )

        critic_loss, actor_loss = agent.update_parameters(dataloader)

        if epoch % training_config.soft_update_per_epoch == 0:
            agent.soft_update()

        print(
            f"Epoch: {epoch}, "
            f"Episode: {episode}, "
            f"Critic Loss: {critic_loss:.8f}, "
            f"Actor Loss: {actor_loss:.8f}, "
            f"Total Reward Average: {sum(total_reward_history) / len(total_reward_history):.8f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "episode": episode,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "total_reward": sum(total_reward_history) / len(total_reward_history),
            }
        )

        if epoch % 10 == 9:
            agent.save_state_dict(f"checkpoints/checkpoint{epoch}.pt")

    agent.save_state_dict("checkpoints/checkpoint.pt")


if __name__ == "__main__":
    train()
