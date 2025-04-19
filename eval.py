import torch

from env import LunarLanderV3
from model import ActorCriticModel, ModelConfig


def eval(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = LunarLanderV3(
        gui=True,
        device=device,
    )

    agent = ActorCriticModel(
        ModelConfig(),
        device=device,
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
    )

    agent.load_state_dict(checkpoint)

    for i in range(100):
        state, _ = env.reset()
        for s in range(10000):
            if s < 50:
                action = env.env.action_space.sample()
            else:
                action = agent.select_action(torch.tensor(state).float().to(device))
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated or truncated
            if done:
                break


if __name__ == "__main__":
    eval("checkpoints/checkpoint999.pt")
