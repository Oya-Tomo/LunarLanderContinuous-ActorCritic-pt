import torch
import gymnasium as gym


class LunarLanderV3:
    def __init__(self, gui: bool = True, device: torch.device = torch.device("cuda")):
        self.env = gym.make(
            "LunarLander-v3",
            render_mode="human" if gui else None,
            continuous=True,
        )
        self.env.reset(seed=42)
        self.device = device

        self.action_dim = self.env.action_space.shape[0]
        self.observation_dim = self.env.observation_space.shape[0]

        self.action_min = (
            torch.tensor(self.env.action_space.low).float().to(self.device)
        )
        self.action_max = (
            torch.tensor(self.env.action_space.high).float().to(self.device)
        )

        self.obs_min = (
            torch.tensor(self.env.observation_space.low).float().to(self.device)
        )
        self.obs_max = (
            torch.tensor(self.env.observation_space.high).float().to(self.device)
        )

    def step(self, action: torch.Tensor):
        action = torch.tensor(action)
        action = action.to(self.device)
        action = (action + 1) / 2.0 * (
            self.action_max - self.action_min
        ) + self.action_min
        action = action.detach().cpu().numpy()
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation = torch.tensor(observation).float().to(self.device)
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation = observation * 2 - 1

        reward = torch.tensor(reward).float().to(self.device)
        reward = reward / 200.0

        return observation, reward, terminated, truncated, info

    def sample(self):
        action = self.env.action_space.sample()
        action = torch.tensor(action).float().to(self.device)
        action = (action - self.action_min) / (self.action_max - self.action_min)
        action = action * 2 - 1
        return action.cpu().numpy()

    def reset(self):
        return self.env.reset()
