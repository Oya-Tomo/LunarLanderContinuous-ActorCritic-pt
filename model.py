from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader


class CriticNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        value_dim: int,
    ):
        super().__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, value_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class ActorNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        norm = Normal(mean, std)
        action = torch.tanh(norm.rsample())
        mean = torch.tanh(mean)
        return action, mean


@dataclass
class ModelConfig:
    lr: float = 3e-4  # learning rate for actor and critic optimizers
    tau: float = 0.005  # the rate of soft update parameters
    gamma: float = 0.99  # the rate of future reward discount


class ActorCriticModel:
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        value_dim: int = 1,
    ):
        super().__init__()

        self.config = config
        self.device = device

        self.actor_net = ActorNet(
            state_dim,
            hidden_dim,
            action_dim,
        ).to(device)
        self.critic_net = CriticNet(
            state_dim,
            action_dim,
            hidden_dim,
            value_dim,
        ).to(device)

        self.critic_net_target = CriticNet(
            state_dim,
            action_dim,
            hidden_dim,
            value_dim,
        ).to(device)

        self.critic_net_target.load_state_dict(
            self.critic_net.state_dict(),
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(),
            lr=self.config.lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(),
            lr=self.config.lr,
        )

    def select_action(self, state: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train:
            action, _ = self.actor_net.sample(state)
        else:
            _, action = self.actor_net(state)
        return action.detach().cpu().reshape(-1).numpy()

    def update_parameters(
        self,
        dataloader: DataLoader,
    ) -> tuple[float, float]:

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for state, action, reward, next_state, mask in dataloader:
            state = state.to(self.device).float()
            action = action.to(self.device).float()
            reward = reward.to(self.device).float()
            next_state = next_state.to(self.device).float()
            mask = mask.to(self.device).float().reshape(reward.shape)

            # Critic update
            with torch.no_grad():
                next_action, _ = self.actor_net.sample(next_state)
                next_q_value = self.critic_net_target(next_state, next_action)
                target_q = reward + self.config.gamma * next_q_value * mask

            current_q = self.critic_net(state, action)
            critic_loss: torch.Tensor = F.mse_loss(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            total_critic_loss += critic_loss.item()

            # Actor update
            action, _ = self.actor_net.sample(state)
            q_value = self.critic_net(state, action)
            actor_loss: torch.Tensor = -q_value.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            total_actor_loss += actor_loss.item()

        return total_critic_loss / len(dataloader), total_actor_loss / len(dataloader)

    def soft_update(self):
        for target_param, param in zip(
            self.critic_net_target.parameters(),
            self.critic_net.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data
                + (1 - self.config.tau) * target_param.data,
            )
            target_param.requires_grad = False

    def save_state_dict(self, path: str = "checkpoint.pt"):
        torch.save(
            {
                "actor": self.actor_net.state_dict(),
                "critic": self.critic_net.state_dict(),
                "actor_optim": self.actor_optimizer.state_dict(),
                "critic_optim": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_state_dict(self, path: str = "checkpoint.pt"):
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optim"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optim"])


# References
# - https://www.dskomei.com/entry/2022/06/09/171712
#
# Note
# - CtiticNet: same as Q-function
#   - input: state, action
#   - output: Q-value
# - ActorNet: same as policy function
#   - input: state
#   - output: action (mean, std)
#
# Caution
# - [required] please scale the state and action to [-1, 1] before passing to the CriticNet and ActorNet


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 8
    action_dim = 2
    model = ActorCriticModel(
        ModelConfig(),
        device,
        state_dim,
        action_dim,
    )
    print(model.actor_net)
    print(model.critic_net)
