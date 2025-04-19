from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    max_size: int = 10000


class ReplayDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.replay_buffer: list[
            tuple[
                torch.Tensor,  # state
                torch.Tensor,  # action
                torch.Tensor,  # reward
                torch.Tensor,  # next state
                torch.Tensor,  # done mask
            ]
        ] = []

        self.config = config

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        return self.replay_buffer[idx]

    def add(self, state, action, reward, next_state, done):
        self.replay_buffer.append(
            (
                torch.tensor(state).cpu(),
                torch.tensor(action).cpu(),
                torch.tensor([reward]).cpu(),
                torch.tensor(next_state).cpu(),
                torch.tensor([done]).cpu(),
            )
        )

    def periodic_delete(self):
        if len(self.replay_buffer) > self.config.max_size:
            self.replay_buffer = self.replay_buffer[
                len(self.replay_buffer) - self.config.max_size :
            ]


@dataclass
class DataloaderConfig:
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = False


def create_dataloader(
    dataset: ReplayDataset,
    config: DataloaderConfig,
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
