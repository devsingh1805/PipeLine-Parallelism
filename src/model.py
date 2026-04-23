
import torch.nn as nn


class ShardedMLP(nn.Module):
    def __init__(self, dim, total_layers, rank, world_size):
        super().__init__()

        self.rank     = rank
        self.is_first = (rank == 0)
        self.is_last  = (rank == world_size - 1)

        layers_per_worker = total_layers // world_size

        layers = []
        for _ in range(layers_per_worker):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        if self.is_last:
            layers.append(nn.Linear(dim, 2))
            self.loss_fn = nn.CrossEntropyLoss()

        self.net = nn.Sequential(*layers)

    def forward(self, x, targets=None):
        import time
        time.sleep(0.3)      # 300ms artificial delay per worker
        x = self.net(x)
        if self.is_last and targets is not None:
            return self.loss_fn(x, targets)
        return x