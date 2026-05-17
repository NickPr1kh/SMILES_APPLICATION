import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)
