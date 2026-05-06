import math
from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, momentum: float = 0.9998) -> None:
        self._base_momentum = momentum
        self.model = deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _momentum(self, step: int) -> float:
        return self._base_momentum * (1.0 - math.exp(-step / 2000.0))

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        m = self._momentum(step)
        ema_state = self.model.state_dict()
        src_state = model.state_dict()
        for name, ema_p in ema_state.items():
            if not ema_p.dtype.is_floating_point:
                continue
            ema_p.mul_(m).add_(src_state[name].detach(), alpha=1.0 - m)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
