from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn


BuildModelFn = Callable[..., nn.Module]
BuildLossFn = Callable[..., nn.Module]
BuildOptimizerFn = Callable[..., torch.optim.Optimizer]


@dataclass(frozen=True)
class ModelSpec:
    family: str
    size: str
    build_model: BuildModelFn
    build_loss: BuildLossFn
    build_optimizer: BuildOptimizerFn

    @property
    def name(self) -> str:
        return f"{self.family}_{self.size}"


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"model {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> Iterable[str]:
    return sorted(_REGISTRY)


def _register_dfine() -> None:
    from adp.model.dfine import build_loss, build_model, build_optimizer

    for size in ("n", "s", "m", "l", "x"):
        register(
            ModelSpec(
                family="dfine",
                size=size,
                build_model=lambda size=size, **kw: build_model(model_name=size, **kw),
                build_loss=lambda size=size, **kw: build_loss(model_name=size, **kw),
                build_optimizer=build_optimizer,
            )
        )


def _register_ane() -> None:
    try:
        from adp.model.ane import build_loss, build_model, build_optimizer
    except ImportError:
        return

    for size in ("n", "s", "m"):
        register(
            ModelSpec(
                family="ane",
                size=size,
                build_model=lambda size=size, **kw: build_model(model_name=size, **kw),
                build_loss=lambda size=size, **kw: build_loss(model_name=size, **kw),
                build_optimizer=build_optimizer,
            )
        )


_register_dfine()
_register_ane()
