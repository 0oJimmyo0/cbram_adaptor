"""Eager AdamW and cosine scheduling for the CBraMod TMLR runtime.

The cluster image's PyTorch 2.12 optimizer wrapper blocks in ``torch._dynamo``
before the first optimizer artifact is written.  These two small classes keep
the locked AdamW defaults and CosineAnnealingLR equation while avoiding that
unrelated import-time path.  They intentionally expose the same ``param_groups``
surface used by the runner and are recorded in run provenance.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


class EagerAdamW:
    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.param_groups = []
        self.state: dict[torch.Tensor, dict[str, Any]] = {}
        self.defaults = {"betas": betas, "eps": float(eps)}
        for group in params:
            copied = dict(group)
            copied["params"] = list(copied["params"])
            copied.setdefault("lr", 1e-3)
            copied.setdefault("weight_decay", 0.0)
            self.param_groups.append(copied)
        if not self.param_groups or not any(group["params"] for group in self.param_groups):
            raise ValueError("optimizer got an empty parameter list")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    if set_to_none:
                        parameter.grad = None
                    else:
                        parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("EagerAdamW does not support sparse gradients")
                state = self.state.setdefault(parameter, {})
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)


class EagerCosineAnnealing:
    def __init__(self, optimizer: EagerAdamW, t_max: int, eta_min: float = 0.0) -> None:
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        self.optimizer = optimizer
        self.t_max = int(t_max)
        self.eta_min = float(eta_min)
        self.last_epoch = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self) -> None:
        self.last_epoch += 1
        progress = min(self.last_epoch, self.t_max)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.eta_min + (base_lr - self.eta_min) * (
                1.0 + math.cos(math.pi * progress / self.t_max)
            ) / 2.0
