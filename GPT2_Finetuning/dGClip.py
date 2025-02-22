"""
This file provides a basic implementation of the delta-gradient-clipping
algorithm as a pytorch optimizer.
"""

from typing import Any, Dict, Iterable

import torch


class dGClip(torch.optim.Optimizer):
    """Class for the delta-Gradient Clipping optimizer."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]],
        lr: float,
        gamma: float = 0.1,
        delta: float = 0.001,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0:
            raise ValueError("Learning rate (eta) is not > 0.")
        if gamma <= 0:
            raise ValueError("Gradient norm threshold (gamma) is not > 0.")
        if delta < 0:
            raise ValueError("delta is not >= 0.")
        if weight_decay < 0:
            raise ValueError("Weight decay is not >= 0.")
        defaults = dict(lr=lr, gamma=gamma, delta=delta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            parameter_list = param_group["params"]
            concatenated_weights = torch.cat(
                [param.grad.view(-1) for param in parameter_list]
            )
            gradient_norm = torch.norm(concatenated_weights)

            if gradient_norm != 0:
                step_size = param_group["lr"] * min(
                    1, max(param_group["delta"], param_group["gamma"] / gradient_norm)
                )
            else:
                step_size = 0

            for p in param_group["params"]:
                if param_group["weight_decay"]:
                    p.grad = p.grad.add(p, alpha=param_group["weight_decay"])
                p.add_(p.grad, alpha=-step_size)

        return loss
