from __future__ import annotations

import torch
import torch.nn as nn


def _beta_sample(alpha: float, batch: int, device, dtype):
    concentration = torch.full((batch, 1, 1, 1), alpha, device=device,
                               dtype=dtype)
    first = torch._standard_gamma(concentration)
    second = torch._standard_gamma(concentration)
    return first / (first + second + 1e-12)


class _TrainingOnlyPerturbation(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def should_apply(self, x):
        return (self.training and x.ndim == 4 and x.shape[0] > 1
                and torch.rand((), device=x.device) < self.p)


class MixStyle(_TrainingOnlyPerturbation):


    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__(p=p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x):
        if not self.should_apply(x):
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = (variance + self.eps).sqrt()
        normalized = (x - mean) / std
        permutation = torch.randperm(x.shape[0], device=x.device)
        lam = _beta_sample(self.alpha, x.shape[0], x.device, x.dtype)
        mixed_mean = lam * mean + (1.0 - lam) * mean[permutation]
        mixed_std = lam * std + (1.0 - lam) * std[permutation]
        return normalized * mixed_std + mixed_mean


class DSU(_TrainingOnlyPerturbation):


    def __init__(self, p: float = 0.5, eps: float = 1e-6):
        super().__init__(p=p)
        self.eps = float(eps)

    def forward(self, x):
        if not self.should_apply(x):
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = (variance + self.eps).sqrt()
        normalized = (x - mean) / std

        mean_uncertainty = mean.var(dim=0, keepdim=True, unbiased=False)
        std_uncertainty = std.var(dim=0, keepdim=True, unbiased=False)
        perturbed_mean = mean + torch.randn_like(mean) * (
            mean_uncertainty + self.eps).sqrt()
        perturbed_std = std + torch.randn_like(std) * (
            std_uncertainty + self.eps).sqrt()
        perturbed_std = perturbed_std.clamp_min(self.eps)
        return normalized * perturbed_std + perturbed_mean


class EFDMix(_TrainingOnlyPerturbation):


    def __init__(self, p: float = 0.5, alpha: float = 1.0):
        super().__init__(p=p)
        self.alpha = float(alpha)

    def forward(self, x):
        if not self.should_apply(x):
            return x
        batch, channels, height, width = x.shape
        flat = x.reshape(batch, channels, -1)
        sorted_values, sorted_indices = flat.sort(dim=-1)
        permutation = torch.randperm(batch, device=x.device)
        lam = _beta_sample(self.alpha, batch, x.device, x.dtype).reshape(
            batch, 1, 1)
        mixed_sorted = (
            lam * sorted_values
            + (1.0 - lam) * sorted_values[permutation]
        )
        mixed = torch.empty_like(flat)
        mixed.scatter_(dim=-1, index=sorted_indices, src=mixed_sorted)
        return mixed.reshape(batch, channels, height, width)


def build_feature_augmentation(cfg):
    cfg = dict(cfg)
    augmentation_type = cfg.pop("type")
    modules = {
        "MixStyle": MixStyle,
        "DSU": DSU,
        "EFDMix": EFDMix,
    }
    if augmentation_type not in modules:
        raise ValueError(
            f"Unsupported feature augmentation: {augmentation_type}")
    return modules[augmentation_type](**cfg)
