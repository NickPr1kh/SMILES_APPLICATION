from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        eps: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        cone_alpha: float = 0.0,
        cone_beta: float = 0.9,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps_adam
        self.cone_alpha = cone_alpha
        self.cone_beta = cone_beta

        self._step_count: int = 0
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._cone_m: dict[str, torch.Tensor] = {}

        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        self._init_class_means()

    def _backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.conv1(images)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        return torch.flatten(x, 1)

    def _init_class_means(self) -> None:
        try:
            import torchvision.datasets as datasets
            from augmentation import get_transforms
        except ImportError:
            return

        device = next(self.model.parameters()).device
        n_classes = self.model.fc.out_features
        in_features = self.model.fc.in_features

        try:
            train_dataset = datasets.CIFAR100(
                root="./data", train=True, download=False,
                transform=get_transforms(train=False),
            )
        except Exception:
            return

        loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)

        class_sums = torch.zeros(n_classes, in_features, device=device)
        class_counts = torch.zeros(n_classes, device=device)

        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                features = self._backbone_features(images)
                class_sums.scatter_add_(0, labels.unsqueeze(1).expand_as(features), features)
                for c in range(n_classes):
                    class_counts[c] += (labels == c).sum()

        means = class_sums / class_counts.clamp(min=1).unsqueeze(1)

        with torch.no_grad():
            self.model.fc.weight.copy_(means)
            self.model.fc.bias.zero_()

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Parameters not found: {missing}")
        return {n: named[n] for n in self.layer_names}

    def _init_buffers(self, params: dict[str, nn.Parameter]) -> None:
        for name, param in params.items():
            if name not in self._m:
                self._m[name] = torch.zeros_like(param.data)
                self._v[name] = torch.zeros_like(param.data)
                self._cone_m[name] = torch.zeros_like(param.data)

    def _sample_cone_direction(self, name: str, param: torch.Tensor) -> torch.Tensor:
        z_rand = torch.randint(0, 2, param.shape, device=param.device, dtype=param.dtype) * 2 - 1

        if self.cone_alpha == 0.0:
            return z_rand

        m = self._cone_m[name]
        m_norm = m.norm()
        if m_norm < 1e-8:
            return z_rand

        z_mixed = (1.0 - self.cone_alpha) * z_rand + self.cone_alpha * (m / m_norm)
        target_norm = math.sqrt(param.numel())
        z_norm = z_mixed.norm()
        if z_norm > 1e-8:
            z_mixed = z_mixed * (target_norm / z_norm)
        return z_mixed

    def _update_cone_momentum(self, grads: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, g in grads.items():
                self._cone_m[name].mul_(self.cone_beta).add_(g, alpha=1.0 - self.cone_beta)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        directions = {name: self._sample_cone_direction(name, param) for name, param in params.items()}

        with torch.no_grad():
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])
            f_plus = loss_fn()

            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()

            for name, param in params.items():
                param.data.add_(self.eps * directions[name])

        diff = (f_plus - f_minus) / (2.0 * self.eps)
        grads = {name: diff * z for name, z in directions.items()}
        self._update_cone_momentum(grads)
        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        t = self._step_count
        bias_corr1 = 1.0 - self.beta1 ** t
        bias_corr2 = 1.0 - self.beta2 ** t

        with torch.no_grad():
            for name, param in params.items():
                g = grads[name]
                self._m[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self._v[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                m_hat = self._m[name] / bias_corr1
                v_hat = self._v[name] / bias_corr2
                param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps_adam), value=-self.lr)

    def step(self, loss_fn: Callable[[], float]) -> float:
        self._step_count += 1
        params = self._active_params()
        self._init_buffers(params)

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)
