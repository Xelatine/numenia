"""Protection des synapses contre le catastrophic forgetting (SI).

L'importance d'une synapse = accumulation de |grad|² pendant l'entraînement.
Au fine-tuning, tout déplacement d'une synapse protégée est pénalisé
quadratiquement : le nouveau savoir apprend AUTOUR de l'ancien.

Le terme est celui de Synaptic Intelligence (Zenke et al., 2017) dans sa forme
cumulative simple ; on ne prétend pas à la variante par tâche.
"""

from __future__ import annotations

import torch


class SynapseProtection:
    def __init__(self, model: torch.nn.Module, damping: float = 1e3):
        self.damping = float(damping)
        self.importance: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.importance[name] = torch.zeros_like(param)
            self.anchor[name] = param.detach().clone()

    def accumulate(self, model: torch.nn.Module) -> None:
        """À appeler après CHAQUE backward du run de consolidation."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.importance[name] += param.grad.detach() ** 2

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Terme à ajouter à la loss de la phase LoRA (0 si rien n'est protégé)."""
        terms = []
        for name, param in model.named_parameters():
            weight = self.importance.get(name)
            if weight is None or float(weight.abs().sum()) == 0.0:
                continue  # jamais entraîné → libre
            terms.append((weight * (param - self.anchor[name]) ** 2).sum())
        if not terms:
            return torch.zeros((), dtype=torch.float32)
        return self.damping * torch.stack(terms).sum()

    def protected_threshold(self, model: torch.nn.Module, frac: float = 0.95) -> float:
        """Seuil des ``frac`` synapses les moins importantes (les autres protègent)."""
        chunks = [
            self.importance[name].flatten().abs()
            for name, _ in model.named_parameters()
            if name in self.importance
        ]
        if not chunks:
            return 0.0
        return float(torch.quantile(torch.cat(chunks), frac))

    def state_dict(self) -> dict:
        return {
            "damping": self.damping,
            "importance": {k: v.detach().cpu().clone() for k, v in self.importance.items()},
            "anchor": {k: v.detach().cpu().clone() for k, v in self.anchor.items()},
        }
