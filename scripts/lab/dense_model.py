"""S8-FOCUS — construction partagée du dense, utilisée par entraînement ET chargement.

Un seul endroit connaît la correspondance config <-> DenseTransformer, donc
load_state_dict(strict=True) passe sans remappage ni devinette de forme.
"""

from __future__ import annotations

from micro_dense import DenseTransformer


def build_dense_from_config(vocab_size: int, config: dict) -> DenseTransformer:
    """Reconstruire un DenseTransformer depuis le dictionnaire ``config``
    sauvé dans un donor.pt (dense).

    ``config`` doit contenir : dim, layers, heads, mlp, seq.
    """
    return DenseTransformer(
        vocab=int(vocab_size),
        dim=int(config["dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        mlp=int(config["mlp"]),
        seq=int(config["seq"]),
    )


def load_dense_donor(path) -> DenseTransformer:
    """Charger un dense depuis un donor.pt et retourner le modèle en eval.

    Le checkpoint dense contient : state_dict, config, vocab_size, extra.
    """
    import torch

    blob = torch.load(str(path), weights_only=False)
    if blob.get("config", {}).get("type") != "dense":
        raise ValueError(f"{path} : config.type != 'dense'")
    model = build_dense_from_config(int(blob["vocab_size"]), blob["config"])
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model
