"""Le lecteur GGUF : inventaire et déquantification d'une ligne réelle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mycelia.gguf_reader import (
    read_embedding_rows,
    read_inventory,
    tensor_data_start,
)

MODEL = Path(__file__).resolve().parent.parent / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="GGUF Qwen absent")


def test_inventory_matches_the_real_qwen() -> None:
    metadata, tensors = read_inventory(MODEL)
    assert metadata["general.architecture"] == "qwen2"
    assert metadata["qwen2.block_count"] == 24
    assert metadata["qwen2.embedding_length"] == 896
    assert len(tensors) == 291
    named = {t.name: t for t in tensors}
    assert named["token_embd.weight"].shape == (896, 151936)
    assert named["token_embd.weight"].type_name == "Q8_0"


def test_embedding_row_is_finite_and_right_shape() -> None:
    _metadata, tensors = read_inventory(MODEL)
    emb = {t.name: t for t in tensors}["token_embd.weight"]
    rows = read_embedding_rows(MODEL, emb, [0, 42, 1000])
    assert rows.shape == (3, 896)
    assert np.isfinite(rows).all()
    assert np.linalg.norm(rows[0]) > 0


def test_data_start_is_inside_the_file() -> None:
    start = tensor_data_start(MODEL)
    assert 0 < start < MODEL.stat().st_size
