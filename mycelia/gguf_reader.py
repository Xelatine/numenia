"""Lecteur GGUF minimal — en-tête, métadonnées, inventaire des tenseurs.

Pourquoi pas le paquet ``gguf`` : PyPI ne résout pas depuis BIO-1 au moment de
la greffe, et un lecteur de 150 lignes est vérifiable contre le fichier réel.
Il ne déquantifie rien : il lit les formes, les types et les offsets, ce qui
suffit à l'étape 1 (extraction et cartographie).

Format (GGUF v2/v3) :
  magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
  kv_count × (clé string, type u32, valeur)
  tensor_count × (nom string, n_dims u32, dims u64[n], type u32, offset u64)
  pad à 32 octets, puis données des tenseurs.

Les offsets sont relatifs au début des données, pas au début du fichier.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, asdict
from pathlib import Path

MAGIC = 0x46554747  # "GGUF" little-endian

# Types de métadonnées GGUF.
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, \
    UINT64, INT64, FLOAT64 = range(13)

# Types de tenseurs et taille d'un bloc (octets par valeur moyenne).
TENSOR_TYPES = {
    0: ("F32", 4.0),
    1: ("F16", 2.0),
    2: ("Q4_0", 18 / 32),
    3: ("Q4_1", 20 / 32),
    6: ("Q5_0", 22 / 32),
    7: ("Q5_1", 24 / 32),
    8: ("Q8_0", 34 / 32),
    9: ("Q8_1", 36 / 32),
    10: ("Q2_K", 84 / 256),
    11: ("Q3_K", 110 / 256),
    12: ("Q4_K", 144 / 256),
    13: ("Q5_K", 176 / 256),
    14: ("Q6_K", 210 / 256),
    15: ("Q8_K", 292 / 256),
}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    type_id: int
    type_name: str
    offset: int
    n_elements: int
    n_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


class _Reader:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("rb")

    def read(self, n: int) -> bytes:
        data = self.handle.read(n)
        if len(data) != n:
            raise EOFError(f"GGUF tronqué : {n} octets demandés, {len(data)} lus")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, kind: int):
        if kind == UINT8:
            return self.read(1)[0]
        if kind == INT8:
            return struct.unpack("<b", self.read(1))[0]
        if kind == UINT16:
            return struct.unpack("<H", self.read(2))[0]
        if kind == INT16:
            return struct.unpack("<h", self.read(2))[0]
        if kind == UINT32:
            return self.u32()
        if kind == INT32:
            return self.i32()
        if kind == FLOAT32:
            return struct.unpack("<f", self.read(4))[0]
        if kind == BOOL:
            return bool(self.read(1)[0])
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item_kind = self.u32()
            length = self.u64()
            return [self.value(item_kind) for _ in range(length)]
        if kind == UINT64:
            return self.u64()
        if kind == INT64:
            return struct.unpack("<q", self.read(8))[0]
        if kind == FLOAT64:
            return struct.unpack("<d", self.read(8))[0]
        raise ValueError(f"type de métadonnée GGUF inconnu : {kind}")


def read_inventory(path: str | Path) -> tuple[dict, list[TensorInfo]]:
    """Renvoie (métadonnées, tenseurs) sans charger les données."""
    reader = _Reader(Path(path))
    magic = reader.u32()
    if magic != MAGIC:
        raise ValueError(f"pas un GGUF : magic {magic:#x}")
    version = reader.u32()
    tensor_count = reader.u64()
    kv_count = reader.u64()

    metadata: dict = {"version": version}
    for _ in range(kv_count):
        key = reader.string()
        kind = reader.u32()
        metadata[key] = reader.value(kind)

    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        shape = tuple(reader.u64() for _ in range(n_dims))
        type_id = reader.u32()
        offset = reader.u64()
        type_name, bytes_per_value = TENSOR_TYPES.get(type_id, (f"UNKNOWN{type_id}", 0.0))
        n_elements = 1
        for dim in shape:
            n_elements *= int(dim)
        tensors.append(TensorInfo(
            name=name,
            shape=shape,
            type_id=type_id,
            type_name=type_name,
            offset=offset,
            n_elements=n_elements,
            n_bytes=int(round(n_elements * bytes_per_value)),
        ))
    return metadata, tensors


def tensor_data_start(path: str | Path) -> int:
    """Offset absolu du début des données, après l'alignement à 32 octets."""
    reader = _Reader(Path(path))
    magic = reader.u32()
    if magic != MAGIC:
        raise ValueError(f"pas un GGUF : magic {magic:#x}")
    reader.u32()
    tensor_count = reader.u64()
    kv_count = reader.u64()
    for _ in range(kv_count):
        reader.string()
        kind = reader.u32()
        reader.value(kind)
    for _ in range(tensor_count):
        reader.string()
        n_dims = reader.u32()
        for _ in range(n_dims):
            reader.u64()
        reader.u32()
        reader.u64()
    position = reader.handle.tell()
    return position + ((32 - position % 32) % 32)


def _dequant_q8_0(block: bytes) -> "list[float]":
    import struct

    scale = struct.unpack("<e", block[:2])[0]
    values = struct.unpack("<32b", block[2:34])
    return [scale * value for value in values]


def read_embedding_rows(path: str | Path, tensor: TensorInfo, rows: list[int]):
    """Lit des lignes (tokens) d'une table d'embedding quantifiée.

    GGUF stocke ne0 (dimension du modèle) en premier : une ligne de token est
    contiguë et se déquantifie par blocs de 32. Renvoie un array numpy
    [len(rows), ne0] en float32.
    """
    import numpy as np

    if len(tensor.shape) != 2:
        raise ValueError(f"tenseur non 2-D : {tensor.shape}")
    ne0, ne1 = int(tensor.shape[0]), int(tensor.shape[1])
    if tensor.type_id not in (0, 1, 8):
        raise ValueError(f"type non géré pour la lecture : {tensor.type_name}")
    row_bytes = tensor.n_bytes // ne1
    start = tensor_data_start(path) + tensor.offset
    out = np.zeros((len(rows), ne0), dtype=np.float32)
    with Path(path).open("rb") as handle:
        for index, token in enumerate(rows):
            if not 0 <= int(token) < ne1:
                raise IndexError(f"token {token} hors [0,{ne1})")
            handle.seek(start + int(token) * row_bytes)
            raw = handle.read(row_bytes)
            if tensor.type_id == 8:
                blocks = ne0 // 32
                values: list[float] = []
                for block in range(blocks):
                    values.extend(_dequant_q8_0(raw[block * 34:(block + 1) * 34]))
                out[index] = np.asarray(values, dtype=np.float32)
            elif tensor.type_id == 1:
                out[index] = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            else:
                out[index] = np.frombuffer(raw, dtype=np.float32)
    return out


def summary(metadata: dict, tensors: list[TensorInfo]) -> dict:
    types: dict[str, int] = {}
    for tensor in tensors:
        types[tensor.type_name] = types.get(tensor.type_name, 0) + 1
    named = {t.name: t for t in tensors}
    wanted = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    ]
    layers = {}
    for layer in range(int(metadata.get("qwen2.block_count", 0))):
        prefix = f"blk.{layer}."
        layers[layer] = {
            key: {
                "shape": named[prefix + key].shape,
                "type": named[prefix + key].type_name,
            }
            for key in (
                "attn_norm.weight", "attn_q.weight", "attn_k.weight",
                "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            )
            if prefix + key in named
        }
    return {
        "version": metadata.get("version"),
        "architecture": metadata.get("general.architecture"),
        "n_layer": metadata.get("qwen2.block_count"),
        "n_embd": metadata.get("qwen2.embedding_length"),
        "n_head": metadata.get("qwen2.attention.head_count"),
        "n_head_kv": metadata.get("qwen2.attention.head_count_kv"),
        "n_fft": metadata.get("qwen2.feed_forward_length"),
        "n_vocab": metadata.get("qwen2.vocab_size") or metadata.get("tokenizer.ggml.tokens") and len(metadata["tokenizer.ggml.tokens"]),
        "tensor_count": len(tensors),
        "types": types,
        "tensors": [t.as_dict() for t in tensors],
        "global": {name: named[name].as_dict() for name in wanted if name in named},
        "layers": layers,
    }
