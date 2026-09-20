"""Measured bits-per-weight for GGUF files (guardrail C6, RDR-008).

Nominal labels are not the real bit depth: token_embd is quantized to its own
type, and K-quants fall back to legacy types when a tensor's row length is not
divisible by 256. Never trust the file name.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, asdict
from pathlib import Path

from gguf import GGUFReader

EMBEDDING_TENSOR_SUBSTRINGS = ("token_embd", "output.weight")


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


@dataclass
class GGUFMeasurement:
    path: str
    nominal_label: str
    file_size_bytes: int
    total_params: int
    embed_params: int
    non_embed_params: int
    bpw_overall: float
    bpw_non_embed: float
    tensor_type_histogram: dict[str, int]
    has_kquant_fallback: bool
    fallback_types: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


# Legacy types: their presence means a K-quant request silently degraded.
_LEGACY_TYPES = {"Q4_0", "Q4_1", "Q5_0", "Q5_1"}


def measure_gguf(path: str | Path, nominal_label: str | None = None) -> GGUFMeasurement:
    """Measured bit depth from the tensor table.

    Bytes are summed from the tensor table, so container overhead is excluded;
    raw file size is reported separately.
    """
    path = Path(path)
    reader = GGUFReader(str(path))

    hist: collections.Counter[str] = collections.Counter()
    total_params = 0
    total_bytes = 0
    embed_params = 0
    embed_bytes = 0

    for tensor in reader.tensors:
        n = _numel(tensor.shape)
        type_name = tensor.tensor_type.name
        hist[type_name] += 1
        total_params += n
        total_bytes += int(tensor.n_bytes)
        if any(s in tensor.name for s in EMBEDDING_TENSOR_SUBSTRINGS):
            embed_params += n
            embed_bytes += int(tensor.n_bytes)

    non_embed_params = total_params - embed_params
    non_embed_bytes = total_bytes - embed_bytes

    fallback = sorted(t for t in hist if t in _LEGACY_TYPES)
    label = nominal_label or path.stem.split("_", 1)[-1]

    return GGUFMeasurement(
        path=str(path),
        nominal_label=label,
        file_size_bytes=path.stat().st_size,
        total_params=total_params,
        embed_params=embed_params,
        non_embed_params=non_embed_params,
        bpw_overall=total_bytes * 8 / total_params if total_params else float("nan"),
        bpw_non_embed=(
            non_embed_bytes * 8 / non_embed_params if non_embed_params else float("nan")
        ),
        tensor_type_histogram=dict(sorted(hist.items())),
        has_kquant_fallback=bool(fallback),
        fallback_types=fallback,
    )


# Approximate nominal bit depth, used only to report drift.
NOMINAL_BPW = {
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.8,
    "Q3_K_M": 3.9,
    "Q2_K": 3.0,
}


def nominal_drift(m: GGUFMeasurement) -> float | None:
    """measured_non_embed_bpw - nominal_non_embed_bpw, or None if unknown label."""
    nominal = NOMINAL_BPW.get(m.nominal_label)
    if nominal is None:
        return None
    return m.bpw_non_embed - nominal
