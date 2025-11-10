from __future__ import annotations
from ._dpe import dpe_mvs as _native  # type: ignore

__all__ = ["dpe_mvs"]

def dpe_mvs(
    dense_folder: str,
    gpu_index: int = 0,
    fusion: bool = True,
    viz: bool = False,
    weak: bool = False,
) -> int:
    """Run DPE-MVS pipeline from Python."""
    return _native(dense_folder, gpu_index, fusion=fusion, viz=viz, weak=weak)
