from __future__ import annotations
from ._dpe import dpe_mvs as _native  # type: ignore

__all__ = ["dpe_mvs"]

def dpe_mvs(
    dense_folder: str,
    gpu_index: int = 0,
    verbose: bool = True,
    fusion: bool = False,
    viz: bool = False,
    depth: bool = True,
    normal: bool = False,
    weak: bool = False,
    edge: bool = False,
) -> int:
    """Run DPE-MVS pipeline from Python."""
    return _native(dense_folder, gpu_index, verbose, fusion, viz, depth, normal, weak, edge)