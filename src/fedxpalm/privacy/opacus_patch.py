"""Opacus's `DPDataLoader` assumes each dataset item is a flat sequence of
tensors (e.g. an `(image, label)` tuple) when it precomputes the zero-shaped
placeholder tensors Poisson sampling needs for an empty batch
(`sample_empty_shapes`/`dtypes` in `opacus/data_loader.py`, derived via
`for x in dataset[0]`). Ultralytics' `YOLODataset.__getitem__` returns a
`dict`, so that loop iterates over dict *keys* (strings) instead of values,
and `dtype_safe("img")` returns Python's `str` type where a `torch.dtype` is
expected -- `torch.zeros(shape, dtype=str)` then raises
`TypeError: zeros() received an invalid combination of arguments` the moment
Poisson sampling ever draws an empty batch. That is rare for any single step
but likely at least once over a full K x sigma sweep of thousands of steps --
exactly the kind of failure that survives a short smoke test untouched and
only surfaces hours into an unattended E1/E2 run.

`patch_opacus_for_dict_datasets()` makes `DPDataLoader.__init__` special-case
dict-returning datasets: shapes/dtypes are inferred from `dataset[0].values()`
(keys preserved), and the empty-batch placeholder is rebuilt as a dict with
those same keys instead of Opacus' bare list. Any dataset whose `__getitem__`
does not return a dict falls through to Opacus' own unmodified implementation.
"""
from __future__ import annotations

import torch
from opacus.data_loader import (
    DistributedUniformWithReplacementSampler,
    DPDataLoader,
    UniformWithReplacementSampler,
    dtype_safe,
    shape_safe,
)
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

_original_init = DPDataLoader.__init__
_patched = False


def _dict_safe_init(
    self,
    dataset,
    *,
    sample_rate,
    collate_fn=None,
    drop_last=False,
    generator=None,
    distributed=False,
    **kwargs,
):
    first = dataset[0]
    if not isinstance(first, dict):
        _original_init(
            self, dataset, sample_rate=sample_rate, collate_fn=collate_fn,
            drop_last=drop_last, generator=generator, distributed=distributed, **kwargs,
        )
        return

    self.sample_rate = sample_rate
    self.distributed = distributed

    if distributed:
        batch_sampler = DistributedUniformWithReplacementSampler(
            total_size=len(dataset), sample_rate=sample_rate, generator=generator,
        )
    else:
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataset), sample_rate=sample_rate, generator=generator,
        )

    # Only tensor-valued keys ("img", "cls", "bboxes", "batch_idx", ...) get a
    # zero-shaped tensor placeholder -- that mirrors what YOLODataset.collate_fn
    # itself does for those keys. Passthrough metadata keys ("im_file",
    # "ori_shape", "ratio_pad", ...) are plain str/tuple per sample and are never
    # stacked into tensors even in a real (non-empty) batch, so an empty tuple is
    # the correct empty-batch placeholder for them, not `torch.zeros(shape, dtype=str)`.
    keys = list(first.keys())
    is_tensor = [isinstance(v, torch.Tensor) for v in first.values()]
    sample_empty_shapes = [(0, *shape_safe(v)) if t else None for v, t in zip(first.values(), is_tensor)]
    dtypes = [dtype_safe(v) if t else None for v, t in zip(first.values(), is_tensor)]
    if collate_fn is None:
        collate_fn = default_collate

    def _dict_collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        return {
            k: torch.zeros(shape, dtype=dtype) if t else ()
            for k, t, shape, dtype in zip(keys, is_tensor, sample_empty_shapes, dtypes)
        }

    DataLoader.__init__(
        self,
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=_dict_collate,
        generator=generator,
        **kwargs,
    )


def patch_opacus_for_dict_datasets() -> None:
    global _patched
    if _patched:
        return
    DPDataLoader.__init__ = _dict_safe_init
    _patched = True
