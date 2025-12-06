from __future__ import annotations

from .. import DataFrame, Series


def _get_dtype(obj):
    if isinstance(obj, Series):
        return obj.dtype
    if isinstance(obj, DataFrame):
        return None
    return getattr(obj, "dtype", None)


def is_datetime64_any_dtype(obj) -> bool:
    dtype = _get_dtype(obj)
    return getattr(dtype, "kind", None) == "M"


def is_object_dtype(obj) -> bool:
    dtype = _get_dtype(obj)
    return getattr(dtype, "kind", None) == "O"


def is_integer_dtype(obj) -> bool:
    dtype = _get_dtype(obj)
    return getattr(dtype, "kind", None) == "i"


def is_float_dtype(obj) -> bool:
    dtype = _get_dtype(obj)
    return getattr(dtype, "kind", None) == "f"
