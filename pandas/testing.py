from __future__ import annotations


def assert_frame_equal(left, right):
    if not hasattr(left, "equals") or not hasattr(right, "equals"):
        raise AssertionError("Objects are not DataFrames")
    if not left.equals(right):
        raise AssertionError("DataFrames are not equal")
