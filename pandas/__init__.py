from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

NA = None


class DType:
    def __init__(self, kind: str, name: Optional[str] = None):
        self.kind = kind
        self.name = name or kind

    def __eq__(self, other: object) -> bool:  # pragma: no cover - simple comparison
        if isinstance(other, DType):
            return self.kind == other.kind and self.name == other.name
        if isinstance(other, str):
            return self.name == other or self.kind == other
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return self.name


class Series:
    def __init__(self, data: List[Any], name: str):
        self.data = data
        self.name = name

    def __iter__(self):
        return iter(self.data)

    @property
    def dtype(self) -> DType:
        sample = next((v for v in self.data if v is not None), None)
        if isinstance(sample, datetime):
            return DType("M", "datetime64[ns]")
        if isinstance(sample, float):
            return DType("f", "float")
        if isinstance(sample, int):
            return DType("i", "int")
        return DType("O", "object")

    def copy(self) -> "Series":
        return Series(list(self.data), self.name)

    def __repr__(self) -> str:  # pragma: no cover - human-readable preview
        return f"Series(name={self.name}, data={self.data})"

    def apply(self, fn):
        return Series([fn(v) for v in self.data], self.name)

    def astype(self, type_name: str):
        casted: List[Any] = []
        for v in self.data:
            if v is None:
                casted.append(None)
                continue
            if type_name in ("Int64", "int", "int64"):
                try:
                    casted.append(int(v))
                except Exception:
                    casted.append(None)
            elif type_name in ("float", "float64", "Float64"):
                try:
                    casted.append(float(v))
                except Exception:
                    casted.append(None)
            else:
                casted.append(v)
        return Series(casted, self.name)

    def round(self):
        rounded = []
        for v in self.data:
            if v is None:
                rounded.append(None)
            else:
                rounded.append(round(float(v)))
        return Series(rounded, self.name)

    def __gt__(self, other: Any):
        return Series([None if v is None else v > other for v in self.data], self.name)

    def __lt__(self, other: Any):
        return Series([None if v is None else v < other for v in self.data], self.name)

    def dropna(self):
        return Series([v for v in self.data if v is not None], self.name)

    def notna(self):
        return Series([v is not None for v in self.data], self.name)

    def any(self):
        return any(self.data)

    def all(self):
        return all(self.data)

    @property
    def empty(self) -> bool:
        return len(self.data) == 0

    def min(self):
        vals = [v for v in self.data if v is not None]
        return min(vals) if vals else None

    def max(self):
        vals = [v for v in self.data if v is not None]
        return max(vals) if vals else None

    def nunique(self, dropna: bool = True):
        values = self.data if not dropna else [v for v in self.data if v is not None]
        return len(set(values))

    @property
    def iloc(self):
        return self

    def __getitem__(self, idx: int):
        return self.data[idx]


class DataFrame:
    def __init__(self, data: Iterable[Dict[str, Any]]):
        self._rows = [dict(row) for row in data]
        self._columns = list(self._rows[0].keys()) if self._rows else []

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def shape(self):
        return (len(self._rows), len(self._columns))

    def copy(self) -> "DataFrame":
        return DataFrame([dict(row) for row in self._rows])

    def __repr__(self) -> str:  # pragma: no cover - human-readable preview
        if not self._rows:
            return "Empty DataFrame"
        lines = [",".join(self._columns)]
        for row in self._rows[:5]:
            lines.append(",".join(str(row.get(col)) for col in self._columns))
        if len(self._rows) > 5:
            lines.append("...")
        return "\n".join(lines)

    def equals(self, other: "DataFrame") -> bool:
        return self._columns == other._columns and self._rows == other._rows

    def drop(self, columns: Optional[str] = None):
        cols = [columns] if isinstance(columns, str) else (columns or [])
        new_rows = []
        for row in self._rows:
            new_row = {k: v for k, v in row.items() if k not in cols}
            new_rows.append(new_row)
        return DataFrame(new_rows)

    def dropna(self, subset: List[str]):
        new_rows = []
        for row in self._rows:
            if all(row.get(col) is not None for col in subset):
                new_rows.append(row)
        return DataFrame(new_rows)

    def __getitem__(self, key):
        if isinstance(key, Series):
            mask = key.data
            filtered = [row for row, keep in zip(self._rows, mask) if keep]
            return DataFrame(filtered)
        if isinstance(key, list):
            subset_rows = []
            for row in self._rows:
                subset_rows.append({col: row.get(col) for col in key})
            return DataFrame(subset_rows)
        return Series([row.get(key) for row in self._rows], key)

    def __setitem__(self, key: str, value):
        if isinstance(value, Series):
            values = value.data
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value for _ in range(len(self._rows))]
        if not self._rows:
            for val in values:
                self._rows.append({key: val})
        else:
            for row, val in zip(self._rows, values):
                row[key] = val
        if key not in self._columns:
            self._columns.append(key)

    def duplicated(self, subset: List[str]):
        seen = set()
        flags = []
        for row in self._rows:
            key = tuple(row.get(col) for col in subset)
            flags.append(key in seen)
            seen.add(key)
        return Series(flags, "duplicated")

    def groupby(self, key: str):
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for row in self._rows:
            groups.setdefault(row.get(key), []).append(row)
        return GroupBy(groups)

    def sort_values(self, by: List[str], kind: str = "mergesort"):
        def sort_key(row):
            return tuple(row.get(col) for col in by)

        ordered = sorted(self._rows, key=sort_key)
        return DataFrame(ordered)

    def reset_index(self, drop: bool = False):
        return DataFrame(self._rows)

    def to_csv(self, path: Path | str, index: bool = False):
        path_obj = Path(path)
        if not self._rows:
            path_obj.write_text("")
            return
        with path_obj.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: _format_value(row.get(k)) for k in self._columns})

    def select_dtypes(self, include: List[str]):
        include_set = set(include)
        cols = []
        for col in self._columns:
            dtype_kind = self[col].dtype.kind
            if "number" in include_set:
                if dtype_kind in {"i", "f"}:
                    cols.append(col)
        return DataFrame([{c: row.get(c) for c in cols} for row in self._rows])

    def head(self, n: int = 5):
        return DataFrame(self._rows[:n])

    @property
    def empty(self) -> bool:
        return len(self._rows) == 0


class GroupBy:
    def __init__(self, groups: Dict[Any, List[Dict[str, Any]]]):
        self.groups = groups

    def __iter__(self):
        for key, rows in self.groups.items():
            yield key, DataFrame(rows)


def _format_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def read_csv(path: str, dtype: Optional[Dict[str, str]] = None, parse_dates: Optional[List[str]] = None):
    parse_dates = parse_dates or []
    dtype = dtype or {}
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, val in row.items():
                if val == "":
                    parsed[key] = None
                    continue
                if key in parse_dates:
                    try:
                        parsed[key] = datetime.fromisoformat(val)
                    except ValueError:
                        parsed[key] = None
                    continue
                desired = dtype.get(key)
                if desired:
                    try:
                        if desired.startswith("int"):
                            parsed[key] = int(float(val))
                        elif desired.startswith("float"):
                            parsed[key] = float(val)
                        elif desired == "string":
                            parsed[key] = str(val)
                        else:
                            parsed[key] = val
                    except Exception:
                        parsed[key] = None
                else:
                    parsed[key] = _coerce_value(val)
            rows.append(parsed)
    return DataFrame(rows)


def _coerce_value(val: str) -> Any:
    try:
        return int(val)
    except Exception:
        try:
            return float(val)
        except Exception:
            return val


def to_numeric(series: Series, errors: str = "raise"):
    coerced = []
    for v in series.data:
        if v is None:
            coerced.append(None)
            continue
        try:
            coerced.append(float(v))
        except Exception:
            if errors == "coerce":
                coerced.append(None)
            else:
                raise
    return Series(coerced, series.name)


def to_datetime(series: Series, errors: str = "raise"):
    parsed = []
    for v in series.data:
        if v is None:
            parsed.append(None)
            continue
        if isinstance(v, datetime):
            parsed.append(v)
            continue
        try:
            parsed.append(datetime.fromisoformat(str(v)))
        except Exception:
            text = str(v)
            for fmt in ("%H:%M", "%H:%M:%S"):
                try:
                    parsed.append(datetime.strptime(text, fmt))
                    break
                except Exception:
                    continue
            else:
                if errors == "coerce":
                    parsed.append(None)
                else:
                    raise
    return Series(parsed, series.name)


def isna(value: Any) -> bool:
    return value is None


class _APINamespace:
    def __init__(self):
        from . import api

        self.types = api.types


def _build_api():
    return _APINamespace()


api = _build_api()
from . import testing  # noqa: E402

__all__ = [
    "DataFrame",
    "Series",
    "read_csv",
    "to_numeric",
    "to_datetime",
    "isna",
    "NA",
    "api",
    "testing",
]
