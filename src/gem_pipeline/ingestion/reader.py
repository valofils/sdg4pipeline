from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
from loguru import logger

try:
    import pyreadstat
    _PYREADSTAT = True
except ImportError:
    _PYREADSTAT = False
    logger.warning("pyreadstat not installed — Stata/SPSS support disabled.")


@dataclass
class SurveyMeta:
    source_path: Path
    file_format: str
    n_rows: int
    n_cols: int
    variable_labels: dict[str, str] = field(default_factory=dict)
    value_labels: dict[str, dict[Any, str]] = field(default_factory=dict)

    def __repr__(self):
        return f"SurveyMeta(source={self.source_path.name}, rows={self.n_rows:,}, cols={self.n_cols})"


def load_survey(path, file_format, columns=None, encoding="utf-8"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Survey file not found: {path}")
    fmt = file_format.lower().strip(".")
    loaders = {"dta": _load_stata, "sav": _load_spss, "csv": _load_csv, "parquet": _load_parquet}
    if fmt not in loaders:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from: {list(loaders)}")
    logger.info(f"Loading {fmt.upper()}: {path.name}")
    df, meta = loaders[fmt](path, columns=columns, encoding=encoding)
    df.columns = [c.strip().lower() for c in df.columns]
    logger.info(f"Loaded {meta.n_rows:,} rows x {meta.n_cols} cols")
    return df, meta


def _load_stata(path, columns=None, **_):
    if not _PYREADSTAT:
        raise ImportError("pip install pyreadstat")
    # Try utf-8 first, fall back to latin1 (needed for Stats SA GHS files)
    for enc in [None, "latin1", "cp1252"]:
        try:
            kwargs = {"usecols": columns, "apply_value_formats": False}
            if enc:
                kwargs["encoding"] = enc
            df, m = pyreadstat.read_dta(str(path), **kwargs)
            if enc:
                logger.debug("Loaded DTA with encoding=" + enc)
            meta = SurveyMeta(path, "dta", len(df), len(df.columns),
                              dict(zip(m.column_names, m.column_labels)), m.value_labels)
            return df, meta
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1,
        "Could not decode DTA file with utf-8, latin1, or cp1252")


def _load_spss(path, columns=None, **_):
    if not _PYREADSTAT:
        raise ImportError("pip install pyreadstat")
    df, m = pyreadstat.read_sav(str(path), usecols=columns, apply_value_formats=False)
    meta = SurveyMeta(path, "sav", len(df), len(df.columns),
                      dict(zip(m.column_names, m.column_labels)), m.value_labels)
    return df, meta


def _load_csv(path, columns=None, encoding="utf-8", **_):
    df = pd.read_csv(path, usecols=columns, encoding=encoding, low_memory=False)
    return df, SurveyMeta(path, "csv", len(df), len(df.columns))


def _load_parquet(path, columns=None, **_):
    df = pd.read_parquet(path, columns=columns)
    return df, SurveyMeta(path, "parquet", len(df), len(df.columns))


def list_variables(meta, search=None):
    df = pd.DataFrame([{"variable": k, "label": v} for k, v in meta.variable_labels.items()])
    if search and not df.empty:
        mask = (df["variable"].str.contains(search, case=False, na=False) |
                df["label"].str.contains(search, case=False, na=False))
        df = df[mask]
    return df.reset_index(drop=True)
