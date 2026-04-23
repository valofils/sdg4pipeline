from __future__ import annotations

"""
ingestion/registry.py
----------------------
Registry that maps survey_name patterns to preprocessor functions.

To add a new survey preprocessor:
  1. Write the function in ingestion/preprocessors.py
  2. Add an entry to PREPROCESSOR_REGISTRY below

The key is matched case-insensitively against cfg.survey_name.
"""

from typing import Callable
import pandas as pd


def _ghs_preprocessor(df: pd.DataFrame, cfg) -> pd.DataFrame:
    from pathlib import Path as _Path
    from gem_pipeline.ingestion.preprocessors import preprocess_ghs
    file_p = _Path(str(cfg.file_path))
    candidates = [
        file_p.parent / file_p.name.replace("person", "hhold"),
        file_p.parent / "ghs-2022-hhold-v1.dta",
        file_p.parent / "ghs-2022-hhold-v1.csv",
    ]
    hhold_path = next((h for h in candidates if h.exists()), None)
    if hhold_path:
        from loguru import logger
        logger.info("GHS: using household file " + hhold_path.name)
    df = preprocess_ghs(df, survey_year=cfg.survey_year, hhold_path=hhold_path)
    df.columns = [c.lower() for c in df.columns]
    return df


def _mics_preprocessor(df: pd.DataFrame, cfg) -> pd.DataFrame:
    from gem_pipeline.ingestion.preprocessors import preprocess_mics
    df = preprocess_mics(df, survey_year=cfg.survey_year)
    df.columns = [c.lower() for c in df.columns]
    return df


PREPROCESSOR_REGISTRY: dict[str, Callable] = {
    "ghs":  _ghs_preprocessor,
    "mics": _mics_preprocessor,
}


def get_preprocessor(survey_name: str) -> Callable | None:
    """Return the preprocessor for a survey name, or None if not registered."""
    name_lower = survey_name.lower()
    for key, fn in PREPROCESSOR_REGISTRY.items():
        if key in name_lower:
            return fn
    return None
