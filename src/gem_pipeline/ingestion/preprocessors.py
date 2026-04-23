from __future__ import annotations
import pandas as pd
from loguru import logger


def preprocess_ghs(df: pd.DataFrame, survey_year: int = 2022) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    if "Q11YRBRTH" in df.columns:
        df["AGE_DERIVED"] = survey_year - pd.to_numeric(df["Q11YRBRTH"], errors="coerce")
        logger.debug("GHS: derived AGE_DERIVED from Q11YRBRTH")

    if "Q41ATNDSCH" in df.columns:
        raw = pd.to_numeric(df["Q41ATNDSCH"], errors="coerce")
        df["ATTEND_RECODE"] = raw.map({1: 1, 2: 0})
        logger.debug("GHS: recoded Q41ATNDSCH -> ATTEND_RECODE")

    if "Q44LITERACY" in df.columns:
        raw = pd.to_numeric(df["Q44LITERACY"], errors="coerce")
        df["LITERACY_RECODE"] = raw.map({1: 1, 2: 0})
        logger.debug("GHS: recoded Q44LITERACY -> LITERACY_RECODE")

    if "GEO_TYPE" in df.columns:
        raw = pd.to_numeric(df["GEO_TYPE"], errors="coerce")
        df["URBAN_RECODE"] = (raw == 1).astype("Int8")
        logger.debug("GHS: derived URBAN_RECODE from GEO_TYPE")

    return df
