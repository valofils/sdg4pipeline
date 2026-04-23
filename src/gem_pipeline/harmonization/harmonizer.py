from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger
from gem_pipeline.config_loader import CountryConfig


def harmonize(df: pd.DataFrame, cfg: CountryConfig) -> pd.DataFrame:
    df = df.copy()
    required = {cfg.age_var, cfg.sex_var, cfg.weight_var, cfg.school_attendance_var}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"[{cfg.survey_id}] Required variables missing: {missing}")

    df["_age"] = pd.to_numeric(df[cfg.age_var], errors="coerce").astype("Int16")
    sex_raw = pd.to_numeric(df[cfg.sex_var], errors="coerce")
    if sex_raw.dropna().isin([0, 1]).all():
        df["_sex"] = sex_raw.map({0: 1, 1: 2}).astype("Int8")
    else:
        df["_sex"] = sex_raw.astype("Int8")
    w = pd.to_numeric(df[cfg.weight_var], errors="coerce")
    n_bad = ((w <= 0) | w.isna()).sum()
    if n_bad:
        logger.warning(f"[{cfg.survey_id}] {n_bad:,} bad weights")
    df["_weight"] = w
    df["_strata"] = df[cfg.strata_var].astype(str) if cfg.strata_var and cfg.strata_var in df.columns else np.nan
    df["_psu"] = df[cfg.psu_var].astype(str) if cfg.psu_var and cfg.psu_var in df.columns else np.nan
    raw = pd.to_numeric(df[cfg.school_attendance_var], errors="coerce")
    df["_attending"] = (raw == cfg.school_attendance_value_yes).astype("Int8")
    df.loc[raw.isna(), "_attending"] = pd.NA
    df["_highest_grade"] = pd.to_numeric(df[cfg.highest_grade_var], errors="coerce").astype("Int16") if cfg.highest_grade_var and cfg.highest_grade_var in df.columns else pd.NA
    df["_highest_level"] = pd.to_numeric(df[cfg.highest_level_var], errors="coerce").astype("Int8") if cfg.highest_level_var and cfg.highest_level_var in df.columns else pd.NA
    if cfg.literacy_var and cfg.literacy_var in df.columns:
        lit = pd.to_numeric(df[cfg.literacy_var], errors="coerce")
        df["_literate"] = (lit == cfg.literacy_value_literate).astype("Int8")
        df.loc[lit.isna(), "_literate"] = pd.NA
    else:
        df["_literate"] = pd.NA
    if cfg.urban_rural_var and cfg.urban_rural_var in df.columns:
        urb = pd.to_numeric(df[cfg.urban_rural_var], errors="coerce")
        df["_urban"] = (urb == cfg.urban_value).astype("Int8")
        df.loc[urb.isna(), "_urban"] = pd.NA
    else:
        df["_urban"] = pd.NA
    if cfg.wealth_quintile_var and cfg.wealth_quintile_var in df.columns:
        wlth = pd.to_numeric(df[cfg.wealth_quintile_var], errors="coerce")
        df["_wealth"] = wlth.where(wlth.between(1, 5)).astype("Int8")
    else:
        df["_wealth"] = pd.NA
    df["_ethnicity"] = df[cfg.ethnicity_var].astype(str) if cfg.ethnicity_var and cfg.ethnicity_var in df.columns else pd.NA
    if cfg.disability_var and cfg.disability_var in df.columns:
        dis = pd.to_numeric(df[cfg.disability_var], errors="coerce")
        df["_disability"] = (dis == 1).astype("Int8")
        df.loc[dis.isna(), "_disability"] = pd.NA
    else:
        df["_disability"] = pd.NA
    logger.debug(f"[{cfg.survey_id}] Harmonization complete")
    return df
