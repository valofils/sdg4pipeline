from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger
from gem_pipeline.config_loader import CountryConfig

DISAGG_VARS = {"total": None, "sex": "_sex", "urban": "_urban", "wealth": "_wealth", "ethnicity": "_ethnicity"}
SEX_LABELS = {1: "male", 2: "female"}
URBAN_LABELS = {1: "urban", 0: "rural"}


def compute_all_indicators(df: pd.DataFrame, cfg: CountryConfig) -> pd.DataFrame:
    rows = []
    dispatch = {"oosr": _oosr, "attendance": _attendance, "completion": _completion,
                "literacy": _literacy, "repetition": _repetition}
    for ind in cfg.indicators:
        if ind not in dispatch:
            continue
        logger.info(f"[{cfg.survey_id}] Computing: {ind}")
        rows.extend(dispatch[ind](df, cfg))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["survey_id"] = cfg.survey_id
    out["country_iso3"] = cfg.country_iso3
    out["survey_year"] = cfg.survey_year
    out["estimate"] = out["estimate"].round(4)
    return out


def _weighted_mean(outcome, weight):
    valid = outcome.notna() & weight.notna() & (weight > 0)
    if valid.sum() == 0:
        return None
    return float(np.average(outcome[valid].astype(float), weights=weight[valid].astype(float)))


def _make_row(indicator, level, group, group_value, estimate, n):
    return {"indicator": indicator, "level": level, "group": group,
            "group_value": group_value, "estimate": estimate, "n_unweighted": n}


def _disaggregate(df, outcome_col, indicator, level, cfg):
    rows = []
    for group, col in DISAGG_VARS.items():
        if col is None:
            est = _weighted_mean(df[outcome_col], df["_weight"])
            if est is not None:
                rows.append(_make_row(indicator, level, group, "total", est, len(df)))
        else:
            if df[col].isna().all():
                continue
            for val in sorted(df[col].dropna().unique()):
                sub = df[df[col] == val]
                if len(sub) < 5:
                    continue
                est = _weighted_mean(sub[outcome_col], sub["_weight"])
                if est is None:
                    continue
                if col == "_sex":
                    label = SEX_LABELS.get(int(val), str(val))
                elif col == "_urban":
                    label = URBAN_LABELS.get(int(val), str(val))
                elif col == "_wealth":
                    label = f"q{int(val)}"
                else:
                    label = str(val)
                rows.append(_make_row(indicator, level, group, label, est, len(sub)))
    return rows


def _oosr(df, cfg):
    rows = []
    for level, (a_min, a_max) in cfg.age_bands.items():
        sub = df[df["_age"].between(a_min, a_max) & df["_attending"].notna() & df["_weight"].notna()].copy()
        if len(sub) < 10:
            continue
        sub["_not_attending"] = 1 - sub["_attending"]
        rows.extend(_disaggregate(sub, "_not_attending", "oosr", level, cfg))
    return rows


def _attendance(df, cfg):
    rows = []
    for level, (a_min, a_max) in cfg.age_bands.items():
        sub = df[df["_age"].between(a_min, a_max) & df["_attending"].notna() & df["_weight"].notna()].copy()
        if len(sub) < 10:
            continue
        rows.extend(_disaggregate(sub, "_attending", "attendance", level, cfg))
    return rows


def _completion(df, cfg):
    rows = []
    specs = {
        "primary": {"age_min": cfg.age_primary_max+3, "age_max": cfg.age_primary_max+5,
                    "min_grade": cfg.age_primary_max - cfg.age_primary_min + 1},
        "lower_secondary": {"age_min": cfg.age_lower_secondary_max+3, "age_max": cfg.age_lower_secondary_max+5,
                            "min_grade": (cfg.age_primary_max-cfg.age_primary_min+1)+(cfg.age_lower_secondary_max-cfg.age_lower_secondary_min+1)},
        "upper_secondary": {"age_min": cfg.age_upper_secondary_max+3, "age_max": cfg.age_upper_secondary_max+5,
                            "min_grade": (cfg.age_primary_max-cfg.age_primary_min+1)+(cfg.age_lower_secondary_max-cfg.age_lower_secondary_min+1)+(cfg.age_upper_secondary_max-cfg.age_upper_secondary_min+1)},
    }
    if df["_highest_grade"].isna().all():
        return rows
    for level, spec in specs.items():
        sub = df[df["_age"].between(spec["age_min"], spec["age_max"]) & df["_highest_grade"].notna() & df["_weight"].notna()].copy()
        if len(sub) < 10:
            continue
        sub["_completed"] = (sub["_highest_grade"] >= spec["min_grade"]).astype(int)
        rows.extend(_disaggregate(sub, "_completed", "completion", level, cfg))
    return rows


def _literacy(df, cfg):
    rows = []
    if df["_literate"].isna().all():
        return rows
    for level, (a_min, a_max) in [("youth_15_24", (15, 24)), ("adult_15plus", (15, 99))]:
        sub = df[df["_age"].between(a_min, a_max) & df["_literate"].notna() & df["_weight"].notna()].copy()
        if len(sub) < 10:
            continue
        rows.extend(_disaggregate(sub, "_literate", "literacy", level, cfg))
    return rows


def _repetition(df, cfg):
    a_min, a_max = cfg.age_bands["primary"]
    sub = df[df["_age"].between(a_min, a_max) & df["_attending"].eq(1) & df["_highest_grade"].notna() & df["_weight"].notna()].copy()
    if len(sub) < 10:
        return []
    expected = sub["_age"] - (a_min - 1)
    sub["_repeating"] = (sub["_highest_grade"] < (expected - 1)).astype(int)
    return _disaggregate(sub, "_repeating", "repetition", "primary", cfg)
