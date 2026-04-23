from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Indicator code mappings
# ---------------------------------------------------------------------------

WIDE_STEM = {
    ("oosr",       "primary"):          "oos_prim",
    ("oosr",       "lower_secondary"):  "oos_lowsec",
    ("oosr",       "upper_secondary"):  "oos_upsec",
    ("attendance", "primary"):          "att_prim",
    ("attendance", "lower_secondary"):  "att_lowsec",
    ("attendance", "upper_secondary"):  "att_upsec",
    ("completion", "primary"):          "comp_prim",
    ("completion", "lower_secondary"):  "comp_lowsec",
    ("completion", "upper_secondary"):  "comp_upsec",
    ("literacy",   "youth_15_24"):      "lit_youth",
    ("literacy",   "adult_15plus"):     "lit_adult",
    ("repetition", "primary"):          "rep_prim",
}

WIDE_SUFFIX = {
    ("total",  "total"):   "t",
    ("sex",    "male"):    "m",
    ("sex",    "female"):  "f",
    ("urban",  "urban"):   "u",
    ("urban",  "rural"):   "r",
    ("wealth", "q1"):      "w1",
    ("wealth", "q2"):      "w2",
    ("wealth", "q3"):      "w3",
    ("wealth", "q4"):      "w4",
    ("wealth", "q5"):      "w5",
}

SCOPE_CODE = {
    ("oosr",       "primary"):          "OOSR.PRIM",
    ("oosr",       "lower_secondary"):  "OOSR.LOWSEC",
    ("oosr",       "upper_secondary"):  "OOSR.UPSEC",
    ("attendance", "primary"):          "ATT.PRIM",
    ("attendance", "lower_secondary"):  "ATT.LOWSEC",
    ("attendance", "upper_secondary"):  "ATT.UPSEC",
    ("completion", "primary"):          "COMP.PRIM",
    ("completion", "lower_secondary"):  "COMP.LOWSEC",
    ("completion", "upper_secondary"):  "COMP.UPSEC",
    ("literacy",   "youth_15_24"):      "LIT.YOUTH",
    ("literacy",   "adult_15plus"):     "LIT.ADULT",
    ("repetition", "primary"):          "REP.PRIM",
}

SUBGROUP_TYPE_LABEL = {
    "total":     "Total",
    "sex":       "Sex",
    "urban":     "Location",
    "wealth":    "Wealth",
    "ethnicity": "Ethnicity",
}

SUBGROUP_VALUE_LABEL = {
    ("total",  "total"):   "Total",
    ("sex",    "male"):    "Male",
    ("sex",    "female"):  "Female",
    ("urban",  "urban"):   "Urban",
    ("urban",  "rural"):   "Rural",
    ("wealth", "q1"):      "Quintile 1 (poorest)",
    ("wealth", "q2"):      "Quintile 2",
    ("wealth", "q3"):      "Quintile 3",
    ("wealth", "q4"):      "Quintile 4",
    ("wealth", "q5"):      "Quintile 5 (richest)",
}

INDICATOR_LABEL = {
    ("oosr",       "primary"):          "Out-of-school rate, primary",
    ("oosr",       "lower_secondary"):  "Out-of-school rate, lower secondary",
    ("oosr",       "upper_secondary"):  "Out-of-school rate, upper secondary",
    ("attendance", "primary"):          "Attendance rate, primary",
    ("attendance", "lower_secondary"):  "Attendance rate, lower secondary",
    ("attendance", "upper_secondary"):  "Attendance rate, upper secondary",
    ("completion", "primary"):          "Completion rate, primary",
    ("completion", "lower_secondary"):  "Completion rate, lower secondary",
    ("completion", "upper_secondary"):  "Completion rate, upper secondary",
    ("literacy",   "youth_15_24"):      "Literacy rate, youth (15-24)",
    ("literacy",   "adult_15plus"):     "Literacy rate, adults (15+)",
    ("repetition", "primary"):          "Repetition rate (proxy), primary",
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def to_wide(results, cfg, output_path=None):
    """
    Convert long-format pipeline results to WIDE flat format.

    One row per subgroup combination; indicator columns follow the GEM
    naming convention (e.g. oos_prim_t, comp_lowsec_w1, lit_youth_f).
    Values are expressed as percentages (0-100).
    """
    if results is None or (hasattr(results, "empty") and results.empty):
        return pd.DataFrame()

    rows = {}

    for _, row in results.iterrows():
        ind   = str(row["indicator"])
        level = str(row["level"])
        group = str(row["group"])
        gval  = str(row["group_value"])
        est   = float(row["estimate"])
        n     = int(row["n_unweighted"])

        stem = WIDE_STEM.get((ind, level))
        if stem is None:
            continue

        suffix = WIDE_SUFFIX.get((group, gval))
        if suffix is None:
            if group == "ethnicity":
                suffix = "eth_" + gval[:8].replace(" ", "_").lower()
            else:
                continue

        col_name = stem + "_" + suffix
        n_col    = stem + "_" + suffix + "_n"
        key      = (group, gval)

        if key not in rows:
            rows[key] = {
                "country_code":      cfg.country_iso3,
                "country_name":      cfg.country_name,
                "survey_name":       cfg.survey_name,
                "survey_type":       cfg.survey_type,
                "year":              cfg.survey_year,
                "subgroup_type":     SUBGROUP_TYPE_LABEL.get(group, group),
                "subgroup_value":    SUBGROUP_VALUE_LABEL.get((group, gval), gval),
                "generated_date":    str(date.today()),
                "pipeline_version":  __version__,
            }
        rows[key][col_name] = round(est * 100, 2)
        rows[key][n_col]    = n

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(list(rows.values()))
    meta_cols = [
        "country_code", "country_name", "survey_name", "survey_type", "year",
        "subgroup_type", "subgroup_value", "generated_date", "pipeline_version",
    ]
    ind_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + ind_cols].reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def to_scope(results, cfg, output_path=None):
    """
    Convert long-format pipeline results to SCOPE long format.

    One row per estimate: country x survey x indicator x subgroup.
    Uses UIS-style indicator codes (OOSR.PRIM, COMP.LOWSEC, LIT.YOUTH ...).
    Values are expressed as percentages (0-100).
    """
    if results is None or (hasattr(results, "empty") and results.empty):
        return pd.DataFrame()

    scope_rows = []

    for _, row in results.iterrows():
        ind   = str(row["indicator"])
        level = str(row["level"])
        group = str(row["group"])
        gval  = str(row["group_value"])

        indicator_code = SCOPE_CODE.get((ind, level))
        if indicator_code is None:
            continue

        scope_rows.append({
            "country_code":     cfg.country_iso3,
            "country_name":     cfg.country_name,
            "survey_name":      cfg.survey_name,
            "survey_type":      cfg.survey_type,
            "year":             cfg.survey_year,
            "indicator_code":   indicator_code,
            "indicator_label":  INDICATOR_LABEL.get((ind, level), ind + ", " + level),
            "subgroup_type":    SUBGROUP_TYPE_LABEL.get(group, group),
            "subgroup_value":   SUBGROUP_VALUE_LABEL.get((group, gval), gval),
            "value":            round(float(row["estimate"]) * 100, 4),
            "n_unweighted":     int(row["n_unweighted"]),
            "generated_date":   str(date.today()),
            "pipeline_version": __version__,
        })

    if not scope_rows:
        return pd.DataFrame()

    df = pd.DataFrame(scope_rows)
    sort_cols = [c for c in
                 ["country_code", "year", "indicator_code", "subgroup_type", "subgroup_value"]
                 if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def to_both(results, cfg, output_dir):
    """Generate both WIDE and SCOPE exports, saving to output_dir."""
    out = Path(output_dir)
    wide_df  = to_wide(results,  cfg, out / "wide_export.csv")
    scope_df = to_scope(results, cfg, out / "scope_export.csv")
    return wide_df, scope_df
