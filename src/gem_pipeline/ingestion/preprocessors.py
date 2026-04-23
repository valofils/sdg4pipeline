from __future__ import annotations

"""
ingestion/preprocessors.py
---------------------------
Survey-specific pre-processing functions applied BEFORE harmonization.

GHS 2022 (Stats SA / DataFirst ZAF-STATSSA-GHS-2022-V1)
---------------------------------------------------------
The real GHS 2022 person file (ghs-2022-person-v1.dta) uses:
  - age          : completed years (direct — no derivation needed)
  - Sex          : 1=Male, 2=Female
  - edu_attend   : 1=Yes, 2=No, 3=Other, 8=Not applicable
  - education    : highest level 0-29, 98=unspecified, 99=DK
                   0=No schooling, 1-12=Grade 1-12, 13+=post-secondary
  - edu_grde     : current grade (for attendees), 88=not applicable
  - population   : 1=Black African, 2=Coloured, 3=Indian/Asian, 4=White
  - geotype      : 1=Urban, 2=Traditional/Tribal, 3=Farm
  - stratum      : design stratum (string)
  - psu          : primary sampling unit
  - person_wgt   : person-level sampling weight
  - disab        : 0=no disability, 1=has disability

No wealth quintile exists in the file — a proxy is derived from
household assets in the household file (ghs-2022-hhold-v1.dta),
merged on uqnr (unique household number).

Education code -> grade mapping:
  0 = no schooling  -> grade 0
  1-12 = Grade 1-12 -> grade 1-12
  13+  = post-sec   -> grade 12 (treated as completed secondary)
  98/99             -> NaN
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# GHS education code -> completed grade
# ---------------------------------------------------------------------------

EDU_CODE_TO_GRADE: dict[int, int] = {
    0: 0,                           # No schooling
    **{i: i for i in range(1, 13)}, # Grade 1–12
    **{i: 12 for i in range(13, 30)},  # Post-secondary -> treat as Gr12
}


# ---------------------------------------------------------------------------
# Wealth index asset columns (1=owns, 2=does not own, 9=unknown)
# Selected to maximise discriminatory power across the wealth distribution
# ---------------------------------------------------------------------------

WEALTH_ASSET_COLS = [
    "hwl_assets_tv",       # Television
    "hwl_assets_fridge",   # Refrigerator
    "hwl_assets_estove",   # Electric stove
    "hwl_assets_washm",    # Washing machine
    "hwl_assets_comp",     # Computer
    "hwl_assets_ac",       # Air conditioner
    "hwl_assets_microw",   # Microwave
    "hwl_assets_geyser",   # Geyser (water heater)
    "hwl_assets_paytv",    # Pay TV / DStv
    "hwl_assets_vac",      # Vacuum cleaner
    "hwl_assets_dishw",    # Dishwasher
    "hwl_assets_freezer",  # Freezer
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_ghs(
    df: pd.DataFrame,
    survey_year: int = 2022,
    hhold_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    GHS-specific pre-processing applied before harmonization.

    Produces canonical recoded columns expected by the control file:
      ATTEND_RECODE   : 1=attending, 0=not attending (NaN if not applicable)
      GRADE_RECODE    : highest completed grade 0-12 (NaN if unknown)
      URBAN_RECODE    : 1=urban, 0=non-urban
      WEALTH_Q        : wealth quintile 1-5 (from asset index, if hhold available)
      DISABILITY      : 1=has disability, 0=none

    Parameters
    ----------
    df : pd.DataFrame
        Raw person-level microdata (column names lowercased by reader).
    survey_year : int
        Reference year (used in methodological notes only).
    hhold_path : str or Path, optional
        Path to the household-level DTA/CSV file. If provided, a wealth
        quintile is derived from household assets and merged in.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with recoded columns appended.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── 1. Age (already in completed years — no transformation needed) ───────
    # Column is 'age' — passed through directly in control file
    logger.debug("GHS: age variable is direct (completed years)")

    # ── 2. Attendance recode: 1=Yes -> 1, 2/3=No -> 0, 8=N/A -> NaN ─────────
    if "edu_attend" in df.columns:
        raw = pd.to_numeric(df["edu_attend"], errors="coerce")
        df["attend_recode"] = np.where(raw == 1, 1,
                              np.where(raw.isin([2, 3]), 0, np.nan))
        df["attend_recode"] = df["attend_recode"].astype("Int8")
        logger.debug("GHS: recoded edu_attend -> attend_recode (1=Yes,0=No)")

    # ── 3. Highest grade recode from education code ──────────────────────────
    if "education" in df.columns:
        raw = pd.to_numeric(df["education"], errors="coerce").astype("Int16")
        df["grade_recode"] = raw.map(EDU_CODE_TO_GRADE).astype("Int16")
        # Set 98/99 (unspecified/DK) to NaN
        df.loc[raw.isin([98, 99]), "grade_recode"] = pd.NA
        logger.debug("GHS: recoded education -> grade_recode (0-12)")

    # ── 4. Urban recode: geotype 1=Urban -> 1, 2/3=Non-urban -> 0 ───────────
    if "geotype" in df.columns:
        raw = pd.to_numeric(df["geotype"], errors="coerce")
        df["urban_recode"] = (raw == 1).astype("Int8")
        df.loc[raw.isna(), "urban_recode"] = pd.NA
        logger.debug("GHS: derived urban_recode from geotype (1=Urban)")

    # ── 5. Disability: already in 'disab' column (0=none, 1=any disability) ──
    if "disab" in df.columns:
        df["disability_recode"] = pd.to_numeric(df["disab"], errors="coerce").astype("Int8")
        logger.debug("GHS: disability_recode from disab column")

    # ── 6. Wealth quintile from household assets (optional merge) ────────────
    if hhold_path is not None:
        df = _merge_wealth_quintile(df, Path(hhold_path))
    else:
        logger.warning(
            "GHS: no hhold_path provided — wealth_q will be unavailable. "
            "Set hhold_path in preprocessor call to enable wealth disaggregation."
        )
        df["wealth_q"] = pd.NA

    return df


# ---------------------------------------------------------------------------
# Wealth quintile derivation from household assets
# ---------------------------------------------------------------------------

def _merge_wealth_quintile(df: pd.DataFrame, hhold_path: Path) -> pd.DataFrame:
    """
    Derive a wealth quintile from household asset ownership and merge into df.

    Method: sum of binary asset ownership indicators (1=owns across 12 items),
    then cut into quintiles. This mirrors the approach used by WIDE/GEM for
    surveys without a pre-built wealth index.
    """
    logger.info("GHS: loading household file for wealth quintile derivation")

    # Load household file
    if hhold_path.suffix == ".dta":
        import pyreadstat
        df_h, _ = pyreadstat.read_dta(str(hhold_path), encoding="latin1")
    else:
        df_h = pd.read_csv(hhold_path, low_memory=False)

    df_h.columns = [c.lower() for c in df_h.columns]

    if "uqnr" not in df_h.columns:
        logger.error("GHS: household file has no 'uqnr' merge key — skipping wealth.")
        df["wealth_q"] = pd.NA
        return df

    # Build asset score
    available_assets = [c for c in WEALTH_ASSET_COLS if c in df_h.columns]
    if not available_assets:
        logger.warning("GHS: no asset columns found in household file.")
        df["wealth_q"] = pd.NA
        return df

    asset_scores = pd.DataFrame(index=df_h.index)
    for col in available_assets:
        raw = pd.to_numeric(df_h[col], errors="coerce")
        asset_scores[col] = (raw == 1).astype(int)  # 1=owns, 2/9=not

    df_h["_asset_score"] = asset_scores.sum(axis=1)

    # Cut into quintiles (handle ties with duplicates='drop')
    df_h["wealth_q"] = pd.qcut(
        df_h["_asset_score"],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    ).astype("Int8")

    n_assets = len(available_assets)
    logger.info(
        "GHS: wealth quintile derived from " + str(n_assets) + " asset indicators"
    )

    # Merge into person file on uqnr
    hhold_merge = df_h[["uqnr", "wealth_q"]].drop_duplicates("uqnr")
    df = df.merge(hhold_merge, on="uqnr", how="left")

    n_matched = df["wealth_q"].notna().sum()
    logger.info(
        "GHS: wealth_q merged — "
        + str(n_matched) + "/" + str(len(df)) + " persons matched"
    )
    return df
