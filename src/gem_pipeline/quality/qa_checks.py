from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from loguru import logger

ERROR = "ERROR"
WARNING = "WARNING"
_FLAG_COLS = [
    "survey_id", "country_iso3", "survey_year",
    "indicator", "level", "group", "group_value",
    "estimate", "flag_code", "flag_message", "severity",
]


@dataclass
class QAReport:
    n_checks: int
    n_flags: int
    flags: pd.DataFrame

    def summary(self):
        lines = ["QA Report: " + str(self.n_checks) + " checks, " + str(self.n_flags) + " flags."]
        if self.n_flags:
            for sev, cnt in self.flags.groupby("severity").size().items():
                lines.append("  " + sev + ": " + str(cnt))
        return "\n".join(lines)

    def has_errors(self):
        return not self.flags.empty and ERROR in self.flags["severity"].values


def run_qa(results, history=None):
    if results.empty:
        return QAReport(0, 0, pd.DataFrame())
    all_flags, n = [], 0
    for fn in [_bounds, _small_n, _coherence, _sex_parity, _monotonicity]:
        f = fn(results)
        n += 1
        if not f.empty:
            all_flags.append(f)
    if history is not None:
        f = _temporal(results, history)
        n += 1
        if not f.empty:
            all_flags.append(f)
    flag_df = pd.concat(all_flags, ignore_index=True) if all_flags else pd.DataFrame(columns=_FLAG_COLS)
    report = QAReport(n, len(flag_df), flag_df)
    logger.info(report.summary())
    return report


def _sel(df):
    return df[[c for c in _FLAG_COLS if c in df.columns]].reset_index(drop=True)


def _bounds(df):
    bad = df[(df["estimate"] < 0) | (df["estimate"] > 1)].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["flag_code"] = "BOUNDS"
    bad["flag_message"] = bad["estimate"].apply(lambda e: "Estimate " + str(round(e, 4)) + " outside [0,1]")
    bad["severity"] = ERROR
    return _sel(bad)


def _small_n(df, min_n=30):
    if "n_unweighted" not in df.columns:
        return pd.DataFrame()
    bad = df[df["n_unweighted"] < min_n].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["flag_code"] = "SMALL_N"
    bad["flag_message"] = bad["n_unweighted"].apply(
        lambda n: "Only " + str(n) + " obs (threshold " + str(min_n) + ")"
    )
    bad["severity"] = WARNING
    return _sel(bad)


def _coherence(df):
    keys = [c for c in ["survey_id", "country_iso3", "survey_year", "level", "group", "group_value"]
            if c in df.columns]
    att = df[df["indicator"] == "attendance"].copy()
    oos = df[df["indicator"] == "oosr"].copy()
    if att.empty or oos.empty:
        return pd.DataFrame()
    merged = att.merge(oos, on=keys, suffixes=("_att", "_oos"))
    total = merged["estimate_att"] + merged["estimate_oos"]
    bad = merged[abs(total - 1.0) > 0.05].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["flag_code"] = "COHERENCE"
    bad["flag_message"] = (total[bad.index] - 1.0).abs().apply(
        lambda d: "att+oosr differs from 1 by " + str(round(d, 3))
    )
    bad["severity"] = WARNING
    bad["indicator"] = "attendance+oosr"
    bad["estimate"] = total[bad.index]
    bad = bad.drop(columns=["estimate_att", "estimate_oos"], errors="ignore")
    return _sel(bad)


def _sex_parity(df, thresh=0.25):
    sex_df = df[df["group"] == "sex"].copy()
    if sex_df.empty:
        return pd.DataFrame()
    keys = [c for c in ["survey_id", "country_iso3", "survey_year", "indicator", "level"]
            if c in sex_df.columns]
    male = sex_df[sex_df["group_value"] == "male"].set_index(keys)["estimate"]
    female = sex_df[sex_df["group_value"] == "female"].set_index(keys)["estimate"]
    common = male.index.intersection(female.index)
    if common.empty:
        return pd.DataFrame()
    gap = (male.loc[common] - female.loc[common]).abs()
    bad_idx = gap[gap > thresh].index
    if len(bad_idx) == 0:
        return pd.DataFrame()
    rows = []
    for idx in bad_idx:
        d = dict(zip(keys, idx if isinstance(idx, tuple) else [idx]))
        d["group"] = "sex"
        d["group_value"] = "male_vs_female"
        d["estimate"] = float(gap[idx])
        d["flag_code"] = "SEX_GAP"
        d["flag_message"] = "Gap " + str(round(float(gap[idx]), 3)) + " > " + str(thresh)
        d["severity"] = WARNING
        rows.append(d)
    return pd.DataFrame(rows)


def _monotonicity(df):
    total = df[(df["indicator"] == "completion") & (df["group"] == "total")].copy()
    if total.empty:
        return pd.DataFrame()
    keys = [c for c in ["survey_id", "country_iso3", "survey_year"] if c in total.columns]
    order = ["primary", "lower_secondary", "upper_secondary"]
    rows = []
    for k, grp in total.groupby(keys):
        by_level = grp.set_index("level")["estimate"]
        avail = [l for l in order if l in by_level.index]
        for i in range(len(avail) - 1):
            hi, lo = avail[i], avail[i + 1]
            if by_level[hi] < by_level[lo] - 0.02:
                d = dict(zip(keys, k if isinstance(k, tuple) else [k]))
                d["indicator"] = "completion"
                d["level"] = hi + "_vs_" + lo
                d["group"] = "total"
                d["group_value"] = "total"
                d["estimate"] = float(by_level[hi] - by_level[lo])
                d["flag_code"] = "MONOTONICITY"
                d["flag_message"] = hi + " < " + lo
                d["severity"] = WARNING
                rows.append(d)
    return pd.DataFrame(rows)


def _temporal(current, history, max_change=0.20):
    keys = [c for c in ["country_iso3", "indicator", "level", "group", "group_value"]
            if c in current.columns and c in history.columns]
    if "survey_year" not in current.columns:
        return pd.DataFrame()
    prev = history[history["survey_year"] < current["survey_year"].max()]
    if prev.empty:
        return pd.DataFrame()
    prev_latest = prev.sort_values("survey_year").drop_duplicates(subset=keys, keep="last")
    merged = current.merge(prev_latest, on=keys, suffixes=("_cur", "_prev"))
    delta = (merged["estimate_cur"] - merged["estimate_prev"]).abs()
    bad = merged[delta > max_change].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["flag_code"] = "TEMPORAL"
    bad["flag_message"] = delta[bad.index].apply(
        lambda d: "YoY change " + str(round(d, 3)) + " > " + str(max_change)
    )
    bad["severity"] = WARNING
    bad["estimate"] = bad["estimate_cur"]
    return _sel(bad.rename(columns={"survey_year_cur": "survey_year"}))
