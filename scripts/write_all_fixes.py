import os

# ── qa_checks.py ────────────────────────────────────────────────────────────
qa = r"""from __future__ import annotations
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
"""

# ── pipeline.py ─────────────────────────────────────────────────────────────
pipeline = r"""from __future__ import annotations
from pathlib import Path
import pandas as pd
from loguru import logger
from gem_pipeline.config_loader import load_control_file, filter_configs
from gem_pipeline.ingestion.reader import load_survey
from gem_pipeline.harmonization.harmonizer import harmonize
from gem_pipeline.indicators.compute import compute_all_indicators
from gem_pipeline.quality.qa_checks import run_qa
from gem_pipeline.output.exporter import export_results


def run_pipeline(control_file, output_dir="data/outputs", countries=None,
                 indicators=None, dry_run=False, history_dir=None):
    configs = load_control_file(control_file)
    if countries or indicators:
        configs = filter_configs(configs, countries=countries, indicators=indicators)
    if not configs:
        logger.warning("No matching configs found.")
        return {}
    logger.info("Pipeline: " + str(len(configs)) + " survey(s), dry_run=" + str(dry_run))
    outputs = {}
    for cfg in configs:
        try:
            out = _process_one(cfg, Path(output_dir), dry_run,
                               Path(history_dir) if history_dir else None)
            outputs[cfg.survey_id] = str(out)
        except Exception as e:
            logger.error("[" + cfg.survey_id + "] Failed: " + str(e))
    logger.info("Done. " + str(len(outputs)) + "/" + str(len(configs)) + " succeeded.")
    return outputs


def _process_one(cfg, output_dir, dry_run=False, history_dir=None):
    if dry_run:
        logger.info("  [DRY RUN] " + cfg.survey_id + ": " + str(cfg.indicators))
        return output_dir / cfg.survey_id
    df, _ = load_survey(cfg.file_path, cfg.file_format)
    df_h = harmonize(df, cfg)
    results = compute_all_indicators(df_h, cfg)
    logger.info("  [" + cfg.survey_id + "] " + str(len(results)) + " estimates computed")
    history = None
    if history_dir:
        files = list(history_dir.glob(cfg.country_iso3 + "_*_indicators.csv"))
        if files:
            history = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    qa = run_qa(results, history)
    if qa.has_errors():
        logger.error("  [" + cfg.survey_id + "] QA errors detected")
    return export_results(results, qa, output_dir, cfg.survey_id)
"""

# ── test_pipeline.py ─────────────────────────────────────────────────────────
tests = r"""from __future__ import annotations
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gem_pipeline.config_loader import CountryConfig, load_control_file
from gem_pipeline.harmonization.harmonizer import harmonize
from gem_pipeline.indicators.compute import compute_all_indicators
from gem_pipeline.quality.qa_checks import run_qa


class TestConfigLoader:
    def test_loads(self, base_cfg):
        assert base_cfg.country_iso3 == "TST"

    def test_iso3_upcased(self):
        cfg = CountryConfig(
            country_iso3="tst", country_name="T", survey_year=2023,
            survey_name="T", survey_type="LFS", file_path=Path("x.csv"),
            file_format="csv", weight_var="w", age_var="a", sex_var="s",
            school_attendance_var="att", indicators=["attendance"],
        )
        assert cfg.country_iso3 == "TST"

    def test_invalid_indicator(self):
        with pytest.raises(Exception, match="Unknown"):
            CountryConfig(
                country_iso3="TST", country_name="T", survey_year=2023,
                survey_name="T", survey_type="LFS", file_path=Path("x.csv"),
                file_format="csv", weight_var="w", age_var="a", sex_var="s",
                school_attendance_var="att", indicators=["bad_indicator"],
            )

    def test_literacy_missing_var(self):
        with pytest.raises(Exception):
            CountryConfig(
                country_iso3="TST", country_name="T", survey_year=2023,
                survey_name="T", survey_type="LFS", file_path=Path("x.csv"),
                file_format="csv", weight_var="w", age_var="a", sex_var="s",
                school_attendance_var="att", indicators=["literacy"],
            )

    def test_age_bands(self, base_cfg):
        assert base_cfg.age_bands["primary"] == (6, 11)

    def test_survey_id(self, base_cfg):
        assert base_cfg.survey_id == "TST_2023_TEST2023"

    def test_load_csv(self, tmp_path):
        header = (
            "country_iso3,country_name,survey_year,survey_name,survey_type,"
            "file_path,file_format,weight_var,strata_var,psu_var,age_var,sex_var,"
            "school_attendance_var,school_attendance_value_yes,highest_grade_var,"
            "highest_level_var,literacy_var,literacy_value_literate,urban_rural_var,"
            "urban_value,wealth_quintile_var,ethnicity_var,disability_var,"
            "age_primary_min,age_primary_max,age_lower_secondary_min,"
            "age_lower_secondary_max,age_upper_secondary_min,age_upper_secondary_max,"
            "indicators,notes"
        )
        row = (
            "GHA,Ghana,2022,GLSS7,MHHS,data/raw/x.dta,dta,wgt,st,psu,"
            "age,sex,att,1,gr,lv,lit,1,urb,1,wlth,,,6,11,12,14,15,17,"
            '"oosr,attendance",note'
        )
        f = tmp_path / "ctrl.csv"
        f.write_text(header + "\n" + row + "\n")
        cfgs = load_control_file(f)
        assert len(cfgs) == 1
        assert cfgs[0].country_iso3 == "GHA"


class TestHarmonizer:
    def test_canonical_cols(self, synthetic_raw_df, base_cfg):
        r = harmonize(synthetic_raw_df, base_cfg)
        for col in ["_age", "_sex", "_weight", "_attending"]:
            assert col in r.columns

    def test_attending_binary(self, synthetic_raw_df, base_cfg):
        r = harmonize(synthetic_raw_df, base_cfg)
        assert set(r["_attending"].dropna().unique()).issubset({0, 1})

    def test_wealth_clipped(self, synthetic_raw_df, base_cfg):
        df = synthetic_raw_df.copy()
        df.loc[0, "wealth"] = 99
        r = harmonize(df, base_cfg)
        assert r["_wealth"].dropna().between(1, 5).all()

    def test_missing_var_raises(self, synthetic_raw_df, base_cfg):
        with pytest.raises(KeyError):
            harmonize(synthetic_raw_df.drop(columns=["age"]), base_cfg)

    def test_sex_01_recoded(self, synthetic_raw_df, base_cfg):
        df = synthetic_raw_df.copy()
        df["sex"] = df["sex"].map({1: 0, 2: 1})
        r = harmonize(df, base_cfg)
        assert set(r["_sex"].dropna().unique()).issubset({1, 2})


class TestIndicators:
    def _h(self, synthetic_raw_df, base_cfg):
        return harmonize(synthetic_raw_df, base_cfg)

    def test_not_empty(self, synthetic_raw_df, base_cfg):
        r = compute_all_indicators(self._h(synthetic_raw_df, base_cfg), base_cfg)
        assert not r.empty

    def test_in_0_1(self, synthetic_raw_df, base_cfg):
        r = compute_all_indicators(self._h(synthetic_raw_df, base_cfg), base_cfg)
        assert r["estimate"].between(0, 1).all()

    def test_all_indicators(self, synthetic_raw_df, base_cfg):
        r = compute_all_indicators(self._h(synthetic_raw_df, base_cfg), base_cfg)
        assert set(base_cfg.indicators).issubset(set(r["indicator"].unique()))

    def test_sex_disagg(self, synthetic_raw_df, base_cfg):
        r = compute_all_indicators(self._h(synthetic_raw_df, base_cfg), base_cfg)
        vals = r[r["group"] == "sex"]["group_value"].values
        assert "male" in vals and "female" in vals

    def test_empty_indicators(self, synthetic_raw_df, base_cfg):
        import copy
        cfg2 = copy.deepcopy(base_cfg)
        cfg2.indicators = []
        r = compute_all_indicators(self._h(synthetic_raw_df, base_cfg), cfg2)
        assert r.empty


class TestQA:
    def _sample(self):
        return pd.DataFrame([
            {"survey_id": "TST_2023", "country_iso3": "TST", "survey_year": 2023,
             "indicator": "oosr", "level": "primary", "group": "total",
             "group_value": "total", "estimate": 0.15, "n_unweighted": 200},
            {"survey_id": "TST_2023", "country_iso3": "TST", "survey_year": 2023,
             "indicator": "attendance", "level": "primary", "group": "total",
             "group_value": "total", "estimate": 0.85, "n_unweighted": 200},
        ])

    def test_valid_no_errors(self):
        assert not run_qa(self._sample()).has_errors()

    def test_bounds_flagged(self):
        r = self._sample()
        r.loc[0, "estimate"] = 1.5
        assert "BOUNDS" in run_qa(r).flags["flag_code"].values

    def test_small_n_flagged(self):
        r = self._sample()
        r["n_unweighted"] = 5
        assert "SMALL_N" in run_qa(r).flags["flag_code"].values

    def test_coherence_flagged(self):
        r = self._sample()
        r.loc[r["indicator"] == "attendance", "estimate"] = 0.95
        assert "COHERENCE" in run_qa(r).flags["flag_code"].values

    def test_empty_returns_empty(self):
        rpt = run_qa(pd.DataFrame())
        assert rpt.n_flags == 0

    def test_summary_string(self):
        assert "QA Report" in run_qa(self._sample()).summary()
"""

files = {
    "src/gem_pipeline/quality/qa_checks.py": qa,
    "src/gem_pipeline/pipeline.py": pipeline,
    "tests/test_pipeline.py": tests,
}

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    print("Written: " + path)

print("\nAll files written.")
