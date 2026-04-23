from __future__ import annotations
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
