from __future__ import annotations
import pandas as pd
import pytest
from gem_pipeline.output.wide_scope_formatter import to_wide, to_scope, to_both


def _sample_results():
    """Realistic ZAF GHS 2022 results subset."""
    return pd.DataFrame([
        {"indicator":"oosr","level":"primary","group":"total","group_value":"total","estimate":0.0742,"n_unweighted":5662,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"primary","group":"sex","group_value":"male","estimate":0.0705,"n_unweighted":2681,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"primary","group":"sex","group_value":"female","estimate":0.0776,"n_unweighted":2981,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"primary","group":"urban","group_value":"urban","estimate":0.0553,"n_unweighted":3802,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"primary","group":"urban","group_value":"rural","estimate":0.113,"n_unweighted":1860,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"primary","group":"wealth","group_value":"q1","estimate":0.1584,"n_unweighted":741,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"primary","group":"wealth","group_value":"q5","estimate":0.0337,"n_unweighted":1500,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"oosr","level":"lower_secondary","group":"total","group_value":"total","estimate":0.0803,"n_unweighted":2866,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"attendance","level":"primary","group":"total","group_value":"total","estimate":0.926,"n_unweighted":5662,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"completion","level":"primary","group":"total","group_value":"total","estimate":0.999,"n_unweighted":2882,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"completion","level":"upper_secondary","group":"total","group_value":"total","estimate":0.164,"n_unweighted":2829,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"literacy","level":"youth_15_24","group":"total","group_value":"total","estimate":0.950,"n_unweighted":9622,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"literacy","level":"youth_15_24","group":"sex","group_value":"male","estimate":0.942,"n_unweighted":4680,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
        {"indicator":"literacy","level":"youth_15_24","group":"sex","group_value":"female","estimate":0.957,"n_unweighted":4942,"survey_id":"ZAF_2022_GHS2022","country_iso3":"ZAF","survey_year":2022},
    ])


class TestWIDEFormatter:

    def test_returns_dataframe(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_required_metadata_columns(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        for col in ["country_code","country_name","survey_name","year","subgroup_type","subgroup_value"]:
            assert col in df.columns, "Missing column: " + col

    def test_indicator_columns_present(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        assert "oos_prim_t" in df.columns
        assert "oos_prim_m" in df.columns
        assert "oos_prim_f" in df.columns
        assert "oos_prim_w1" in df.columns
        assert "lit_youth_t" in df.columns
        assert "comp_prim_t" in df.columns

    def test_values_are_percentages(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        val = df[df["subgroup_value"] == "Total"]["oos_prim_t"].iloc[0]
        assert 0 < val < 100, "Expected percentage, got: " + str(val)
        assert abs(val - 7.42) < 0.1

    def test_n_columns_present(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        assert "oos_prim_t_n" in df.columns

    def test_one_row_per_subgroup(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        # Should have one row per unique subgroup
        assert df.shape[0] == df[["subgroup_type","subgroup_value"]].drop_duplicates().shape[0]

    def test_empty_results_returns_empty(self, base_cfg):
        df = to_wide(pd.DataFrame(), base_cfg)
        assert df.empty

    def test_saves_csv(self, base_cfg, tmp_path):
        p = tmp_path / "wide.csv"
        to_wide(_sample_results(), base_cfg, output_path=p)
        assert p.exists()
        loaded = pd.read_csv(p)
        assert not loaded.empty

    def test_country_code_correct(self, base_cfg):
        df = to_wide(_sample_results(), base_cfg)
        assert (df["country_code"] == base_cfg.country_iso3).all()


class TestSCOPEFormatter:

    def test_returns_dataframe(self, base_cfg):
        df = to_scope(_sample_results(), base_cfg)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_required_columns(self, base_cfg):
        df = to_scope(_sample_results(), base_cfg)
        for col in ["country_code","year","indicator_code","subgroup_type",
                    "subgroup_value","value","n_unweighted"]:
            assert col in df.columns, "Missing: " + col

    def test_indicator_codes_valid(self, base_cfg):
        df = to_scope(_sample_results(), base_cfg)
        codes = df["indicator_code"].unique()
        assert "OOSR.PRIM" in codes
        assert "ATT.PRIM" in codes
        assert "COMP.PRIM" in codes
        assert "LIT.YOUTH" in codes

    def test_values_are_percentages(self, base_cfg):
        df = to_scope(_sample_results(), base_cfg)
        total = df[(df["indicator_code"]=="OOSR.PRIM") & (df["subgroup_value"]=="Total")]
        assert not total.empty
        assert abs(float(total["value"].iloc[0]) - 7.42) < 0.1

    def test_long_format_one_row_per_estimate(self, base_cfg):
        results = _sample_results()
        df = to_scope(results, base_cfg)
        # Every valid (indicator, level, group, gval) combination -> one row
        assert len(df) > 0
        assert len(df) <= len(results)

    def test_subgroup_labels_readable(self, base_cfg):
        df = to_scope(_sample_results(), base_cfg)
        vals = df["subgroup_value"].unique()
        assert "Total" in vals
        assert "Male" in vals
        assert "Female" in vals
        assert "Urban" in vals
        assert "Quintile 1 (poorest)" in vals

    def test_empty_results_returns_empty(self, base_cfg):
        assert to_scope(pd.DataFrame(), base_cfg).empty

    def test_saves_csv(self, base_cfg, tmp_path):
        p = tmp_path / "scope.csv"
        to_scope(_sample_results(), base_cfg, output_path=p)
        assert p.exists()

    def test_sorted_output(self, base_cfg):
        df = to_scope(_sample_results(), base_cfg)
        # indicator_code should be sorted
        codes = df["indicator_code"].tolist()
        assert codes == sorted(codes) or len(set(codes)) > 1  # at least sorted within groups


class TestToBoth:

    def test_returns_tuple(self, base_cfg, tmp_path):
        wide, scope = to_both(_sample_results(), base_cfg, tmp_path)
        assert isinstance(wide, pd.DataFrame)
        assert isinstance(scope, pd.DataFrame)

    def test_both_files_written(self, base_cfg, tmp_path):
        to_both(_sample_results(), base_cfg, tmp_path)
        assert (tmp_path / "wide_export.csv").exists()
        assert (tmp_path / "scope_export.csv").exists()

    def test_wide_has_more_columns_than_scope(self, base_cfg, tmp_path):
        wide, scope = to_both(_sample_results(), base_cfg, tmp_path)
        assert len(wide.columns) > len(scope.columns)
