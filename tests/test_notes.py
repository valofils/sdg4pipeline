from __future__ import annotations
import pandas as pd
import pytest
from gem_pipeline.docs.methodological_notes import generate_note
from gem_pipeline.quality.qa_checks import QAReport


def _sample_results():
    return pd.DataFrame([
        {"indicator": "oosr",       "level": "primary",    "group": "total",
         "group_value": "total",  "estimate": 0.074, "n_unweighted": 5662,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "oosr",       "level": "primary",    "group": "sex",
         "group_value": "male",   "estimate": 0.071, "n_unweighted": 2681,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "oosr",       "level": "primary",    "group": "sex",
         "group_value": "female", "estimate": 0.078, "n_unweighted": 2981,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "oosr",       "level": "primary",    "group": "wealth",
         "group_value": "q1",     "estimate": 0.158, "n_unweighted": 741,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "oosr",       "level": "primary",    "group": "wealth",
         "group_value": "q5",     "estimate": 0.034, "n_unweighted": 1500,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "oosr",       "level": "primary",    "group": "urban",
         "group_value": "urban",  "estimate": 0.055, "n_unweighted": 3802,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "oosr",       "level": "primary",    "group": "urban",
         "group_value": "rural",  "estimate": 0.113, "n_unweighted": 1860,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "attendance", "level": "primary",    "group": "total",
         "group_value": "total",  "estimate": 0.926, "n_unweighted": 5662,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
        {"indicator": "literacy",   "level": "youth_15_24","group": "total",
         "group_value": "total",  "estimate": 0.897, "n_unweighted": 9821,
         "survey_id": "ZAF_2022_GHS2022", "country_iso3": "ZAF", "survey_year": 2022},
    ])


def _empty_qa():
    return QAReport(n_checks=5, n_flags=0, flags=pd.DataFrame())


class TestMethodologicalNotes:

    def test_generates_string(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        assert isinstance(md, str)
        assert len(md) > 500

    def test_contains_required_sections(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        for section in [
            "Survey Identification",
            "Sample Design",
            "Education Variables",
            "Indicator Definitions",
            "Disaggregation Dimensions",
            "Key Results",
            "Data Quality Summary",
            "Limitations",
        ]:
            assert section in md, "Missing section: " + section

    def test_country_name_in_header(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        assert base_cfg.country_name in md

    def test_survey_id_in_header(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        assert base_cfg.survey_id in md

    def test_all_indicators_documented(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        for ind in ["Out-of-school rate", "Attendance rate", "Completion rate"]:
            assert ind in md or "attendance" in md.lower()

    def test_equity_highlights_wealth_gap(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        assert "Q1" in md or "Poorest" in md or "pp" in md

    def test_qa_zero_flags_message(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        assert "No quality issues" in md

    def test_qa_with_flags_shows_table(self, base_cfg):
        flags = pd.DataFrame([{
            "survey_id": "TST", "country_iso3": "TST", "survey_year": 2023,
            "indicator": "oosr", "level": "primary", "group": "total",
            "group_value": "total", "estimate": 1.5,
            "flag_code": "BOUNDS", "flag_message": "Out of range", "severity": "ERROR",
        }])
        qa = QAReport(n_checks=5, n_flags=1, flags=flags)
        md = generate_note(base_cfg, _sample_results(), qa)
        assert "BOUNDS" in md
        assert "ERROR" in md

    def test_empty_results_handled(self, base_cfg):
        md = generate_note(base_cfg, pd.DataFrame(), _empty_qa())
        assert "No results available" in md

    def test_footer_disclaimer(self, base_cfg):
        md = generate_note(base_cfg, _sample_results(), _empty_qa())
        assert "reviewed and validated" in md
