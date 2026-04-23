from __future__ import annotations

"""
docs/methodological_notes.py
-----------------------------
Generates a structured methodological note (Markdown) for each country
survey processed by the GEM SDG4 pipeline.

Output matches the content expected for GEM Report Output 2:
  - Survey identification and provenance
  - Sample design summary
  - Education variables and coding decisions
  - Indicator definitions and age thresholds
  - Disaggregation dimensions available
  - Data quality summary (QA flags)
  - Key results table
  - Known limitations and analytical notes

Usage (programmatic):
    from gem_pipeline.docs.methodological_notes import generate_note
    md = generate_note(cfg, results, qa_report)
    Path("docs/notes/ZAF_2022_GHS2022.md").write_text(md)

Usage (CLI):
    gem-pipeline notes --control config/control_file.csv --countries ZAF
"""

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gem_pipeline.config_loader import CountryConfig
    from gem_pipeline.quality.qa_checks import QAReport

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

INDICATOR_FULL_NAMES = {
    "oosr":       "Out-of-school rate",
    "attendance": "School attendance rate",
    "completion": "Completion rate",
    "literacy":   "Literacy rate",
    "repetition": "Repetition rate (proxy)",
}

INDICATOR_SDG = {
    "oosr":       "SDG 4.1.4",
    "attendance": "SDG 4.1",
    "completion": "SDG 4.1.2",
    "literacy":   "SDG 4.6.1",
    "repetition": "SDG 4.1",
}

INDICATOR_DEFINITION = {
    "oosr": (
        "Percentage of children and young people in the official school-age "
        "range for a given level of education who are not enrolled in pre-primary, "
        "primary, secondary or higher education."
    ),
    "attendance": (
        "Percentage of children and young people in the official school-age "
        "range for a given level who report currently attending school."
    ),
    "completion": (
        "Percentage of young people 3-5 years older than the intended exit age "
        "for a given level who have completed at least the final grade of that level."
    ),
    "literacy": (
        "Percentage of the population in a given age group who can both read "
        "and write, with understanding, a short simple statement about their "
        "everyday life (self-reported or tested)."
    ),
    "repetition": (
        "Proxy estimate: percentage of primary-school-age children currently "
        "attending who are enrolled in a grade lower than expected for their age. "
        "Note: this is an approximation; precise repetition rates require panel "
        "or grade-transition data."
    ),
}

SURVEY_TYPE_FULL = {
    "MHHS": "Multi-Purpose Household Survey",
    "LFS":  "Labour Force Survey",
    "DHS":  "Demographic and Health Survey",
    "MICS": "Multiple Indicator Cluster Survey (UNICEF)",
    "HIES": "Household Income and Expenditure Survey",
    "LSMS": "Living Standards Measurement Study",
}

LEVEL_LABELS = {
    "primary":         "Primary education",
    "lower_secondary": "Lower secondary education",
    "upper_secondary": "Upper secondary education",
    "youth_15_24":     "Youth (15-24 years)",
    "adult_15plus":    "Adults (15+ years)",
}

GROUP_LABELS = {
    "total":     "Total",
    "sex":       "Sex",
    "urban":     "Urban/Rural",
    "wealth":    "Wealth quintile",
    "ethnicity": "Ethnicity / population group",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_note(
    cfg: "CountryConfig",
    results: pd.DataFrame,
    qa_report: "QAReport",
    generated_by: str = "GEM SDG4 Pipeline",
) -> str:
    """
    Generate a methodological note in Markdown for one country survey.

    Parameters
    ----------
    cfg : CountryConfig
        Validated country configuration.
    results : pd.DataFrame
        Long-format indicator results from compute_all_indicators().
    qa_report : QAReport
        Quality assurance report from run_qa().
    generated_by : str
        Tool name to include in the header.

    Returns
    -------
    str
        Full methodological note as a Markdown string.
    """
    sections = [
        _header(cfg, generated_by),
        _section_survey_id(cfg),
        _section_sample_design(cfg, results),
        _section_education_variables(cfg),
        _section_indicator_definitions(cfg),
        _section_disaggregations(cfg, results),
        _section_key_results(results),
        _section_qa_summary(qa_report),
        _section_limitations(cfg),
        _footer(),
    ]
    return "\n\n".join(s.strip() for s in sections if s.strip()) + "\n"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header(cfg, generated_by: str) -> str:
    return (
        "# Methodological Note\n"
        "## " + cfg.country_name + " — " + cfg.survey_name
        + " (" + str(cfg.survey_year) + ")\n\n"
        + "> **Survey ID:** `" + cfg.survey_id + "`  \n"
        + "> **Generated:** " + str(date.today()) + "  \n"
        + "> **Generated by:** " + generated_by + "  \n"
        + "> **ISO 3166-1 alpha-3:** " + cfg.country_iso3
    )


def _section_survey_id(cfg) -> str:
    survey_type_label = SURVEY_TYPE_FULL.get(cfg.survey_type, cfg.survey_type)
    lines = [
        "## 1. Survey Identification",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Country | " + cfg.country_name + " (" + cfg.country_iso3 + ") |",
        "| Survey name | " + cfg.survey_name + " |",
        "| Survey type | " + survey_type_label + " |",
        "| Reference year | " + str(cfg.survey_year) + " |",
        "| File format | " + cfg.file_format.upper() + " |",
        "| Source file | `" + str(cfg.file_path) + "` |",
    ]
    if cfg.notes:
        lines += ["", "**Additional notes:** " + str(cfg.notes)]
    return "\n".join(lines)


def _section_sample_design(cfg, results: pd.DataFrame) -> str:
    # Derive sample size from results if available
    total_rows = results[
        (results["group"] == "total") &
        (results["level"].isin(["primary", "youth_15_24"]))
    ] if not results.empty and "group" in results.columns else pd.DataFrame()
    n_note = ""
    if not total_rows.empty:
        max_n = int(total_rows["n_unweighted"].max())
        n_note = (
            "\n\nThe largest sub-sample used in indicator computation "
            "contains **{:,} unweighted observations**.".format(max_n)
        )

    design_rows = [
        "| Field | Variable | Notes |",
        "|---|---|---|",
        "| Sampling weight | `" + cfg.weight_var + "` | Applied to all estimates |",
    ]
    if cfg.strata_var:
        design_rows.append(
            "| Stratification | `" + cfg.strata_var + "` | Used in survey design |"
        )
    else:
        design_rows.append("| Stratification | — | Not specified |")

    if cfg.psu_var:
        design_rows.append(
            "| Primary sampling unit | `" + cfg.psu_var + "` | Used in survey design |"
        )
    else:
        design_rows.append("| Primary sampling unit | — | Not specified |")

    return (
        "## 2. Sample Design\n\n"
        "All estimates are probability-weighted using the survey's sampling "
        "weight variable. Standard errors account for the complex survey design "
        "where stratification and PSU variables are available.\n\n"
        + "\n".join(design_rows)
        + n_note
    )


def _section_education_variables(cfg) -> str:
    rows = [
        "## 3. Education Variables and Coding",
        "",
        "The following variables were used from the source data file. "
        "All variable names are as they appear in the raw microdata "
        "(lowercased for processing).",
        "",
        "| Concept | Variable | Coding |",
        "|---|---|---|",
        "| Age | `" + cfg.age_var + "` | Completed years |",
        "| Sex | `" + cfg.sex_var + "` | 1 = Male, 2 = Female |",
        "| School attendance | `" + cfg.school_attendance_var + "` | "
        + str(cfg.school_attendance_value_yes) + " = Currently attending |",
    ]

    if cfg.highest_grade_var:
        rows.append(
            "| Highest grade completed | `" + cfg.highest_grade_var
            + "` | Numeric grade (0 = none) |"
        )
    if cfg.highest_level_var:
        rows.append(
            "| Highest level completed | `" + cfg.highest_level_var
            + "` | ISCED-style level codes |"
        )
    if cfg.literacy_var:
        rows.append(
            "| Literacy | `" + cfg.literacy_var + "` | "
            + str(cfg.literacy_value_literate) + " = Literate |"
        )
    if cfg.urban_rural_var:
        rows.append(
            "| Urban/rural | `" + cfg.urban_rural_var + "` | "
            + str(cfg.urban_value) + " = Urban |"
        )
    if cfg.wealth_quintile_var:
        rows.append(
            "| Wealth quintile | `" + cfg.wealth_quintile_var
            + "` | 1 = Poorest, 5 = Richest |"
        )
    if cfg.ethnicity_var:
        rows.append(
            "| Ethnicity / group | `" + cfg.ethnicity_var + "` | Survey-specific codes |"
        )
    if cfg.disability_var:
        rows.append(
            "| Disability | `" + cfg.disability_var + "` | 1 = Has disability |"
        )

    rows += [
        "",
        "### Official school-age thresholds",
        "",
        "| Level | Age range |",
        "|---|---|",
        "| Primary | " + str(cfg.age_primary_min) + "-" + str(cfg.age_primary_max) + " years |",
        "| Lower secondary | "
        + str(cfg.age_lower_secondary_min) + "-"
        + str(cfg.age_lower_secondary_max) + " years |",
        "| Upper secondary | "
        + str(cfg.age_upper_secondary_min) + "-"
        + str(cfg.age_upper_secondary_max) + " years |",
    ]
    return "\n".join(rows)


def _section_indicator_definitions(cfg) -> str:
    rows = [
        "## 4. Indicator Definitions",
        "",
        "The following indicators were computed for this survey.",
        "",
    ]
    for ind in cfg.indicators:
        rows += [
            "### " + INDICATOR_FULL_NAMES.get(ind, ind),
            "",
            "**SDG target:** " + INDICATOR_SDG.get(ind, "—"),
            "",
            INDICATOR_DEFINITION.get(ind, "No definition available."),
            "",
        ]
    return "\n".join(rows)


def _section_disaggregations(cfg, results: pd.DataFrame) -> str:
    rows = [
        "## 5. Disaggregation Dimensions",
        "",
        "The table below shows which disaggregation dimensions are available "
        "for each indicator based on variable coverage in this survey.",
        "",
        "| Dimension | Variable | Available |",
        "|---|---|---|",
    ]

    dims = [
        ("Sex",       cfg.sex_var,          True),
        ("Urban/Rural", cfg.urban_rural_var, bool(cfg.urban_rural_var)),
        ("Wealth quintile", cfg.wealth_quintile_var, bool(cfg.wealth_quintile_var)),
        ("Ethnicity",  cfg.ethnicity_var,    bool(cfg.ethnicity_var)),
        ("Disability", cfg.disability_var,   bool(cfg.disability_var)),
    ]
    for label, var, avail in dims:
        var_str = "`" + str(var) + "`" if var else "—"
        tick = "Yes" if avail else "No"
        rows.append("| " + label + " | " + var_str + " | " + tick + " |")

    # Count actual disaggregated estimates produced
    if not results.empty:
        n_disagg = len(results[results["group"] != "total"])
        n_total  = len(results[results["group"] == "total"])
        rows += [
            "",
            "In total, **" + str(n_total) + " aggregate estimates** and "
            "**" + str(n_disagg) + " disaggregated estimates** were produced.",
        ]
    return "\n".join(rows)


def _section_key_results(results: pd.DataFrame) -> str:
    if results.empty:
        return "## 6. Key Results\n\n_No results available._"

    rows = [
        "## 6. Key Results",
        "",
        "Summary of total (aggregate) estimates. "
        "All values are weighted proportions (0-100 scale).",
        "",
        "| Indicator | Level | Estimate (%) | N (unweighted) |",
        "|---|---|---|---|",
    ]

    total_df = (
        results[results["group"] == "total"]
        .sort_values(["indicator", "level"])
    )

    for _, row in total_df.iterrows():
        ind_label   = INDICATOR_FULL_NAMES.get(row["indicator"], row["indicator"])
        level_label = LEVEL_LABELS.get(row["level"], row["level"])
        estimate    = round(float(row["estimate"]) * 100, 1)
        n           = "{:,}".format(int(row["n_unweighted"]))
        rows.append(
            "| " + ind_label + " | " + level_label
            + " | " + str(estimate) + "% | " + n + " |"
        )

    # Highlight largest equity gaps
    rows += ["", "### Equity highlights", ""]
    highlights = _equity_highlights(results)
    if highlights:
        for h in highlights:
            rows.append("- " + h)
    else:
        rows.append("_No significant equity gaps detected._")

    return "\n".join(rows)


def _equity_highlights(results: pd.DataFrame) -> list[str]:
    highlights = []

    for indicator in results["indicator"].unique():
        ind_df = results[results["indicator"] == indicator]
        ind_label = INDICATOR_FULL_NAMES.get(indicator, indicator)

        for level in ind_df["level"].unique():
            level_label = LEVEL_LABELS.get(level, level)

            # Wealth gap: Q1 vs Q5
            wealth = ind_df[
                (ind_df["level"] == level) & (ind_df["group"] == "wealth")
            ].set_index("group_value")["estimate"]
            if "q1" in wealth.index and "q5" in wealth.index:
                gap = abs(float(wealth["q1"]) - float(wealth["q5"])) * 100
                if gap >= 5.0:
                    direction = "higher" if float(wealth["q1"]) > float(wealth["q5"]) else "lower"
                    highlights.append(
                        "**" + ind_label + " — " + level_label + ":** "
                        "Poorest quintile (Q1) is "
                        + str(round(gap, 1)) + " percentage points "
                        + direction + " than richest (Q5)."
                    )

            # Urban-rural gap
            urban = ind_df[
                (ind_df["level"] == level) & (ind_df["group"] == "urban")
            ].set_index("group_value")["estimate"]
            if "urban" in urban.index and "rural" in urban.index:
                gap = abs(float(urban["rural"]) - float(urban["urban"])) * 100
                if gap >= 3.0:
                    direction = "higher" if float(urban["rural"]) > float(urban["urban"]) else "lower"
                    highlights.append(
                        "**" + ind_label + " — " + level_label + ":** "
                        "Rural is " + str(round(gap, 1)) + " pp "
                        + direction + " than urban."
                    )

    return highlights[:8]  # Cap at 8 bullet points


def _section_qa_summary(qa_report: "QAReport") -> str:
    rows = [
        "## 7. Data Quality Summary",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Total QA checks run | " + str(qa_report.n_checks) + " |",
        "| Total flags raised | " + str(qa_report.n_flags) + " |",
    ]

    if qa_report.n_flags == 0:
        rows += [
            "",
            "No quality issues were detected. All estimates pass bounds checks, "
            "sample size thresholds, cross-indicator coherence tests, and "
            "sex parity plausibility checks.",
        ]
    else:
        by_sev = qa_report.flags.groupby("severity").size()
        for sev, cnt in by_sev.items():
            rows.append("| " + sev + " flags | " + str(cnt) + " |")

        rows += ["", "### Flag details", ""]
        flag_cols = ["indicator", "level", "group", "group_value",
                     "flag_code", "flag_message", "severity"]
        available = [c for c in flag_cols if c in qa_report.flags.columns]
        # Header
        rows.append("| " + " | ".join(available) + " |")
        rows.append("|" + "|".join(["---"] * len(available)) + "|")
        for _, frow in qa_report.flags[available].iterrows():
            rows.append("| " + " | ".join(str(frow[c]) for c in available) + " |")

    return "\n".join(rows)


def _section_limitations(cfg) -> str:
    paras = [
        "## 8. Limitations and Analytical Notes",
        "",
        "The following limitations should be considered when using these estimates:",
        "",
        "1. **Cross-sectional design.** The " + cfg.survey_name + " is a "
        "cross-sectional survey. Causal inference and cohort tracking are not "
        "possible from a single survey round.",
        "",
        "2. **Self-reported attendance.** School attendance is based on "
        "respondent self-report (or proxy report by household head). It may "
        "not capture irregular attendance or seasonal dropout.",
        "",
        "3. **Grade-for-age completion proxy.** Completion rates are estimated "
        "using highest grade completed among young adults 3-5 years above the "
        "official exit age. This is a standard proxy but may understate "
        "completion for late entrants or repeaters.",
    ]

    if "literacy" in cfg.indicators:
        paras += [
            "",
            "4. **Self-reported literacy.** The literacy indicator uses "
            "self-reported (or proxy-reported) reading and writing ability. "
            "This typically overestimates functional literacy compared to "
            "direct assessments (e.g. LAMP, RAMAA).",
        ]

    if "repetition" in cfg.indicators:
        paras += [
            "",
            "5. **Repetition rate proxy.** The repetition rate is a rough "
            "approximation based on grade-for-age comparisons. It cannot "
            "distinguish between late school entry and actual repetition. "
            "Administrative data from education management information systems "
            "(EMIS) are preferred for this indicator.",
        ]

    if cfg.strata_var is None or cfg.psu_var is None:
        paras += [
            "",
            "6. **Incomplete survey design variables.** Stratification and/or "
            "PSU variables are not specified for this survey. Standard errors "
            "and confidence intervals may be underestimated.",
        ]

    paras += [
        "",
        "7. **Comparability.** Variable definitions, reference periods, and "
        "age thresholds may differ from those used in other national surveys "
        "or international databases. Users should consult the survey "
        "questionnaire and UIS metadata before making cross-country comparisons.",
    ]

    return "\n".join(paras)


def _footer() -> str:
    return (
        "---\n\n"
        "*This note was generated automatically by the GEM SDG4 Pipeline. "
        "It should be reviewed and validated by a monitoring analyst before "
        "publication or submission to GEM Report platforms.*"
    )
