from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from gem_pipeline.quality.qa_checks import QAReport

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
LEVEL_ORDER = ["primary", "lower_secondary", "upper_secondary"]


def export_results(results, qa_report, output_dir, survey_id, make_figures=True):
    d = Path(output_dir) / survey_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    results.to_csv(d / "indicators.csv", index=False)
    _write_excel(results, d / "indicators_wide.xlsx")
    qa_report.flags.to_csv(d / "qa_flags.csv", index=False)
    if make_figures and not results.empty:
        _plot_oosr(results, d / "figures")
        _plot_completion(results, d / "figures")
        _plot_literacy(results, d / "figures")
    logger.info(f"Results written to: {d}")
    return d


def _write_excel(df, path):
    if df.empty: return
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="All", index=False)
        for ind, grp in df.groupby("indicator"):
            grp.pivot_table(index=["level","group","group_value"], values="estimate", aggfunc="first").reset_index().to_excel(w, sheet_name=ind[:31], index=False)


def _plot_oosr(df, fig_dir):
    data = df[(df["indicator"]=="oosr") & (df["group"]=="sex") & (df["level"].isin(LEVEL_ORDER))].copy()
    if data.empty: return
    data["level"] = pd.Categorical(data["level"], LEVEL_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(8,4))
    for sex, grp in data.sort_values("level").groupby("group_value"):
        ax.plot(grp["level"], grp["estimate"]*100, marker="o", linewidth=2, label=sex.capitalize())
    ax.set_title("Out-of-School Rate by Sex"); ax.set_ylabel("%"); ax.legend(title="Sex")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    fig.tight_layout(); fig.savefig(fig_dir/"oosr_by_sex.png"); plt.close(fig)


def _plot_completion(df, fig_dir):
    data = df[(df["indicator"]=="completion") & (df["group"]=="total") & (df["level"].isin(LEVEL_ORDER))].copy()
    if data.empty: return
    data["level"] = pd.Categorical(data["level"], LEVEL_ORDER, ordered=True)
    data = data.sort_values("level")
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(data["level"], data["estimate"]*100, color=sns.color_palette("colorblind", len(data)), edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3); ax.set_ylim(0,110)
    ax.set_title("Completion Rate by Level"); ax.set_ylabel("%")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    fig.tight_layout(); fig.savefig(fig_dir/"completion_by_level.png"); plt.close(fig)


def _plot_literacy(df, fig_dir):
    data = df[(df["indicator"]=="literacy") & (df["group"]=="wealth") & (df["level"]=="youth_15_24")].copy()
    if data.empty: return
    data = data.sort_values("group_value")
    data["quintile"] = data["group_value"].str.replace("q","Q",regex=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(data["quintile"], data["estimate"]*100, color=sns.color_palette("Blues_d", len(data)), edgecolor="white")
    ax.set_title("Youth Literacy Rate by Wealth Quintile"); ax.set_ylabel("%"); ax.set_ylim(0,110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    fig.tight_layout(); fig.savefig(fig_dir/"literacy_by_wealth.png"); plt.close(fig)
