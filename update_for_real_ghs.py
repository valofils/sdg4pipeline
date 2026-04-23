"""
update_for_real_ghs.py
----------------------
Run this once in the Codespace after placing the real GHS files
in data/raw/ to update the control file and pipeline for real data.

Usage:
    python3 update_for_real_ghs.py
"""
from pathlib import Path

# ── 1. Update pipeline.py to pass hhold_path to GHS preprocessor ─────────────
pipeline_path = Path("src/gem_pipeline/pipeline.py")
text = pipeline_path.read_text()

old = (
    "    if 'GHS' in cfg.survey_name.upper():\n"
    "        from gem_pipeline.ingestion.preprocessors import preprocess_ghs\n"
    "        df = preprocess_ghs(df, survey_year=cfg.survey_year)\n"
    "        df.columns = [c.lower() for c in df.columns]"
)
new = (
    "    if 'GHS' in cfg.survey_name.upper():\n"
    "        from gem_pipeline.ingestion.preprocessors import preprocess_ghs\n"
    "        # Look for companion household file\n"
    "        p = Path(str(cfg.file_path))\n"
    "        hhold_candidates = [\n"
    "            p.parent / p.name.replace('person', 'hhold'),\n"
    "            p.parent / 'ghs-2022-hhold-v1.dta',\n"
    "            p.parent / 'ghs-2022-hhold-v1.csv',\n"
    "        ]\n"
    "        hhold_path = next((h for h in hhold_candidates if h.exists()), None)\n"
    "        if hhold_path:\n"
    "            logger.info('GHS: using household file ' + str(hhold_path.name))\n"
    "        df = preprocess_ghs(df, survey_year=cfg.survey_year, hhold_path=hhold_path)\n"
    "        df.columns = [c.lower() for c in df.columns]"
)

if old in text:
    pipeline_path.write_text(text.replace(old, new))
    print("Patched: pipeline.py — hhold_path auto-detection added")
else:
    print("WARNING: pipeline.py pattern not found — may already be patched or different")

# ── 2. Update control file with correct real GHS 2022 variable names ─────────
ctrl_path = Path("config/control_file_template.csv")
lines = ctrl_path.read_text().splitlines()

# Remove old ZAF row
lines = [l for l in lines if not l.startswith("ZAF,")]

# Real GHS 2022 control file row — all variable names verified from codebook
real_row = (
    "ZAF,South Africa,2022,GHS2022,MHHS,"
    "data/raw/ghs-2022-person-v1.dta,dta,"   # file
    "person_wgt,stratum,psu,"                  # weight, strata, PSU
    "age,sex,"                                 # age (direct), sex (1=M,2=F)
    "attend_recode,1,"                         # attendance (preprocessed)
    "grade_recode,,"                           # highest grade (preprocessed); no level var
    ",,,"                                      # no literacy var in GHS 2022
    "urban_recode,1,"                          # urban (preprocessed)
    "wealth_q,population,disability_recode,"  # wealth, ethnicity, disability
    "6,11,12,14,15,17,"                       # official SA school age bands
    '"oosr,attendance,completion",'            # indicators (no literacy — not in GHS)
    "Real GHS 2022 data. Stats SA via DataFirst ZAF-STATSSA-GHS-2022-V1. "
    "Wealth quintile derived from 12 asset indicators. "
    "Literacy not available in GHS 2022 person file."
)

lines.append(real_row)
ctrl_path.write_text("\n".join(lines) + "\n")
print("Updated: config/control_file_template.csv (real GHS 2022 row)")

print("\nNext steps:")
print("  1. Place ghs-2022-person-v1.dta AND ghs-2022-hhold-v1.dta in data/raw/")
print("  2. gem-pipeline validate --control config/control_file_template.csv")
print("  3. gem-pipeline run --control config/control_file_template.csv --countries ZAF")
