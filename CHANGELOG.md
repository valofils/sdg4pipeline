# Changelog

All notable changes to the GEM SDG4 Pipeline are documented here.

## [0.1.0] — 2026-04-23

### Added
- Multi-format ingestion: Stata (.dta), SPSS (.sav), CSV, Parquet
- Harmonization engine with canonical `_*` schema
- Weighted indicator computation: OOSR, attendance, completion, literacy, repetition
- Disaggregation by sex, urban/rural, wealth quintile, ethnicity, disability
- Six automated QA checks: bounds, coherence, temporal, sex gap, monotonicity, small-N
- CSV, Excel, PNG figure export
- WIDE flat export (GEM column naming convention)
- SCOPE long export (UIS indicator codes)
- Methodological notes generator (8-section markdown, UNESCO Output 2 format)
- CLI: `gem-pipeline run / validate / inventory / notes / export`
- GitHub Actions CI (lint, type-check, test, control file validation)
- South Africa GHS 2022 preprocessor (real data, 66,144 persons)
  - Wealth quintile derived from 12 household asset indicators
  - Household file merge on `uqnr`
  - latin1 encoding fallback for Stats SA DTA files
- 54 unit tests, all passing

### Notes
- Confidence intervals not yet implemented (planned for v0.2.0)
- Preprocessor registration is currently hardcoded in pipeline.py (planned refactor)
