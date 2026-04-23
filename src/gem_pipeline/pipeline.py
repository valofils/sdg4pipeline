from __future__ import annotations
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
    if 'GHS' in cfg.survey_name.upper():
        from gem_pipeline.ingestion.preprocessors import preprocess_ghs
        from pathlib import Path as _Path
        _p = _Path(str(cfg.file_path))
        _hhold_candidates = [
            _p.parent / _p.name.replace('person', 'hhold'),
            _p.parent / 'ghs-2022-hhold-v1.dta',
            _p.parent / 'ghs-2022-hhold-v1.csv',
        ]
        _hhold_path = next((h for h in _hhold_candidates if h.exists()), None)
        if _hhold_path:
            logger.info('GHS: using household file ' + _hhold_path.name)
        df = preprocess_ghs(df, survey_year=cfg.survey_year, hhold_path=_hhold_path)
        df.columns = [c.lower() for c in df.columns]
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
    return export_results(results, qa, output_dir, cfg.survey_id, cfg=cfg)
