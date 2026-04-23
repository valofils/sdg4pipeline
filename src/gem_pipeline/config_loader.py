from __future__ import annotations
import csv
from pathlib import Path
from typing import Literal
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_FORMATS = Literal["dta", "sav", "csv", "parquet"]
SUPPORTED_INDICATORS = {"oosr", "attendance", "completion", "literacy", "repetition"}

class CountryConfig(BaseModel):
    country_iso3: str = Field(..., min_length=3, max_length=3)
    country_name: str
    survey_year: int
    survey_name: str
    survey_type: str
    file_path: Path
    file_format: SUPPORTED_FORMATS
    weight_var: str
    strata_var: str | None = None
    psu_var: str | None = None
    age_var: str
    sex_var: str
    school_attendance_var: str
    school_attendance_value_yes: int = 1
    highest_grade_var: str | None = None
    highest_level_var: str | None = None
    literacy_var: str | None = None
    literacy_value_literate: int | None = None
    urban_rural_var: str | None = None
    urban_value: int | None = None
    wealth_quintile_var: str | None = None
    ethnicity_var: str | None = None
    disability_var: str | None = None
    age_primary_min: int = 6
    age_primary_max: int = 11
    age_lower_secondary_min: int = 12
    age_lower_secondary_max: int = 14
    age_upper_secondary_min: int = 15
    age_upper_secondary_max: int = 17
    indicators: list[str] = Field(default_factory=list)
    notes: str | None = ""

    @field_validator("country_iso3")
    @classmethod
    def iso3_upper(cls, v):
        return v.strip().upper()

    @field_validator("indicators", mode="before")
    @classmethod
    def parse_indicators(cls, v):
        parsed = [i.strip() for i in v.split(",")] if isinstance(v, str) else list(v)
        parsed = [i for i in parsed if i]
        unknown = set(parsed) - SUPPORTED_INDICATORS
        if unknown:
            raise ValueError("Unknown indicators: " + str(unknown))
        return parsed

    @model_validator(mode="after")
    def check_literacy_fields(self):
        if "literacy" in self.indicators:
            if not self.literacy_var:
                raise ValueError("literacy_var required when literacy indicator is requested")
            if self.literacy_value_literate is None:
                raise ValueError("literacy_value_literate required")
        return self

    @property
    def survey_id(self):
        return self.country_iso3 + "_" + str(self.survey_year) + "_" + self.survey_name

    @property
    def age_bands(self):
        return {
            "primary": (self.age_primary_min, self.age_primary_max),
            "lower_secondary": (self.age_lower_secondary_min, self.age_lower_secondary_max),
            "upper_secondary": (self.age_upper_secondary_min, self.age_upper_secondary_max),
        }


def load_control_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Control file not found: " + str(path))
    configs, errors = [], []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("##")]
    reader = csv.DictReader(lines)
    for i, row in enumerate(reader, start=2):
        if not any(row.values()):
            continue
        cleaned = {k: (v.strip() if v.strip() else None) for k, v in row.items()}
        try:
            configs.append(CountryConfig(**cleaned))
        except Exception as e:
            errors.append("Row " + str(i) + " (" + str(row.get("country_iso3","?")) + "): " + str(e))
    if errors:
        raise ValueError("Control file validation failed:\n" + "\n".join(errors))
    logger.info("Loaded " + str(len(configs)) + " configs from " + path.name)
    return configs


def filter_configs(configs, countries=None, indicators=None):
    result = configs
    if countries:
        up = [c.upper() for c in countries]
        result = [c for c in result if c.country_iso3 in up]
    if indicators:
        result = [c for c in result if any(i in c.indicators for i in indicators)]
    return result
