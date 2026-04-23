import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from gem_pipeline.config_loader import CountryConfig

@pytest.fixture(scope="session")
def synthetic_raw_df():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        "wgt": rng.uniform(0.5, 5.0, n), "age": rng.integers(5, 25, n),
        "sex": rng.choice([1, 2], n), "attend": rng.choice([0, 1], n, p=[0.25, 0.75]),
        "grade": rng.integers(0, 13, n),
        "literate": np.where(rng.integers(5, 25, n) >= 15, rng.choice([0, 1], n, p=[0.2, 0.8]), np.nan),
        "urbrur": rng.choice([1, 0], n), "wealth": rng.integers(1, 6, n),
        "strata": rng.integers(1, 5, n), "psu": rng.integers(1, 20, n),
    })

@pytest.fixture(scope="session")
def base_cfg():
    return CountryConfig(
        country_iso3="TST", country_name="Test Country", survey_year=2023,
        survey_name="TEST2023", survey_type="MHHS", file_path=Path("data/raw/dummy.dta"),
        file_format="dta", weight_var="wgt", strata_var="strata", psu_var="psu",
        age_var="age", sex_var="sex", school_attendance_var="attend",
        school_attendance_value_yes=1, highest_grade_var="grade",
        literacy_var="literate", literacy_value_literate=1,
        urban_rural_var="urbrur", urban_value=1, wealth_quintile_var="wealth",
        age_primary_min=6, age_primary_max=11,
        age_lower_secondary_min=12, age_lower_secondary_max=14,
        age_upper_secondary_min=15, age_upper_secondary_max=17,
        indicators=["oosr","attendance","completion","literacy","repetition"],
    )
