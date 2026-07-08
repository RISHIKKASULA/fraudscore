"""Shared fixtures: the committed synthetic dataset and the generator that builds it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = REPO_ROOT / "data" / "fixtures" / "synthetic.csv"


def load_make_fixture_module():
    """Load scripts/make_fixture.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "make_fixture", REPO_ROOT / "scripts" / "make_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """The committed synthetic fixture, as shipped."""
    return pd.read_csv(FIXTURE_CSV)
