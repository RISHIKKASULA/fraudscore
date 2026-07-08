"""Unit tests for champion selection (ADR-002)."""

from __future__ import annotations

from fraudscore.train import select_champion


def test_lower_calibration_cost_wins():
    assert select_champion({"main": 100.0, "baseline": 200.0}) == "main"
    assert select_champion({"main": 200.0, "baseline": 100.0}) == "baseline"


def test_tie_goes_to_the_simpler_model():
    assert select_champion({"main": 150.0, "baseline": 150.0}) == "baseline"
