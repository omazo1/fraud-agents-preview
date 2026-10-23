"""The run report's arithmetic (feedback #3: numbers at the real fraud rate). No database needed."""

import pytest

from fraud_agents.services.metrics.report import offline_test, precision_at, wilson_bounds


def test_precision_falls_when_fraud_is_rarer():
    # Catch 80% of frauds, stop 0.1% of genuine payments by mistake.
    common, rare = precision_at(0.05, 0.8, 0.001), precision_at(0.001, 0.8, 0.001)
    assert common == pytest.approx(0.04 / (0.04 + 0.00095))
    assert rare == pytest.approx(0.0008 / (0.0008 + 0.000999))
    assert rare < common


def test_the_95_percent_range_contains_the_measured_rate():
    lo, hi = wilson_bounds(397, 553_574)
    assert lo < 397 / 553_574 < hi
    assert wilson_bounds(0, 995)[0] == 0


def test_the_offline_numbers_come_from_the_live_models_test_at_the_flag_line():
    off = offline_test()
    assert off["flag_at"] == 0.5
    assert 0 < off["caught"] <= off["frauds"] and off["genuine"] > 500_000
