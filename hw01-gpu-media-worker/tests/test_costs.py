import math

import pytest

from costs import api_price_per_asset, breakeven, project, report

# Lesson 05, H100 run (assets/example_runs/h100.json), 2026-11-14
STARTUP_S = 157.7
CLIP_LOAD_S = 1045.976
CLIP_MEDIAN_S = 76.675
ALL_LOADS_S = 1143.0  # clip + Sana + Kokoro, as the lesson rounds it
CARD_MEDIAN_S = 78.7
FAL_WAN_CLIP = 81 / 16 * 0.04  # $0.2025


def test_clip_breakeven_regular_is_15():
    p = project(STARTUP_S + CLIP_LOAD_S, [CLIP_MEDIAN_S], 4.50, FAL_WAN_CLIP)
    assert p.breakeven_assets == 15


def test_clip_breakeven_spot_is_5():
    p = project(STARTUP_S + CLIP_LOAD_S, [CLIP_MEDIAN_S], 2.07, FAL_WAN_CLIP)
    assert p.breakeven_assets == 5


def test_card_cost_at_20_is_018():
    p = project(STARTUP_S + ALL_LOADS_S, [CARD_MEDIAN_S], 4.50, counts=(1, 20, 200))
    assert p.fixed_usd == pytest.approx(1.63, abs=0.005)
    assert p.marginal_usd == pytest.approx(0.098, abs=0.0005)
    assert p.per_asset_usd[20] == pytest.approx(0.18, abs=0.005)
    assert p.per_asset_usd[1] == pytest.approx(1.72, abs=0.01)
    assert p.per_asset_usd[200] == pytest.approx(0.11, abs=0.005)


def test_median_not_mean_keeps_warmup_out():
    p = project(0, [6.0, 0.8, 0.8, 0.8, 0.8], 3600)  # $1/s makes c equal the median
    assert p.median_latency_s == 0.8
    assert p.marginal_usd == pytest.approx(0.8)


def test_api_cheaper_than_marginal_means_never():
    assert breakeven(fixed_usd=1.0, marginal_usd=0.01, api_price_usd=0.01) == 0
    assert breakeven(fixed_usd=1.0, marginal_usd=0.02, api_price_usd=0.01) == 0


def test_zero_fixed_cost_breaks_even_at_one():
    assert breakeven(0.0, 0.01, 0.02) == 1


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_rejects_bad_numbers(bad):
    with pytest.raises(ValueError):
        project(bad, [1.0], 1.0)
    with pytest.raises(ValueError):
        project(1.0, [bad], 1.0)


def test_api_units():
    assert api_price_per_asset({"unit": "asset", "unit_price_usd": 0.03}, None) == 0.03
    mp = api_price_per_asset({"unit": "megapixel", "unit_price_usd": 0.001}, "512x512")
    assert mp == pytest.approx(0.000262144)
    clip = {"unit": "video_second", "unit_price_usd": 0.04, "video_seconds_per_asset": 81 / 16}
    assert api_price_per_asset(clip, None) == pytest.approx(0.2025)


def test_report_refuses_failed_run():
    with pytest.raises(ValueError, match="not 'completed'"):
        report({"status": "failed"}, {"gpu": {}, "api": {}})
