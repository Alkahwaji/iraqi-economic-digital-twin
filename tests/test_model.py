import numpy as np
import pytest

from iedt import BASELINE, SCENARIOS, anomaly, compare, monte_carlo, scenario, simulate, sobol


def test_baseline_shape_and_2025_anchor():
    r = simulate(BASELINE)
    assert all(len(v) == 6 for v in r.values())
    # baseline is tuned to approximate IMF 2025 moments (deficit ~7.5-8% GDP, reserves ~91bn)
    assert r["deficit_pct"][0] == pytest.approx(7.7, abs=0.2)
    assert r["reserves"][0] == pytest.approx(91.2, abs=0.5)


def test_oil_shock_worsens_fiscal_and_reserves():
    base, oil = simulate(BASELINE), simulate(scenario("oil"))
    assert oil["deficit_pct"][-1] > base["deficit_pct"][-1]
    assert oil["reserves"][-1] < base["reserves"][-1]


def test_all_presets_run_and_compare():
    for name in SCENARIOS:
        r = compare(name)
        assert set(r) == {"deficit_2030_pp", "capex_2030_%", "invest_2030_%",
                          "nonoil_cum_%", "reserves_2030_bn"}
    assert compare("base")["deficit_2030_pp"] == 0.0


def test_monte_carlo_is_reproducible_and_ordered():
    a, b = monte_carlo("oil", n=100, seed=1), monte_carlo("oil", n=100, seed=1)
    assert np.allclose(a["p50"], b["p50"])
    assert (a["p5"] <= a["p50"] + 1e-9).all() and (a["p50"] <= a["p95"] + 1e-9).all()


def test_sobol_returns_every_uncertain_parameter():
    res = sobol("oil", "invest", n=64)
    assert set(res) == {"k_cap", "psi", "rho", "d_star", "ceiling"}


def test_anomaly_flags_extreme_observation():
    assert anomaly(7.7, 91.2)["status"] == "consistent"
    assert anomaly(30.0, 5.0)["status"].startswith("ANOMALY")
