"""ADK function tools exposing the IEDT quantitative core (`iedt.model`) to the agents.

All tools return plain JSON-serializable dicts and never raise: errors come back as
{"status": "error", ...} so the LLM can correct its call. Every successful result carries
`provenance` (model version + parameter hash) and `warnings` for the Validation agent.

This module is the bridge between the two halves of the project: the agents in
`iedt.agents` call these functions; these functions call the model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, replace

import numpy as np

from .model import (
    BASELINE, C0, G0, I0, M0, MODEL_VERSION, SCENARIOS, X0, Y0, YEARS, Params,
    anomaly, compare_params, sample, scenario, simulate, sobol,
)

YEARS_LIST = [int(y) for y in YEARS]
RANGES = {  # hard bounds on LLM-supplied overrides
    "oil": (0.0, 0.6), "drought": (0.0, 1.0), "drought_years": (0, 6), "fin_shock": (0.0, 0.5),
    "wage_growth": (-0.05, 0.10), "capex_mult": (0.0, 2.0), "k_cap": (0.0, 1.0), "psi": (0.0, 1.5),
    "rho": (0.0, 0.95), "d_star": (0.0, 0.2), "ceiling": (0.0, 0.2), "offset": (0.0, 20.0),
}
OUTPUTS = ["deficit_pct", "capex", "invest", "nonoil_gdp", "reserves", "fin_stress",
           "imports", "gdp", "arrears"]


def _L(a):  # numpy -> rounded list
    return [round(float(x), 3) for x in a]


def _params(preset: str, overrides: str = "") -> Params:
    if preset not in SCENARIOS:
        raise ValueError(f"unknown preset '{preset}'. Valid: {list(SCENARIOS)}")
    ov = json.loads(overrides) if overrides and overrides.strip() else {}
    if not isinstance(ov, dict):
        raise ValueError("overrides must be a JSON object string, e.g. '{\"oil\": 0.15}'")
    valid = {f.name for f in fields(Params)}
    for k, v in ov.items():
        if k not in valid:
            raise ValueError(f"unknown parameter '{k}'. Valid: {sorted(valid)}")
        lo, hi = RANGES[k]
        if not (lo <= float(v) <= hi):
            raise ValueError(f"'{k}'={v} outside allowed range [{lo}, {hi}]")
    return scenario(preset, **ov)


def _prov(p: Params) -> dict:
    h = hashlib.sha1(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest()[:10]
    return {"model_version": MODEL_VERSION, "param_hash": h, "parameters": asdict(p)}


def _warnings(base, scen) -> list:
    w = ["Behavioural parameters (k_cap, psi, rho, d_star, ceiling, offset) and the baseline oil-price "
         "path are ASSUMPTIONS, not estimates. Model is not validated against historical episodes.",
         "Annual aggregate flow-consistent model, not stock-flow consistent; no banking balance sheets."]
    for name, r in (("baseline", base), ("scenario", scen)):
        if (r["reserves"] < 40).any():
            w.append(f"{name}: reserves fall below the 40bn floor (import compression active); "
                     "later-year results are mechanically distorted.")
        if (r["reserves"] < 0).any():
            w.append(f"{name}: simulated reserves turn negative, i.e. unfinanceable; read as a stress "
                     "flag, not a forecast.")
    return w


def get_model_card() -> dict:
    """Describe the quantitative IEDT core: scope, equations, calibration anchors, limits. Call before
    interpreting any simulation so claims are traceable to a model version."""
    return {
        "status": "ok", "model_version": MODEL_VERSION,
        "scope": "Annual nominal flow-consistent aggregate model, IQD trillion, 2024 anchor, 2025-2030.",
        "blocks": ["oil revenue (quota-constrained volume, exogenous price)", "rigid wage bill and fiscal gap",
                   "financing ceiling -> capex cut and arrears", "financing-stress state -> private investment",
                   "consumption, imports, output identity", "reserves with 40bn floor and import compression"],
        "calibration_anchors_2024": {"gdp": Y0, "consumption": C0, "gov_consumption": G0,
                                     "gfcf": I0, "exports": X0, "imports": M0},
        "presets": list(SCENARIOS), "parameter_ranges": RANGES, "outputs": OUTPUTS, "years": YEARS_LIST,
        "known_limitations": [
            "Baseline matches IMF 2025 (deficit ~7.5%, reserves ~91bn) but drifts afterwards.",
            "Private-investment response to oil shock is weaker than the source paper reports.",
            "Output-loss factor from import compression is not implemented.",
            "Behavioural parameters are assumed ranges."],
    }


def list_scenarios() -> dict:
    """List preset scenarios and the parameters each sets (oil=0.2 means a persistent 20% oil-price decline)."""
    return {"status": "ok", "presets": {k: v for k, v in SCENARIOS.items()}, "parameter_ranges": RANGES}


def run_scenario(preset: str = "base", overrides: str = "") -> dict:
    """Run the quantitative model for a preset plus optional parameter overrides and compare with baseline.

    Args:
        preset: one of base, oil, drought, fin, comp, comp10, protect.
        overrides: optional JSON object STRING of numeric parameters, e.g. '{"oil": 0.15, "capex_mult": 1.2}'.
            Keys: oil, drought, drought_years, fin_shock, wage_growth, capex_mult, k_cap, psi, rho,
            d_star, ceiling, offset.
    Returns:
        Yearly series (2025-2030) for baseline and scenario, 2030 comparison metrics, warnings, provenance.
    """
    try:
        p = _params(preset, overrides)
        a, b = simulate(p), simulate(BASELINE)
        return {"status": "ok", "years": YEARS_LIST,
                "scenario_series": {k: _L(a[k]) for k in OUTPUTS},
                "baseline_series": {k: _L(b[k]) for k in OUTPUTS},
                "comparison": {k: round(float(v), 3) for k, v in compare_params(p).items()},
                "warnings": _warnings(b, a), "provenance": _prov(p),
                "note": "All numbers are model-generated under stated assumptions."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_monte_carlo(preset: str = "oil", overrides: str = "", n: int = 2000) -> dict:
    """Parameter-uncertainty robustness run (k_cap, psi, rho, d_star, ceiling varied jointly).
    Returns 5/50/95th percentile paths of non-oil GDP deviation from baseline (%) plus 2030 reserves and capex."""
    try:
        p = _params(preset, overrides); n = int(min(max(n, 50), 5000))
        rng = np.random.default_rng(0)
        dev, res, cap = [], [], []
        for d in sample(n, rng):
            a, b = simulate(replace(p, **d)), simulate(replace(BASELINE, **d))
            dev.append(100 * (a["nonoil_gdp"] / b["nonoil_gdp"] - 1))
            res.append(a["reserves"][-1]); cap.append(a["capex"][-1])
        dev = np.array(dev)
        pc = lambda x: [round(float(v), 2) for v in np.percentile(x, [5, 50, 95])]
        return {"status": "ok", "years": YEARS_LIST, "draws": n,
                "nonoil_gdp_dev_pct": {"p5": _L(np.percentile(dev, 5, 0)), "p50": _L(np.percentile(dev, 50, 0)),
                                       "p95": _L(np.percentile(dev, 95, 0))},
                "reserves_2030_p5_p50_p95": pc(res), "capex_2030_p5_p50_p95": pc(cap),
                "warnings": _warnings(simulate(BASELINE), simulate(p)), "provenance": _prov(p)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_sobol(preset: str = "oil", output: str = "invest", n: int = 256) -> dict:
    """Global sensitivity (Sobol first-order S1 and total ST) of a 2030 output to the uncertain parameters.
    output is one of: deficit_pct, capex, invest, nonoil_gdp, reserves, fin_stress, imports, gdp, arrears."""
    try:
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}")
        _params(preset); n = int(min(max(n, 64), 512))
        r = sobol(preset, output, n)
        return {"status": "ok", "preset": preset, "output": f"{output} (2030)", "samples": n,
                "indices": {k: {"S1": round(float(s1), 3), "ST": round(float(st), 3)} for k, (s1, st) in r.items()},
                "interpretation_hint": "High ST = parameter drives the conclusion; headline results should not be "
                                       "stronger than the least-identified high-ST parameter.",
                "provenance": {"model_version": MODEL_VERSION, "preset": preset}}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_anomaly(deficit_pct_2025: float, reserves_usd_bn_2025: float) -> dict:
    """Compare OBSERVED 2025 deficit (% of GDP) and year-end reserves (US$ bn) with the baseline model
    ensemble; returns z-scores and a status (consistent / WATCH / ANOMALY). Only pass observed values."""
    try:
        r = anomaly(float(deficit_pct_2025), float(reserves_usd_bn_2025))
        return {"status": "ok", "z_deficit": round(float(r["z_deficit"]), 2),
                "z_reserves": round(float(r["z_reserves"]), 2), "assessment": r["status"],
                "provenance": {"model_version": MODEL_VERSION}}
    except Exception as e:
        return {"status": "error", "message": str(e)}
