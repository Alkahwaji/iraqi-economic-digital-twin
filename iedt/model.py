"""Quantitative core of the Iraqi Economic Digital Twin (IEDT).

Minimum Viable Model (MVM): annual, nominal, flow-consistent aggregate model in IQD trillion,
2024 anchor, 2025-2030 projection. Includes preset scenarios, Monte Carlo robustness, Sobol
sensitivity and an anomaly check. Depends only on numpy (matplotlib optional, for plots).

This module knows nothing about agents: `iedt.tools` wraps it for the LLM agents and
`iedt.cli` exposes it on the command line.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

MODEL_VERSION = "iedt-mvm-0.1"

YEARS = np.arange(2025, 2031)
BASE_PRICE = np.array([.92, .90, .90, .90, .90, .90])  # baseline oil-price index (assumption)

# ---- 2024 anchors (IQD trillion) ----
Y0, C0, G0, I0, X0, M0 = 327.6, 168.7, 80.6, 52.2, 144.6, 135.9
S0 = Y0 - (C0 + G0 + I0 + X0 - M0)          # statistical discrepancy (= 17.4)
XOIL0, TAU = 135.0, 0.873                    # oil exports; oil-revenue conversion
WG0, OTH0, CAP0 = 72.1, 43.9, 26.5           # wages, other current, capex target
G_TREND = 0.03


@dataclass(frozen=True)
class Params:
    # scenario drivers
    oil: float = 0.0        # persistent oil-price decline (0.2 = -20%)
    drought: float = 0.0    # agricultural loss severity (share of 5% of GDP)
    drought_years: int = 2
    fin_shock: float = 0.0  # financing-stress shock per year, 2025-26
    wage_growth: float = 0.03
    capex_mult: float = 1.0
    # uncertain behavioural parameters
    k_cap: float = 0.7      # share of excess gap absorbed by capex
    psi: float = 0.5        # financing stress -> investment
    rho: float = 0.6        # persistence of financing stress
    d_star: float = 0.05    # deficit/GDP threshold for stress
    ceiling: float = 0.075  # financing ceiling (share of GDP)
    offset: float = 4.5     # capital-outflow offset, US$bn per year


BASELINE = Params()
UNCERTAIN = {  # name: (low, high) used by Monte Carlo and Sobol
    "k_cap": (0.4, 0.9), "psi": (0.2, 0.8), "rho": (0.3, 0.8),
    "d_star": (0.03, 0.07), "ceiling": (0.06, 0.09),
}


def simulate(p: Params = BASELINE) -> dict[str, np.ndarray]:
    """Run the MVM. Returns arrays over 2025-2030."""
    yp, fs, res = Y0, 0.0, 100.3
    out = {k: [] for k in ("deficit_pct", "capex", "invest", "nonoil_gdp",
                           "reserves", "fin_stress", "imports", "gdp", "arrears")}
    for t in range(1, 7):
        xoil = XOIL0 * BASE_PRICE[t - 1] * (1 - p.oil)           # quota-constrained volume
        x = xoil + 9.6 * (1 + G_TREND) ** t
        rev = TAU * xoil + 0.033 * yp                            # oil + non-oil revenue
        dag = 0.05 * Y0 * p.drought if t <= p.drought_years else 0.0
        sup = 0.3 * dag
        wg = WG0 * (1 + p.wage_growth) ** t
        oth = OTH0 * 1.02 ** t
        target = CAP0 * p.capex_mult
        pre = wg + oth + sup + target - rev
        excess = max(0.0, pre - p.ceiling * yp)
        cap = max(0.0, target - p.k_cap * excess)
        arrears = (1 - p.k_cap) * excess
        deficit = wg + oth + sup + cap - rev
        fs = p.rho * fs + max(0.0, deficit / yp - p.d_star) + (p.fin_shock if t <= 2 else 0.0)
        inv = max(0.0, I0 * (1 + G_TREND) ** t * (1 - min(1.0, p.psi * fs)) - 0.5 * arrears)
        c = 0.5 * (wg + 0.5 * oth + sup) + 0.371 * yp
        g = 0.695 * (wg + oth)
        m = 0.40 * c + 0.45 * inv + 0.20 * g + 28.8 * (1 + G_TREND) ** t + 0.9 * dag
        if res < 40:                                             # reserve floor -> import compression
            m *= 0.8
        y = c + g + inv + x - m + S0 * (1 + G_TREND) ** t - dag
        res = res + (x - m - 2.15) * 1000 / 1310 - p.offset
        for k, v in zip(out, (100 * deficit / yp, cap, inv, y - xoil, res, fs, m, y, arrears)):
            out[k].append(v)
        yp = y
    return {k: np.array(v) for k, v in out.items()}


SCENARIOS = {
    "base": {},
    "oil": {"oil": .20},
    "drought": {"drought": .25},
    "fin": {"fin_shock": .08},
    "comp": {"oil": .20, "drought": .25, "fin_shock": .08},
    "comp10": {"oil": .10, "drought": .25, "fin_shock": .08},
    "protect": {"oil": .20, "capex_mult": 1.2, "wage_growth": 0.0},
}


def scenario(name: str, **kw) -> Params:
    return replace(BASELINE, **{**SCENARIOS[name], **kw})


def compare(name: str) -> dict[str, float]:
    a, b = simulate(scenario(name)), simulate(BASELINE)
    return {
        "deficit_2030_pp": a["deficit_pct"][-1] - b["deficit_pct"][-1],
        "capex_2030_%": 100 * (a["capex"][-1] / b["capex"][-1] - 1),
        "invest_2030_%": 100 * (a["invest"][-1] / b["invest"][-1] - 1),
        "nonoil_cum_%": 100 * (a["nonoil_gdp"].sum() / b["nonoil_gdp"].sum() - 1),
        "reserves_2030_bn": a["reserves"][-1],
    }


def compare_params(p: Params) -> dict[str, float]:
    a, b = simulate(p), simulate(BASELINE)
    return {
        "deficit_2030_pp": a["deficit_pct"][-1] - b["deficit_pct"][-1],
        "capex_2030_pct": 100 * (a["capex"][-1] / b["capex"][-1] - 1),
        "invest_2030_pct": 100 * (a["invest"][-1] / b["invest"][-1] - 1),
        "nonoil_cum_pct": 100 * (a["nonoil_gdp"].sum() / b["nonoil_gdp"].sum() - 1),
        "reserves_2030_bn": a["reserves"][-1],
    }


def sample(n: int, rng: np.random.Generator) -> list[dict]:
    return [{k: rng.uniform(lo, hi) for k, (lo, hi) in UNCERTAIN.items()} for _ in range(n)]


def monte_carlo(name: str, n: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dev, res, cap = [], [], []
    for d in sample(n, rng):
        a, b = simulate(replace(scenario(name), **d)), simulate(replace(BASELINE, **d))
        dev.append(100 * (a["nonoil_gdp"] / b["nonoil_gdp"] - 1))
        res.append(a["reserves"][-1]); cap.append(a["capex"][-1])
    dev = np.array(dev)
    return {"p5": np.percentile(dev, 5, 0), "p50": np.percentile(dev, 50, 0),
            "p95": np.percentile(dev, 95, 0),
            "reserves": np.array(res), "capex": np.array(cap)}


def sobol(name: str = "oil", output: str = "invest", n: int = 1024, seed: int = 0):
    """Saltelli (2010) first-order and Jansen total-order Sobol indices.
    Output: 2030 value of `output` for the scenario `name`."""
    rng = np.random.default_rng(seed)
    keys = list(UNCERTAIN)
    lo = np.array([UNCERTAIN[k][0] for k in keys]); hi = np.array([UNCERTAIN[k][1] for k in keys])
    A = lo + rng.random((n, len(keys))) * (hi - lo)
    B = lo + rng.random((n, len(keys))) * (hi - lo)

    def f(M):
        return np.array([simulate(replace(scenario(name), **dict(zip(keys, r))))[output][-1] for r in M])
    fA, fB = f(A), f(B)
    var = np.var(np.concatenate([fA, fB]))
    res = {}
    for i, k in enumerate(keys):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fAB = f(ABi)
        s1 = np.mean(fB * (fAB - fA)) / var
        st = 0.5 * np.mean((fA - fAB) ** 2) / var
        res[k] = (s1, st)
    return res


def anomaly(deficit_pct: float, reserves: float, n: int = 400, seed: int = 0) -> dict:
    """Compare observed 2025 values to the baseline parameter ensemble."""
    rng = np.random.default_rng(seed)
    ds, rs = [], []
    for d in sample(n, rng):
        r = simulate(replace(BASELINE, offset=rng.uniform(3, 6), **d))
        ds.append(r["deficit_pct"][0]); rs.append(r["reserves"][0])
    zd = (deficit_pct - np.mean(ds)) / (np.std(ds) or 1)
    zr = (reserves - np.mean(rs)) / (np.std(rs) or 1)
    s = max(abs(zd), abs(zr))
    return {"z_deficit": zd, "z_reserves": zr,
            "status": "ANOMALY (>2 sigma)" if s > 2 else "WATCH" if s > 1 else "consistent"}


def plot(name: str, path: str = "iedt_plot.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = simulate(scenario(name)), simulate(BASELINE)
    titles = {"deficit_pct": "Deficit (% GDP)", "capex": "Capex (IQD tn)", "invest": "Private investment (IQD tn)",
              "nonoil_gdp": "Non-oil GDP (IQD tn)", "reserves": "Reserves (US$ bn)", "fin_stress": "Financing stress"}
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for axi, (k, t) in zip(ax.ravel(), titles.items()):
        axi.plot(YEARS, b[k], "--", color="gray", label="baseline")
        axi.plot(YEARS, a[k], color="teal", label=name); axi.set_title(t)
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(path, dpi=130)
    print("saved", path)
