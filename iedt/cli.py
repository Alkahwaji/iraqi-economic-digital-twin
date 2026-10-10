"""Command line interface for the IEDT quantitative core (no ADK needed).

  python -m iedt scenarios            # table of preset scenarios (2030 vs baseline)
  python -m iedt mc --scenario comp   # Monte Carlo fan for a scenario
  python -m iedt sobol                # Sobol indices (first-order and total)
  python -m iedt anomaly --deficit 7.5 --reserves 91
  python -m iedt plot --scenario comp # saves iedt_plot.png (needs matplotlib)
  python -m iedt card                 # print the model card (JSON)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .model import (
    BASELINE, SCENARIOS, YEARS, anomaly, compare, monte_carlo, plot, simulate, sobol,
)
from .tools import get_model_card


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scenarios", "mc", "sobol", "anomaly", "plot", "card"])
    ap.add_argument("--scenario", default="oil", choices=list(SCENARIOS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--deficit", type=float, default=7.5)
    ap.add_argument("--reserves", type=float, default=91.0)
    a = ap.parse_args()
    if a.cmd == "scenarios":
        b = simulate(BASELINE)
        print("Baseline deficit % GDP:", np.round(b["deficit_pct"], 1), " reserves:", np.round(b["reserves"], 1))
        print(f"{'scenario':9}{'def pp':>8}{'capex%':>8}{'inv%':>8}{'nonoil cum%':>13}{'res30':>8}")
        for s in SCENARIOS:
            r = compare(s)
            print(f"{s:9}{r['deficit_2030_pp']:8.1f}{r['capex_2030_%']:8.1f}{r['invest_2030_%']:8.1f}"
                  f"{r['nonoil_cum_%']:13.1f}{r['reserves_2030_bn']:8.0f}")
    elif a.cmd == "mc":
        r = monte_carlo(a.scenario, a.n)
        print("Non-oil GDP deviation %, by year", list(YEARS))
        for k in ("p5", "p50", "p95"):
            print(k, np.round(r[k], 1))
        print("2030 reserves p5/p50/p95:", np.round(np.percentile(r["reserves"], [5, 50, 95]), 0))
        print("2030 capex    p5/p50/p95:", np.round(np.percentile(r["capex"], [5, 50, 95]), 1))
    elif a.cmd == "sobol":
        for out in ("invest", "capex", "reserves"):
            print(f"\nSobol, scenario={a.scenario}, output=2030 {out}")
            for k, (s1, st) in sobol(a.scenario, out).items():
                print(f"  {k:8} S1={s1:6.2f}  ST={st:6.2f}")
    elif a.cmd == "anomaly":
        print(anomaly(a.deficit, a.reserves))
    elif a.cmd == "plot":
        plot(a.scenario)
    elif a.cmd == "card":
        print(json.dumps(get_model_card(), indent=2))


if __name__ == "__main__":
    main()
