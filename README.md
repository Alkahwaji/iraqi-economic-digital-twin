# Iraqi Economic Digital Twin (IEDT)

A minimum viable macro model of the Iraqi economy, connected to a multi-agent analysis system
built on [Google ADK](https://google.github.io/adk-docs/). The agents do not guess numbers: they
call function tools that run the quantitative model, and every result carries provenance
(model version and parameter hash).

> Research prototype. Behavioural parameters and the baseline oil-price path are assumptions, not
> estimates, and the model is not validated against historical episodes. Not a forecast.

## How the two systems connect

```
 user question
      |
      v
 iedt/agents.py      orchestrator + 7 specialist agents (Google ADK)
      |  call
      v
 iedt/tools.py       JSON-safe function tools: bounds checking, warnings, provenance
      |  call
      v
 iedt/model.py       quantitative core (numpy only): simulate, Monte Carlo, Sobol, anomaly
```

`iedt/cli.py` and `web/index.html` use the same model directly, with no agents involved.

| Path | Role |
|---|---|
| `iedt/model.py` | Annual, nominal, flow-consistent model (IQD trillion), 2024 anchor, 2025-2030 projection. Scenarios, Monte Carlo, Sobol sensitivity, anomaly check. |
| `iedt/tools.py` | Tools the agents call: `get_model_card`, `list_scenarios`, `run_scenario`, `run_monte_carlo`, `run_sobol`, `check_anomaly`. Errors are returned as `{"status": "error"}` so the LLM can correct its call. |
| `iedt/agents.py` | Orchestrator, data, state, scenario, simulation, risk, red-team and validation agents. Imports without `google-adk` installed (agents are then `None`). |
| `iedt/cli.py` | Command line access to the model. |
| `app/agent.py` | ADK entry point (`root_agent`, `app`) for `adk run` / `adk web`. |
| `web/index.html` | Standalone browser demo: a JavaScript port of the same model with sliders, Monte Carlo and anomaly panels. |
| `tests/` | pytest suite for the model and the tool layer. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # model + tools + CLI (numpy only)
pip install -e ".[plot]"         # + matplotlib
pip install -e ".[agents]"       # + google-adk for the multi-agent layer
pip install -e ".[dev]"          # + pytest
```

Python 3.10 or newer.

## Use

Quantitative model (no API keys needed):

```bash
python -m iedt scenarios                  # preset scenarios, 2030 vs baseline
python -m iedt mc --scenario comp         # Monte Carlo fan
python -m iedt sobol --scenario oil       # Sobol indices
python -m iedt anomaly --deficit 7.5 --reserves 91
python -m iedt plot --scenario comp       # writes iedt_plot.png
python -m iedt card                       # model card as JSON
```

```python
from iedt import simulate, scenario
r = simulate(scenario("oil", capex_mult=1.2))
print(r["deficit_pct"], r["reserves"])
```

Multi-agent system (needs `google-adk` and Gemini credentials configured as ADK requires):

```bash
adk web          # run from the repository root, then pick "app"
adk run app
```

The agent model name is set once in `iedt/agents.py` (`AGENT_MODEL`).

Browser demo: open `web/index.html` in any browser.

Tests:

```bash
pytest -q
```

## Presets

`base`, `oil` (-20% oil price), `drought`, `fin` (financing stress), `comp` (compound),
`comp10` (-10% compound), `protect` (oil shock with capex protected).

## Known limitations

- The baseline matches IMF 2025 moments (deficit about 7.5% of GDP, reserves about US$91bn) but
  drifts afterwards: reserves fall below the US$40bn import-compression floor around 2029.
- Under the oil shock the model cuts capex to zero and drives reserves negative; read that as a
  stress flag, not a forecast. The output loss from import compression is not implemented.
- The private-investment response to oil shocks is weaker than in the source paper.
- Not stock-flow consistent; no banking balance sheets.
- `web/index.html` re-implements the model in JavaScript, so it must be kept in sync with
  `iedt/model.py` by hand.
