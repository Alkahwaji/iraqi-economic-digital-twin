"""Iraqi Economic Digital Twin (IEDT).

Two layers, connected through `iedt.tools`:

  iedt.model   quantitative core (numpy only)
  iedt.tools   JSON-safe function tools wrapping the model, with bounds and provenance
  iedt.agents  Google ADK orchestrator + 7 specialist agents that call the tools
               (imported lazily: it needs the optional `google-adk` dependency)
"""
from .model import (  # noqa: F401
    BASELINE, MODEL_VERSION, SCENARIOS, UNCERTAIN, YEARS, Params,
    anomaly, compare, monte_carlo, scenario, simulate, sobol,
)

__version__ = "0.1.0"
