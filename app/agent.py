"""ADK entry point: `adk run app` or `adk web` discover `root_agent` / `app` here.

All logic lives in the `iedt` package; this file only re-exports the objects ADK looks for.
"""
from iedt.agents import ADK_AVAILABLE, app, root_agent  # noqa: F401

if not ADK_AVAILABLE:
    raise RuntimeError(
        "google-adk is not installed. Install it with:  pip install -e \".[agents]\""
    )
