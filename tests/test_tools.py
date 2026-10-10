import json

from iedt import tools


def test_run_scenario_ok_with_provenance():
    r = tools.run_scenario("comp", '{"oil": 0.15}')
    assert r["status"] == "ok"
    assert r["provenance"]["model_version"] == tools.MODEL_VERSION
    assert r["provenance"]["parameters"]["oil"] == 0.15
    assert r["warnings"]
    json.dumps(r)  # must be JSON-serializable for the LLM


def test_out_of_range_override_is_rejected_not_raised():
    r = tools.run_scenario("oil", '{"oil": 0.9}')
    assert r["status"] == "error" and "outside allowed range" in r["message"]


def test_unknown_preset_and_parameter_are_rejected():
    assert tools.run_scenario("nope")["status"] == "error"
    assert tools.run_scenario("oil", '{"bogus": 1}')["status"] == "error"
    assert tools.run_scenario("oil", "[1, 2]")["status"] == "error"


def test_param_hash_depends_on_parameters():
    a = tools.run_scenario("oil")["provenance"]["param_hash"]
    b = tools.run_scenario("oil", '{"oil": 0.25}')["provenance"]["param_hash"]
    assert a != b


def test_monte_carlo_sobol_anomaly_tools():
    assert tools.run_monte_carlo("oil", n=50)["status"] == "ok"
    assert tools.run_sobol("oil", "invest", n=64)["status"] == "ok"
    assert tools.run_sobol("oil", "not_an_output")["status"] == "error"
    assert tools.check_anomaly(7.7, 91.2)["assessment"] == "consistent"


def test_model_card_lists_presets():
    card = tools.get_model_card()
    assert card["status"] == "ok" and "comp" in card["presets"]


def test_agents_module_imports_without_adk():
    # the agent layer must not break the package when google-adk is absent
    from iedt import agents
    if not agents.ADK_AVAILABLE:
        assert agents.root_agent is None
