from src.fl.client import runtime_config_from_env


def test_runtime_config_defaults() -> None:
    runtime = runtime_config_from_env()
    assert runtime.client_id == "0"
    assert runtime.server_address == "localhost:8080"
