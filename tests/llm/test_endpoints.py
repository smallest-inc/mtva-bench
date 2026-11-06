from typing import get_args

import pytest
from pydantic import ValidationError

from mtva_bench.llm.endpoints import PROVIDERS, Endpoint, ProviderName


def test_provider_names_match_the_registry() -> None:
    assert set(get_args(ProviderName)) == set(PROVIDERS)


def test_known_provider_defaults() -> None:
    endpoint = Endpoint(provider="anthropic", model="claude-opus-5", params={"max_tokens": 8000})
    assert endpoint.wire_api == "messages"
    assert endpoint.url == "https://api.anthropic.com/v1"


def test_smallest_reads_its_own_key(monkeypatch) -> None:
    monkeypatch.setenv("SMALLEST_API_KEY", "sm-test")
    endpoint = Endpoint(provider="smallest", model="electron", base_url="https://example.test/v1")
    assert endpoint.wire_api == "chat"
    assert endpoint.api_key() == "sm-test"


def test_api_override_and_custom_url() -> None:
    endpoint = Endpoint(
        provider="custom", model="electron", base_url="http://localhost:8000/v1/", api="chat"
    )
    assert endpoint.url == "http://localhost:8000/v1"
    assert endpoint.api_key() is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"provider": "acme", "model": "m"}, "Input should be 'openai'"),
        ({"provider": "custom", "model": "m"}, "requires base_url"),
        ({"provider": "smallest", "model": "electron"}, "requires base_url"),
        ({"provider": "anthropic", "model": "claude-opus-5"}, "requires params.max_tokens"),
        ({"provider": "openai", "model": "m", "api_key": "sk"}, "Extra inputs"),
    ],
)
def test_invalid_endpoints(fields: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Endpoint(**fields)


def test_key_from_environment_then_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# keys\nOPENAI_API_KEY='from-file'\n")
    endpoint = Endpoint(provider="openai", model="gpt-5.2")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert endpoint.api_key(env_file) == "from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert endpoint.api_key(env_file) == "from-env"


def test_missing_key_names_the_variable(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
    endpoint = Endpoint(
        provider="custom", model="electron", base_url="http://x", api_key_env="SMALLEST_API_KEY"
    )
    with pytest.raises(ValueError, match="SMALLEST_API_KEY is not set"):
        endpoint.api_key(tmp_path / ".env")
