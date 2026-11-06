import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Api = Literal["chat", "responses", "messages"]
ProviderName = Literal["openai", "anthropic", "gemini", "openrouter", "smallest", "custom"]


@dataclass(frozen=True, kw_only=True)
class Provider:
    base_url: str | None
    key_env: str | None
    api: Api


PROVIDERS: dict[ProviderName, Provider] = {
    "openai": Provider(base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY", api="chat"),
    "anthropic": Provider(
        base_url="https://api.anthropic.com/v1",
        key_env="ANTHROPIC_API_KEY",
        api="messages",
    ),
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
        api="chat",
    ),
    "openrouter": Provider(
        base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY", api="chat"
    ),
    "smallest": Provider(base_url=None, key_env="SMALLEST_API_KEY", api="chat"),
    "custom": Provider(base_url=None, key_env=None, api="chat"),
}


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api: Api | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    unsupported: list[str] = Field(default_factory=list)
    timeout_s: float = Field(default=600, gt=0)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "Endpoint":
        if not (self.base_url or self.spec.base_url):
            raise ValueError(f"provider '{self.provider}' requires base_url")
        if self.wire_api == "messages" and "max_tokens" not in self.params:
            raise ValueError("the messages api requires params.max_tokens")
        return self

    @property
    def spec(self) -> Provider:
        return PROVIDERS[self.provider]

    @property
    def wire_api(self) -> Api:
        return self.api or self.spec.api

    @property
    def url(self) -> str:
        return (self.base_url or self.spec.base_url or "").rstrip("/")

    def api_key(self, env_file: Path = Path(".env")) -> str | None:
        name = self.api_key_env or self.spec.key_env
        if name is None:
            return None
        value = read_env(name, env_file)
        if not value:
            raise ValueError(f"{name} is not set in the environment or {env_file}")
        return value


def read_env(name: str, env_file: Path = Path(".env")) -> str | None:
    if name in os.environ:
        return os.environ[name]
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip("'\"")
    return None
