"""Ollama Cloud client with prompt-repair JSON handling."""

from __future__ import annotations

import json
from typing import Any

from .config import LeadgenConfig
from .errors import ConfigError, ProviderError, ValidationError
from .json_utils import parse_json_object
from .net import post_json


class OllamaCloudClient:
    """Minimal Ollama Cloud chat client.

    MVP is cloud-only; it intentionally does not use local Ollama structured-output
    format because the cloud API does not support that mode today.
    """

    def __init__(self, config: LeadgenConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.ollama_api_key and self.config.ollama_model)

    def require_available(self) -> None:
        if not self.config.ollama_api_key:
            raise ConfigError("OLLAMA_API_KEY is required for Ollama Cloud LLM calls")

    def chat_text(self, messages: list[dict[str, str]]) -> str:
        self.require_available()
        payload = {
            "model": self.config.ollama_model,
            "messages": messages,
            "stream": False,
        }
        response = post_json(
            f"{self.config.ollama_base_url.rstrip('/')}/chat",
            payload,
            headers={"Authorization": f"Bearer {self.config.ollama_api_key}"},
            timeout=self.config.timeout_seconds,
            ca_bundle=self.config.ca_bundle,
        )
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(response.get("response"), str):
            return str(response["response"])
        raise ProviderError("Ollama response did not contain message.content")

    def json_object(
        self,
        *,
        task: str,
        payload: dict[str, Any],
        max_repairs: int = 1,
    ) -> dict[str, Any]:
        prompt = _json_prompt(task, payload)
        content = self.chat_text(
            [
                {
                    "role": "system",
                    "content": "You return only strict JSON objects. No markdown.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        try:
            return parse_json_object(content)
        except ValidationError as exc:
            if max_repairs < 1:
                raise
            repair = self.chat_text(
                [
                    {
                        "role": "system",
                        "content": "Repair invalid JSON. Return only a strict JSON object.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not parseable JSON. Repair it without "
                            f"changing the intended data. Error: {exc}. Raw response:\n{content}"
                        ),
                    },
                ]
            )
            return parse_json_object(repair)


def _json_prompt(task: str, payload: dict[str, Any]) -> str:
    return (
        f"Task: {task}\n"
        "Return only a JSON object. Do not include markdown fences or commentary.\n"
        "Input JSON:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )
