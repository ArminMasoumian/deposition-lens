"""Optional LLM backends. The tool works without any of this (baseline mode).

Set OPENAI_API_KEY or ANTHROPIC_API_KEY and install the matching package to use them.
Override the model with DEPO_MODEL. These backends are thin wrappers and have not been
run against the live APIs in this repo's test suite; the tests use a fake model instead.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIBackend:
    def __init__(self, model: str | None = None):
        import openai  # imported lazily so the package is optional

        self.client = openai.OpenAI()
        self.model = model or os.environ.get("DEPO_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


class AnthropicBackend:
    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so the package is optional

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("DEPO_MODEL", "claude-sonnet-5-5")

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))


def get_backend(name: str) -> LLM | None:
    if name == "none":
        return None
    if name == "openai":
        return OpenAIBackend()
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"unknown LLM backend: {name}")
