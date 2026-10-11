
"""Model providers for AI Engineer Mentor.

Groq is the hosted API provider.
Ollama remains available for future local-model development.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


class ProviderError(RuntimeError):
    """A readable model-provider error."""


def get_secret(name: str) -> Optional[str]:
    """Read a secret from environment variables or Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


class ModelProvider:
    name = "base"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GroqProvider(ModelProvider):
    """Generate responses using the Groq API."""

    name = "Groq API"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
    ):
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatGroq(
                model=self.model,
                groq_api_key=self.api_key,
                temperature=0.4,
                max_retries=1,
            )

            result = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            content = result.content

            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", ""))
                    if isinstance(part, dict)
                    else str(part)
                    for part in content
                )

            text = str(content).strip()

            if not text:
                raise ProviderError("Groq returned an empty response.")

            return text

        except ProviderError:
            raise
        except Exception as exc:
            # Keep the API key and sensitive request details out of the UI.
            raise ProviderError(
                f"Groq could not complete the request "
                f"({type(exc).__name__}). Check the API key, model access, "
                "rate limits, and connection."
            ) from exc


class OllamaProvider(ModelProvider):
    """Call an Ollama server running on the same computer as this code."""

    name = "Local Ollama"

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        endpoint: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.4},
        }

        request = urllib.request.Request(
            self.endpoint + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))

            text = data.get("message", {}).get("content", "").strip()

            if not text:
                raise ProviderError("The local model returned an empty response.")

            return text

        except urllib.error.URLError as exc:
            raise ProviderError(
                "The local Ollama server is not reachable. "
                f"Make sure Ollama is running and the model "
                f"'{self.model}' is installed."
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "The local model took too long to respond. Try a shorter question."
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"The local model request failed ({type(exc).__name__})."
            ) from exc


def available_provider(mode: str = "Automatic") -> ModelProvider:
    """Select a provider.

    Automatic: try local Ollama first, then Groq.
    API/Groq: use Groq.
    Local: use Ollama.
    """
    mode = (mode or "Automatic").strip().lower()

    if mode in {"local", "offline", "local ollama"} or mode.startswith("local"):
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        )

    if mode in {"api", "groq", "online api"} or mode.startswith("api"):
        api_key = get_secret("GROQ_API_KEY")

        if not api_key:
            raise ProviderError(
                "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
                "or your local environment."
            )

        return GroqProvider(api_key)

    # Automatic mode: first check for a local Ollama server.
    local = OllamaProvider(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    )

    try:
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            method="GET",
        )

        with urllib.request.urlopen(request, timeout=1.5):
            return local

    except Exception:
        pass

    # Fall back to Groq when local Ollama is unavailable.
    api_key = get_secret("GROQ_API_KEY")

    if api_key:
        return GroqProvider(api_key)

    raise ProviderError(
        "No model is configured. Add GROQ_API_KEY to Streamlit Secrets "
        "or start a local Ollama server."
    )
