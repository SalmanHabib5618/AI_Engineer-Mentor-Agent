"""Replaceable model providers for AI Engineer Mentor.

Gemini is intended for temporary development. Ollama is the future local/offline
provider. No provider is called until the user sends a non-social request.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional


class ProviderError(RuntimeError):
    """A readable model-provider error."""


def get_secret(name: str) -> Optional[str]:
    """Read a key from environment first, then Streamlit secrets if available."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


class ModelProvider:
    name = "base"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(ModelProvider):
    name = "Gemini API"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
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
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content).strip()
        except Exception as exc:
            raise ProviderError(
                f"The configured Gemini provider did not complete the request "
                f"({type(exc).__name__}). Check the API key, provider limits, and connection."
            ) from exc


class OllamaProvider(ModelProvider):
    """Calls an Ollama server running on this computer."""
    name = "Local Ollama"

    def __init__(self, model: str = "qwen2.5:3b", endpoint: str = "http://127.0.0.1:11434"):
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
                "The local model is not reachable. Install/start Ollama and download "
                f"the configured model ({self.model}), then try again."
            ) from exc
        except TimeoutError as exc:
            raise ProviderError("The local model took too long to respond. Try a shorter question.") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"The local model request failed ({type(exc).__name__}).") from exc


def available_provider(mode: str = "Automatic") -> ModelProvider:
    """Select a provider. Automatic prefers local when available, then API.

    Local availability is checked by attempting a lightweight Ollama tags request.
    No external network call is made by this check.
    """
    mode = (mode or "Automatic").strip().lower()
    api_key = get_secret("GOOGLE_API_KEY")
    local_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    local = OllamaProvider(model=local_model)
    if mode in {"local", "offline", "local ollama"} or mode.startswith("local"):
        return local
    if mode in {"api", "gemini", "online api"} or mode.startswith("api"):
        if not api_key:
            raise ProviderError(
                "No GOOGLE_API_KEY is configured. Add it to your local .env environment "
                "or Streamlit secrets, or select Local mode after setting up Ollama."
            )
        return GeminiProvider(api_key)
    # Automatic: use a running local Ollama first, then API if a key is configured.
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5):
            return local
    except Exception:
        pass
    if api_key:
        return GeminiProvider(api_key)
    raise ProviderError(
        "No model is configured yet. For development, configure GOOGLE_API_KEY; "
        "for local/offline use, install Ollama and download a small model. "
        "The app will not make a paid API call automatically."
    )
