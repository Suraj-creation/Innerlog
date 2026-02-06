"""Interface to local LLMs via Ollama and LM Studio (OpenAI-compatible)."""

import ollama as ollama_client
import httpx
from typing import Optional, List, Dict, Any
import json
import re

from src.utils import get_logger

logger = get_logger("local_llm")


# ---------------------------------------------------------------------------
# Provider: Ollama
# ---------------------------------------------------------------------------
class OllamaProvider:
    """Ollama LLM backend."""

    def __init__(self, model: str = "phi3"):
        self.model = model
        self._available = False
        self._check()

    def _check(self):
        try:
            ollama_client.show(self.model)
            self._available = True
            logger.info(f"[Ollama] Model '{self.model}' is available")
        except Exception as e:
            logger.warning(f"[Ollama] Model '{self.model}' not available: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        response = ollama_client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]


# ---------------------------------------------------------------------------
# Provider: LM Studio (OpenAI-compatible API)
# ---------------------------------------------------------------------------
class LMStudioProvider:
    """LM Studio backend via OpenAI-compatible REST API."""

    def __init__(self, model: str = "mistral-7b-instruct-v0.1", base_url: str = "http://localhost:1234"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available = False
        self._check()

    def _check(self):
        try:
            r = httpx.get(f"{self.base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                models = r.json().get("data", [])
                model_ids = [m.get("id", "") for m in models]
                logger.info(f"[LMStudio] Available models: {model_ids}")
                self._available = True
            else:
                logger.warning(f"[LMStudio] Server responded {r.status_code}")
        except Exception as e:
            logger.warning(f"[LMStudio] Not reachable at {self.base_url}: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        r = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Unified LLM interface
# ---------------------------------------------------------------------------
class LocalLLM:
    """Unified interface to local LLMs.

    Supports Ollama (phi3) as primary and LM Studio (mistral) as secondary.
    Falls back automatically when the primary provider is unavailable.
    """

    def __init__(
        self,
        model: str = "phi3",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        provider: str = "ollama",
        lmstudio_model: str = "mistral-7b-instruct-v0.1",
        lmstudio_url: str = "http://localhost:1234",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Primary: Ollama
        self.ollama = OllamaProvider(model=model)

        # Secondary: LM Studio
        self.lmstudio = LMStudioProvider(model=lmstudio_model, base_url=lmstudio_url)

        # Active provider selection
        if provider == "lmstudio" and self.lmstudio.is_available:
            self._active = "lmstudio"
        elif self.ollama.is_available:
            self._active = "ollama"
        elif self.lmstudio.is_available:
            self._active = "lmstudio"
        else:
            self._active = "none"

        logger.info(f"Active LLM provider: {self._active} "
                     f"(Ollama/{model}: {'OK' if self.ollama.is_available else 'N/A'}, "
                     f"LMStudio/{lmstudio_model}: {'OK' if self.lmstudio.is_available else 'N/A'})")

    @property
    def is_available(self) -> bool:
        return self._active != "none"

    @property
    def active_provider(self) -> str:
        return self._active

    def _get_provider(self):
        if self._active == "ollama":
            return self.ollama
        elif self._active == "lmstudio":
            return self.lmstudio
        return None

    def _fallback_generate(self, messages, temperature, max_tokens):
        """Try the other provider."""
        alt = "lmstudio" if self._active == "ollama" else "ollama"
        alt_provider = self.lmstudio if alt == "lmstudio" else self.ollama
        if alt_provider.is_available:
            logger.warning(f"Primary provider failed, falling back to {alt}")
            return alt_provider.generate(messages, temperature, max_tokens)
        raise RuntimeError("All LLM providers unavailable")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from LLM."""
        if not self.is_available:
            return '{"error": "LLM not available"}'

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature or self.temperature
        tokens = max_tokens or self.max_tokens

        provider = self._get_provider()
        try:
            return provider.generate(messages, temp, tokens)
        except Exception as e:
            logger.error(f"Primary LLM ({self._active}) failed: {e}")
            try:
                return self._fallback_generate(messages, temp, tokens)
            except Exception as e2:
                logger.error(f"Fallback LLM also failed: {e2}")
                return f'{{"error": "{str(e)}"}}'

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        json_instruction = "\n\nRespond with valid JSON only. No other text."
        if schema:
            json_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\nOnly output the JSON, no other text."

        response = self.generate(prompt + json_instruction, system_prompt=system_prompt, temperature=0.3)
        return self._parse_json(response)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """Multi-turn chat."""
        if not self.is_available:
            return "LLM not available"

        temp = temperature or self.temperature
        provider = self._get_provider()
        try:
            return provider.generate(messages, temp, self.max_tokens)
        except Exception as e:
            logger.error(f"Chat via {self._active} failed: {e}")
            try:
                return self._fallback_generate(messages, temp, self.max_tokens)
            except Exception as e2:
                return f"Error: {str(e2)}"

    def switch_provider(self, provider_name: str) -> bool:
        """Switch active provider at runtime."""
        if provider_name == "ollama" and self.ollama.is_available:
            self._active = "ollama"
            logger.info("Switched to Ollama provider")
            return True
        elif provider_name == "lmstudio" and self.lmstudio.is_available:
            self._active = "lmstudio"
            logger.info("Switched to LM Studio provider")
            return True
        logger.warning(f"Cannot switch to {provider_name} — not available")
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return provider status info."""
        return {
            "active_provider": self._active,
            "ollama": {"model": self.ollama.model, "available": self.ollama.is_available},
            "lmstudio": {"model": self.lmstudio.model, "available": self.lmstudio.is_available, "url": self.lmstudio.base_url},
        }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw_response": text}
