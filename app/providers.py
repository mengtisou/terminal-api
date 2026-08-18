"""Provider abstraction.

One interface, three backends. Every provider exposes:

    json_call(model, system, user, schema, max_tokens) -> dict
    stream(model, system, messages, tools, max_tokens)  -> event generator

Streaming events are normalised across providers:
    {"type": "text",     "text": str}
    {"type": "tool_use", "id": str, "name": str, "input": dict,
     "signature": str | None}   # Gemini 3 thought signature, echoed back verbatim
    {"type": "end",      "stop_reason": "tool_use" | "end"}

Message format is Anthropic-shaped internally; each provider translates.
That keeps chat.py provider-agnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Protocol

import httpx

# --- key resolution ----------------------------------------------------------
# Belt and braces. os.getenv is the normal path, but if .env was never loaded
# (wrong import order, missing python-dotenv, app/__init__.py not updated) we
# parse the file ourselves rather than reporting "no key" and leaving the user
# to guess which of five things went wrong.

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_file_cache: dict | None = None


def _from_file(name: str) -> str:
    global _file_cache
    if _file_cache is None:
        _file_cache = {}
        try:
            # utf-8-sig strips the BOM Notepad adds, which otherwise corrupts
            # the first key name into "\ufeffGEMINI_API_KEY".
            for line in _ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                _file_cache[k.strip()] = v.strip().strip("\"'")
        except OSError:
            pass
    return _file_cache.get(name, "")


def resolve_key(name: str) -> str:
    """Environment first, then .env directly."""
    return (os.getenv(name) or "").strip() or _from_file(name)


class ProviderError(RuntimeError):
    pass


class MissingKey(ProviderError):
    pass


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...
    def json_call(self, *, model, system, user, schema, max_tokens) -> dict: ...
    def stream(self, *, model, system, messages, tools, max_tokens) -> Iterator[dict]: ...


# ---------------------------------------------------------------- Anthropic

class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        self._client = None

    def available(self) -> bool:
        return bool(resolve_key("ANTHROPIC_API_KEY"))

    @property
    def client(self):
        if self._client is None:
            import anthropic
            key = resolve_key("ANTHROPIC_API_KEY")
            if not key:
                raise MissingKey("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def json_call(self, *, model, system, user, schema, max_tokens=1500) -> dict:
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason == "refusal":
            raise ProviderError("model refused the request")
        return json.loads("".join(b.text for b in resp.content if b.type == "text"))

    def stream(self, *, model, system, messages, tools=None, max_tokens=2000):
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools

        with self.client.messages.stream(**kwargs) as s:
            for chunk in s.text_stream:
                yield {"type": "text", "text": chunk}
            final = s.get_final_message()

        for b in final.content:
            if b.type == "tool_use":
                yield {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
        yield {
            "type": "end",
            "stop_reason": "tool_use" if final.stop_reason == "tool_use" else "end",
        }


# ------------------------------------------------------------------- Gemini

def _to_gemini_schema(s: dict) -> dict:
    """Gemini uses an OpenAPI subset, not full JSON Schema.

    Differences that bite: no additionalProperties, and nullable is a flag
    rather than a union type.
    """
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k == "additionalProperties":
            continue
        if k == "type" and isinstance(v, list):
            real = [t for t in v if t != "null"]
            out["type"] = real[0] if real else "string"
            if "null" in v:
                out["nullable"] = True
            continue
        if k == "properties":
            out["properties"] = {pk: _to_gemini_schema(pv) for pk, pv in v.items()}
            continue
        if k == "items":
            out["items"] = _to_gemini_schema(v)
            continue
        out[k] = v
    return out


class GeminiProvider:
    """Google AI Studio. Free tier with real daily limits - good for testing."""

    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def available(self) -> bool:
        return bool(resolve_key("GEMINI_API_KEY"))

    def _key(self) -> str:
        k = resolve_key("GEMINI_API_KEY")
        if not k:
            raise MissingKey("GEMINI_API_KEY not set")
        return k

    def _auth(self) -> dict:
        """Auth keys (AQ. prefix, the current default) must go in the header.
        The legacy ?key= query param only works for old AIza standard keys."""
        return {"x-goog-api-key": self._key(), "Content-Type": "application/json"}

    def json_call(self, *, model, system, user, schema, max_tokens=1500) -> dict:
        r = httpx.post(
            f"{self.BASE}/{model}:generateContent",
            headers=self._auth(),
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": _to_gemini_schema(schema),
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.3,
                },
            },
            timeout=90,
        )
        if r.status_code != 200:
            raise ProviderError(f"gemini {r.status_code}: {r.text[:300]}")

        cands = r.json().get("candidates") or []
        if not cands:
            raise ProviderError(f"gemini returned no candidates: {r.text[:300]}")
        text = "".join(p.get("text", "") for p in cands[0]["content"]["parts"])
        return json.loads(text)

    def _translate(self, system, messages, tools):
        """Anthropic message shape -> Gemini contents."""
        contents = []
        for m in messages:
            parts, role = [], ("user" if m["role"] == "user" else "model")
            content = m["content"]
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                for b in content:
                    t = b.get("type")
                    if t == "text":
                        parts.append({"text": b["text"]})
                    elif t == "tool_use":
                        part = {"functionCall": {"name": b["name"], "args": b["input"]}}
                        if b.get("signature"):
                            part["thoughtSignature"] = b["signature"]
                        parts.append(part)
                    elif t == "tool_result":
                        role = "user"
                        try:
                            payload = json.loads(b["content"])
                        except (TypeError, json.JSONDecodeError):
                            payload = {"result": b["content"]}
                        parts.append({
                            "functionResponse": {
                                "name": b.get("name", "tool"),
                                "response": payload,
                            }
                        })
            if parts:
                contents.append({"role": role, "parts": parts})

        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
        }
        if tools:
            body["tools"] = [{
                "functionDeclarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": _to_gemini_schema(t["input_schema"]),
                    }
                    for t in tools
                ]
            }]
        return body

    def stream(self, *, model, system, messages, tools=None, max_tokens=2000):
        body = self._translate(system, messages, tools)
        body["generationConfig"] = {"maxOutputTokens": max_tokens, "temperature": 0.4}

        calls, saw_call = [], False
        # Gemini 3 attaches an encrypted thoughtSignature to the model's
        # reasoning parts. It must be returned verbatim in the next request or
        # the API rejects the whole turn with a 400. On parallel calls only the
        # FIRST functionCall carries it, so carry it forward.
        pending_sig = None
        with httpx.stream(
            "POST",
            f"{self.BASE}/{model}:streamGenerateContent",
            headers=self._auth(),
            params={"alt": "sse"},
            json=body,
            timeout=120,
        ) as r:
            if r.status_code != 200:
                raise ProviderError(f"gemini {r.status_code}: {r.read()[:300]}")

            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                for cand in chunk.get("candidates", []):
                    for p in cand.get("content", {}).get("parts", []):
                        sig = p.get("thoughtSignature") or p.get("thought_signature")
                        if "text" in p:
                            yield {"type": "text", "text": p["text"]}
                            if sig:
                                pending_sig = sig
                        elif "functionCall" in p:
                            saw_call = True
                            fc = p["functionCall"]
                            calls.append({
                                "type": "tool_use",
                                "id": f"gem_{len(calls)}",
                                "name": fc["name"],
                                "input": fc.get("args", {}),
                                "signature": sig or pending_sig,
                            })
                            pending_sig = None

        for c in calls:
            yield c
        yield {"type": "end", "stop_reason": "tool_use" if saw_call else "end"}


# ------------------------------------------------------------------- OpenAI

class OpenAIProvider:
    name = "openai"
    BASE = "https://api.openai.com/v1"

    def available(self) -> bool:
        return bool(resolve_key("OPENAI_API_KEY"))

    def _headers(self):
        k = resolve_key("OPENAI_API_KEY")
        if not k:
            raise MissingKey("OPENAI_API_KEY not set")
        return {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}

    def json_call(self, *, model, system, user, schema, max_tokens=1500) -> dict:
        strict = dict(schema)
        strict.setdefault("additionalProperties", False)
        r = httpx.post(
            f"{self.BASE}/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "max_completion_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": strict, "strict": True},
                },
            },
            timeout=90,
        )
        if r.status_code != 200:
            raise ProviderError(f"openai {r.status_code}: {r.text[:300]}")
        return json.loads(r.json()["choices"][0]["message"]["content"])

    def _translate(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            text_parts, tool_calls, results = [], [], []
            for b in content:
                if b.get("type") == "text":
                    text_parts.append(b["text"])
                elif b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    })
                elif b.get("type") == "tool_result":
                    results.append({
                        "role": "tool", "tool_call_id": b["tool_use_id"],
                        "content": b["content"],
                    })
            if text_parts or tool_calls:
                msg = {"role": m["role"], "content": "\n".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            out.extend(results)
        return out

    def stream(self, *, model, system, messages, tools=None, max_tokens=2000):
        body = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": self._translate(system, messages),
            "stream": True,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t["description"],
                    "parameters": t["input_schema"]}}
                for t in tools
            ]

        acc: dict[int, dict] = {}
        with httpx.stream("POST", f"{self.BASE}/chat/completions",
                          headers=self._headers(), json=body, timeout=120) as r:
            if r.status_code != 200:
                raise ProviderError(f"openai {r.status_code}: {r.read()[:300]}")

            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                if delta.get("content"):
                    yield {"type": "text", "text": delta["content"]}

                # Tool calls arrive fragmented across chunks; accumulate by index.
                for tc in delta.get("tool_calls", []):
                    i = tc.get("index", 0)
                    slot = acc.setdefault(i, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

        for slot in acc.values():
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool_use", "id": slot["id"], "name": slot["name"], "input": args}
        yield {"type": "end", "stop_reason": "tool_use" if acc else "end"}


class XAIProvider(OpenAIProvider):
    """xAI Grok. The API is OpenAI-compatible, so only the base URL and key
    differ - everything else is inherited."""

    name = "xai"
    BASE = "https://api.x.ai/v1"

    def available(self) -> bool:
        return bool(resolve_key("XAI_API_KEY"))

    def _headers(self):
        k = resolve_key("XAI_API_KEY")
        if not k:
            raise MissingKey("XAI_API_KEY not set")
        return {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek. OpenAI-compatible API, roughly a tenth of Claude's price.

    One difference that matters: deepseek-chat supports JSON mode but NOT
    json_schema, so structured output is enforced by putting the schema in the
    prompt and validating what comes back.
    """

    name = "deepseek"
    BASE = "https://api.deepseek.com/v1"

    def available(self) -> bool:
        return bool(resolve_key("DEEPSEEK_API_KEY"))

    def _headers(self):
        k = resolve_key("DEEPSEEK_API_KEY")
        if not k:
            raise MissingKey("DEEPSEEK_API_KEY not set")
        return {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}

    def json_call(self, *, model, system, user, schema, max_tokens=1500) -> dict:
        r = httpx.post(
            f"{self.BASE}/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "messages": [
                    {"role": "system",
                     "content": system + "\n\nRespond with a single JSON object "
                     "matching this schema exactly. No markdown fences, no "
                     "commentary before or after:\n"
                     + json.dumps(schema, indent=1)},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        if r.status_code != 200:
            raise ProviderError(f"deepseek {r.status_code}: {r.text[:300]}")

        text = r.json()["choices"][0]["message"]["content"].strip()
        # JSON mode still occasionally wraps output in fences.
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)

        missing = [k for k in schema.get("required", []) if k not in parsed]
        if missing:
            raise ProviderError(f"deepseek response missing required keys: {missing}")
        return parsed


REGISTRY: dict[str, Provider] = {
    "anthropic": AnthropicProvider(),
    "gemini": GeminiProvider(),
    "openai": OpenAIProvider(),
    "xai": XAIProvider(),
    "deepseek": DeepSeekProvider(),
}


def get(name: str) -> Provider:
    if name not in REGISTRY:
        raise ProviderError(f"unknown provider '{name}'. options: {list(REGISTRY)}")
    return REGISTRY[name]


def status() -> dict:
    return {n: ("ready" if p.available() else "no key") for n, p in REGISTRY.items()}
