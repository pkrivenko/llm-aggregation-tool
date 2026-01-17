from __future__ import annotations
import os
import time
from pathlib import Path
import httpx

# Secure key loading: environment variable first, then file fallback
def _load_key(env_var: str, file_name: str) -> str | None:
    """Load API key from environment variable or file."""
    key = os.environ.get(env_var)
    if key:
        return key.strip()

    key_file = Path(__file__).parent.parent / file_name
    if key_file.exists():
        return key_file.read_text().strip()

    return None

OPENROUTER_API_KEY = _load_key("OPENROUTER_API_KEY", "OpenRouterAPIKey.txt")
GOOGLE_API_KEY = _load_key("GOOGLE_API_KEY", "GoogleAPIKey.txt")
OPENAI_API_KEY = _load_key("OPENAI_API_KEY", "OpenAIAPIKey.txt")
ANTHROPIC_API_KEY = _load_key("ANTHROPIC_API_KEY", "AnthropicAPIKey.txt")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"


def _is_google_model(model_id: str) -> bool:
    """Check if model should use Google API directly."""
    return model_id.startswith("google/") or model_id.startswith("gemini")


def _is_openai_model(model_id: str) -> bool:
    """Check if model should use OpenAI API directly."""
    return model_id.startswith("openai/") or model_id.startswith("gpt-")


def _is_anthropic_model(model_id: str) -> bool:
    """Check if model should use Anthropic API directly."""
    return model_id.startswith("anthropic/") or model_id.startswith("claude-")


def _convert_to_google_format(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages to Google Generative AI format."""
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content if isinstance(content, str) else str(content)
            continue

        # Handle multimodal content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # Parse data URL: data:image/jpeg;base64,<data>
                        mime_end = image_url.find(";")
                        data_start = image_url.find(",") + 1
                        mime_type = image_url[5:mime_end] if mime_end > 5 else "image/jpeg"
                        base64_data = image_url[data_start:]
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        })
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": parts
            })
        else:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": content}]
            })

    return contents, system_instruction


async def _call_google(
    messages: list[dict],
    model_id: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float | None = None,
) -> dict:
    """Call Google Generative AI API directly."""
    if not GOOGLE_API_KEY:
        return {
            "status": "error",
            "response_text": None,
            "latency_ms": 0,
            "http_status": None,
            "error_message": "Google API key not found. Set GOOGLE_API_KEY env var or create GoogleAPIKey.txt",
            "usage": None,
            "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            "response_json": None,
        }

    # Extract model name from model_id (e.g., "google/gemini-2.0-flash-001" -> "gemini-2.0-flash-001")
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Convert messages to Google format
    contents, system_instruction = _convert_to_google_format(messages)

    # Build request body
    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        }
    }

    if temperature is not None:
        body["generationConfig"]["temperature"] = temperature

    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"{GOOGLE_ENDPOINT}/{model_name}:generateContent?key={GOOGLE_API_KEY}"

    start_ms = time.perf_counter() * 1000
    result = {
        "status": "error",
        "response_text": None,
        "latency_ms": 0,
        "http_status": None,
        "error_message": None,
        "usage": None,
        "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        "response_json": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=body
            )
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["http_status"] = resp.status_code

        if resp.status_code != 200:
            result["error_message"] = resp.text
            return result

        data = resp.json()
        result["response_json"] = data

        # Extract text from Google response
        candidates = data.get("candidates", [])
        if not candidates:
            result["error_message"] = "No candidates in response"
            return result

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = "".join(p.get("text", "") for p in parts)

        if not text or not text.strip():
            result["error_message"] = "Empty response content"
            return result

        result["status"] = "ok"
        result["response_text"] = text

        # Extract usage metadata
        usage_metadata = data.get("usageMetadata", {})
        result["usage"] = {
            "prompt_tokens": usage_metadata.get("promptTokenCount"),
            "completion_tokens": usage_metadata.get("candidatesTokenCount"),
            "total_tokens": usage_metadata.get("totalTokenCount"),
            "cost_usd": None,  # Google API doesn't return cost directly
        }

    except httpx.TimeoutException:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["status"] = "timeout"
        result["error_message"] = f"Request timed out after {timeout_s}s"

    except httpx.RequestError as e:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["error_message"] = str(e)

    return result


async def _call_openai(
    messages: list[dict],
    model_id: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float | None = None,
) -> dict:
    """Call OpenAI API directly."""
    if not OPENAI_API_KEY:
        return {
            "status": "error",
            "response_text": None,
            "latency_ms": 0,
            "http_status": None,
            "error_message": "OpenAI API key not found. Set OPENAI_API_KEY env var or create OpenAIAPIKey.txt",
            "usage": None,
            "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            "response_json": None,
        }

    # Extract model name (e.g., "openai/gpt-4o" -> "gpt-4o")
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": max_tokens,  # Newer OpenAI models use this instead of max_tokens
    }
    if temperature is not None:
        body["temperature"] = temperature

    start_ms = time.perf_counter() * 1000
    result = {
        "status": "error",
        "response_text": None,
        "latency_ms": 0,
        "http_status": None,
        "error_message": None,
        "usage": None,
        "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        "response_json": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(OPENAI_ENDPOINT, headers=headers, json=body)
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["http_status"] = resp.status_code

        if resp.status_code != 200:
            result["error_message"] = resp.text
            return result

        data = resp.json()
        result["response_json"] = data

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            result["error_message"] = "Empty response content"
            return result

        result["status"] = "ok"
        result["response_text"] = content

        usage = data.get("usage", {})
        result["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": None,  # OpenAI API doesn't return cost directly
        }

    except httpx.TimeoutException:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["status"] = "timeout"
        result["error_message"] = f"Request timed out after {timeout_s}s"

    except httpx.RequestError as e:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["error_message"] = str(e)

    return result


async def _call_anthropic(
    messages: list[dict],
    model_id: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float | None = None,
) -> dict:
    """Call Anthropic API directly."""
    if not ANTHROPIC_API_KEY:
        return {
            "status": "error",
            "response_text": None,
            "latency_ms": 0,
            "http_status": None,
            "error_message": "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or create AnthropicAPIKey.txt",
            "usage": None,
            "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            "response_json": None,
        }

    # Extract model name (e.g., "anthropic/claude-3-5-haiku-latest" -> "claude-3-5-haiku-latest")
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Extract system message if present
    system_content = None
    api_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            api_messages.append(msg)

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": model_name,
        "messages": api_messages,
        "max_tokens": max_tokens,
    }
    if system_content:
        body["system"] = system_content
    if temperature is not None:
        body["temperature"] = temperature

    start_ms = time.perf_counter() * 1000
    result = {
        "status": "error",
        "response_text": None,
        "latency_ms": 0,
        "http_status": None,
        "error_message": None,
        "usage": None,
        "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        "response_json": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(ANTHROPIC_ENDPOINT, headers=headers, json=body)
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["http_status"] = resp.status_code

        if resp.status_code != 200:
            result["error_message"] = resp.text
            return result

        data = resp.json()
        result["response_json"] = data

        # Extract text from Anthropic response
        content_blocks = data.get("content", [])
        text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
        content = "".join(text_parts)

        if not content or not content.strip():
            result["error_message"] = "Empty response content"
            return result

        result["status"] = "ok"
        result["response_text"] = content

        usage = data.get("usage", {})
        result["usage"] = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            "cost_usd": None,  # Anthropic API doesn't return cost directly
        }

    except httpx.TimeoutException:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["status"] = "timeout"
        result["error_message"] = f"Request timed out after {timeout_s}s"

    except httpx.RequestError as e:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["error_message"] = str(e)

    return result


async def _call_openrouter(
    messages: list[dict],
    model_id: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float | None = None,
) -> dict:
    """Call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        return {
            "status": "error",
            "response_text": None,
            "latency_ms": 0,
            "http_status": None,
            "error_message": "OpenRouter API key not found. Set OPENROUTER_API_KEY env var or create OpenRouterAPIKey.txt",
            "usage": None,
            "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            "response_json": None,
        }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "LLM Aggregation Tool",
    }

    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if temperature is not None:
        body["temperature"] = temperature

    start_ms = time.perf_counter() * 1000
    result = {
        "status": "error",
        "response_text": None,
        "latency_ms": 0,
        "http_status": None,
        "error_message": None,
        "usage": None,
        "request": {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        "response_json": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(OPENROUTER_ENDPOINT, headers=headers, json=body)
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["http_status"] = resp.status_code

        if resp.status_code != 200:
            result["error_message"] = resp.text
            return result

        data = resp.json()
        result["response_json"] = data

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            result["error_message"] = "Empty response content"
            return result

        result["status"] = "ok"
        result["response_text"] = content

        usage = data.get("usage", {})
        result["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": data.get("usage", {}).get("cost") or data.get("cost"),
        }

    except httpx.TimeoutException:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["status"] = "timeout"
        result["error_message"] = f"Request timed out after {timeout_s}s"

    except httpx.RequestError as e:
        result["latency_ms"] = time.perf_counter() * 1000 - start_ms
        result["error_message"] = str(e)

    return result


async def call_openrouter(
    messages: list[dict],
    model_id: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float | None = None,
) -> dict:
    """
    Call LLM API - routes to appropriate provider:
    - Google models (google/*, gemini*) → Google Generative AI API
    - OpenAI models (openai/*, gpt-*) → OpenAI API
    - Anthropic models (anthropic/*, claude-*) → Anthropic API
    - Everything else → OpenRouter API
    """
    if _is_google_model(model_id) and GOOGLE_API_KEY:
        return await _call_google(messages, model_id, timeout_s, max_tokens, temperature)
    elif _is_openai_model(model_id) and OPENAI_API_KEY:
        return await _call_openai(messages, model_id, timeout_s, max_tokens, temperature)
    elif _is_anthropic_model(model_id) and ANTHROPIC_API_KEY:
        return await _call_anthropic(messages, model_id, timeout_s, max_tokens, temperature)
    else:
        return await _call_openrouter(messages, model_id, timeout_s, max_tokens, temperature)
