"""
Compatibility shim for the `ollama` package when it's not
installed in the environment. This provides a minimal
`chat()` function matching the shape used by the project's
agent modules and proxies to `agents.llm_client.call_llm`.

Notes:
- If you later install the official `ollama` Python package,
  this file will still be imported first (project root), so
  consider removing it to use the real SDK.
"""

from typing import Any, Dict, List

try:
    # Import the project's HTTP-based LLM client
    from agents import llm_client
except Exception:
    # Fallback import style
    import agents.llm_client as llm_client


def _extract_prompt(messages: Any) -> str:
    if isinstance(messages, list):
        # join all message contents (preserve order)
        parts = [m.get("content", "") if isinstance(m, dict) else str(m)
                 for m in messages]
        return "\n\n".join(parts).strip()
    if isinstance(messages, dict):
        return messages.get("content", "")
    return str(messages)


def chat(model: str, messages: Any = None, options: Dict = None, **kwargs) -> Dict:
    """
    Minimal replacement for `ollama.chat` used in this repo.

    Args:
        model: model name (passed to `agents.llm_client.call_llm`)
        messages: list/dict/string containing the prompt content
        options: dict with optional `temperature` etc.

    Returns:
        dict compatible with the callers in `agents/*`:
          {'message': {'content': '<model response string>'}}
    """

    prompt = _extract_prompt(messages) if messages is not None else ""

    temperature = None
    if isinstance(options, dict):
        temperature = options.get("temperature")

    if temperature is None:
        temperature = 0.2

    try:
        # Use the project's llm_client which handles HTTP, caching, and fallbacks
        # prefer the requested model, then fallback models
        models = [m for m in (model, "phi3", "mistral") if m]
        response_text = llm_client.call_llm(prompt, models=models, timeout=10, temperature=temperature)
    except Exception as e:
        # On failure return empty string (no silent numeric corruption)
        print("ollama shim error:", e)
        response_text = ""

    return {"message": {"content": str(response_text)}}


# Keep a small API surface to avoid surprises
__all__ = ["chat"]
