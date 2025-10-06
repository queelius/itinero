"""
Adapters that connect core abstractions to concrete implementations.

This module bridges the clean core domain with messy external dependencies
like Playwright, Ollama, and specific model configurations.

Following the Ports & Adapters (Hexagonal) architecture pattern.
"""

import json
import hashlib
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

from core import LLM, Executor, PromptBuilder, ActionParser, ExecutionResult, Action
from model_prompts import MODEL_CONFIGS, ModelConfig


# ============================================================================
# LLM Adapters
# ============================================================================

class OllamaLLM:
    """Adapter for Ollama API."""

    def __init__(self, model: str = "gemma3n:e2b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._config = MODEL_CONFIGS.get(model, MODEL_CONFIGS["default"])

    def generate(self, prompt: str) -> str:
        """Generate response from Ollama."""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self._config.temperature,
                        "num_predict": self._config.max_tokens
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")

            return '{"type":"pass"}'

        except Exception as e:
            print(f"LLM error: {e}")
            return '{"type":"pass"}'

    def _get_system_prompt(self) -> str:
        """Get model-specific system prompt."""
        if self.model == "gemma3n:e2b":
            return "Output JSON only. Example: {\"type\":\"click\",\"selector\":\"#btn\",\"value\":\"\"}"
        return "You are a web automation assistant. Output valid JSON only."


class CallableLLM:
    """Adapter for any callable that takes prompt and returns response."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def generate(self, prompt: str) -> str:
        return self.fn(prompt)


# ============================================================================
# Executor Adapters
# ============================================================================

class PlaywrightExecutor:
    """Adapter for Playwright execution."""

    def __init__(self, page=None):
        self.page = page
        self._retry_count = 0

    def set_page(self, page):
        """Set the Playwright page object."""
        self.page = page

    def execute(self, action_str: str) -> ExecutionResult:
        """Execute action using Playwright."""
        if self.page is None:
            return ExecutionResult(
                success=False,
                action=Action(type='unknown', selector=''),
                error="Page not set",
                error_type="ConfigError"
            )

        # Use existing executor logic
        from executor import execute_action
        result = execute_action(self.page, action_str)

        # Convert to our ExecutionResult format
        return ExecutionResult(
            success=result.success,
            action=self._parse_action_from_string(action_str),
            error=result.error,
            error_type=result.error_type,
            metadata={
                'execution_time': result.execution_time,
                'suggestion': result.suggestion
            }
        )

    def _parse_action_from_string(self, action_str: str) -> Action:
        """Parse action string back into Action object."""
        import re

        # Extract method and args
        match = re.match(r"page\.(\w+)\((.*)\)", action_str.strip())
        if not match:
            return Action(type='pass', selector='')

        method = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        args = []
        if args_str:
            # Simple parsing - split on comma outside quotes
            current = ""
            in_quotes = False
            for char in args_str:
                if char in ["'", '"'] and (not current or current[-1] != '\\'):
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    args.append(current.strip().strip("'\""))
                    current = ""
                    continue
                current += char
            if current:
                args.append(current.strip().strip("'\""))

        selector = args[0] if args else ""
        value = args[1] if len(args) > 1 else ""

        return Action(
            type=method,
            selector=selector,
            value=value
        )


# ============================================================================
# Prompt Builder Adapters
# ============================================================================

class ModelSpecificPromptBuilder:
    """Prompt builder that adapts to different model capabilities."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])
        self._cache = {}

    def build(self, context: Dict[str, Any]) -> str:
        """Build prompt optimized for the model."""
        # Create cache key
        cache_key = self._cache_key(context)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build based on model style
        if self.config.prompt_style == "minimal":
            prompt = self._build_minimal(context)
        elif self.config.prompt_style == "structured":
            prompt = self._build_structured(context)
        else:
            prompt = self._build_verbose(context)

        # Cache it
        if len(self._cache) < 100:
            self._cache[cache_key] = prompt

        return prompt

    def _build_minimal(self, ctx: Dict[str, Any]) -> str:
        """Minimal prompt for small models like gemma3n:e2b."""
        state = ctx.get('state', {})
        elements = state.get('elements', [])

        # Compress elements to minimal representation
        els = []
        for el in elements[:10]:
            if el.get('tag') in ['input', 'button', 'select']:
                tag = el.get('tag', '?')[0]
                sel = el.get('selector', '')
                val = el.get('value', '')
                v = f"={val[:10]}" if val else ":EMPTY"
                r = 'REQ' if el.get('required') else ''
                els.append(f"{tag}:{sel}{v}{r}")

        elements_str = "\n".join(els[:5])

        return f"""Form status:
{elements_str}

Goal: {ctx.get('goal', '')[:30]}
JSON: {{"type":"fill","selector":"#field","value":"val"}}"""

    def _build_structured(self, ctx: Dict[str, Any]) -> str:
        """Structured prompt for capable models."""
        state = ctx.get('state', {})
        elements = state.get('elements', [])

        # Format elements as compact JSON
        els = []
        for el in elements[:25]:
            if el.get('visible', False) and el.get('tag') in ['input', 'button', 'select', 'textarea']:
                els.append({
                    "tag": el.get('tag', ''),
                    "selector": el.get('selector', ''),
                    "value": el.get('value', ''),
                    "required": el.get('required', False),
                    "label": el.get('label', '')
                })

        elements_json = json.dumps(els[:15], separators=(',', ':'))

        # Get recent history
        history = ctx.get('history', [])
        last_action = None
        last_error = None

        if history:
            last_result = history[-1]
            last_action = f"{last_result.action.type}:{last_result.action.selector}"
            if not last_result.success:
                last_error = last_result.error[:50] if last_result.error else "Unknown error"

        return f"""{{
  "goal": "{ctx.get('goal', '')}",
  "elements": {elements_json},
  "last_action": "{last_action or 'none'}",
  "last_error": "{last_error or 'none'}"
}}

Response format:
{{
  "type": "click|fill|type|select|wait|pass",
  "selector": "#id or .class",
  "value": "if needed"
}}"""

    def _build_verbose(self, ctx: Dict[str, Any]) -> str:
        """Verbose prompt with detailed guidance."""
        # Fallback to structured for now
        return self._build_structured(ctx)

    def _cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key from context."""
        state = context.get('state', {})
        key_parts = [
            context.get('goal', '')[:30],
            str(len(state.get('elements', []))),
            str(len(context.get('history', [])))
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()[:16]


# ============================================================================
# Parser Adapters
# ============================================================================

class JSONActionParser:
    """Parse JSON responses into action data."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse response into action dictionary."""
        try:
            # Clean response
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            # Extract JSON
            if '{' in response:
                json_str = response[response.index('{'):response.rindex('}')+1]
                data = json.loads(json_str)

                # Handle abbreviated keys (for minimal models)
                if self.config.prompt_style == "minimal":
                    return self._expand_abbreviated(data)

                return data

        except Exception as e:
            pass

        # Fallback
        return {'type': 'pass', 'selector': '', 'value': ''}

    def _expand_abbreviated(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Expand abbreviated field names used by small models."""
        # Map abbreviated keys
        result = {}

        # Action type
        action = data.get('a') or data.get('action') or data.get('type', '')
        action_map = {
            'c': 'click', 'f': 'fill', 't': 'type',
            's': 'select', 'w': 'wait', 'p': 'pass',
            'set': 'fill', '': 'fill'
        }
        result['type'] = action_map.get(action, action) if action else 'fill'

        # Selector
        selector = data.get('s') or data.get('selector', '')
        # Fix selector if missing prefix
        if selector and not selector.startswith(('#', '.', '[')):
            selector = f"#{selector}"
        result['selector'] = selector

        # Value
        result['value'] = data.get('v') or data.get('value', '')

        return result


# ============================================================================
# Factory Functions
# ============================================================================

def create_prompt_builder(model_name: str) -> PromptBuilder:
    """Create appropriate prompt builder for model."""
    return ModelSpecificPromptBuilder(model_name)


def create_parser(model_name: str) -> ActionParser:
    """Create appropriate parser for model."""
    return JSONActionParser(model_name)


def create_llm(model_name: str = "gemma3n:e2b", **kwargs) -> LLM:
    """Create LLM adapter based on model name."""
    if model_name.startswith("ollama:"):
        model = model_name.split(":", 1)[1]
        return OllamaLLM(model, **kwargs)

    # Default to Ollama
    return OllamaLLM(model_name, **kwargs)
