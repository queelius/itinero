"""
Model-specific prompt optimization system.

Each model gets its own optimized prompts and settings.
No backward compatibility - pure performance focus.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    max_tokens: int
    supports_json: bool
    supports_cot: bool  # Chain of thought
    temperature: float
    prompt_style: str  # "minimal", "structured", "verbose"
    special_tokens: Dict[str, str] = None


# Model-specific configurations
MODEL_CONFIGS = {
    "gemma3n:e2b": ModelConfig(
        name="gemma3n:e2b",
        max_tokens=100,
        supports_json=True,
        supports_cot=False,  # Too small for CoT
        temperature=0.3,
        prompt_style="minimal",
        special_tokens={"separator": "→"}
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        max_tokens=500,
        supports_json=True,
        supports_cot=True,
        temperature=0.5,
        prompt_style="structured",
        special_tokens=None
    ),
    "claude-3": ModelConfig(
        name="claude-3",
        max_tokens=500,
        supports_json=True,
        supports_cot=True,
        temperature=0.5,
        prompt_style="structured",
        special_tokens=None
    ),
    "llama-70b": ModelConfig(
        name="llama-70b",
        max_tokens=300,
        supports_json=True,
        supports_cot=True,
        temperature=0.4,
        prompt_style="structured",
        special_tokens=None
    ),
    "default": ModelConfig(
        name="default",
        max_tokens=200,
        supports_json=True,
        supports_cot=False,
        temperature=0.5,
        prompt_style="structured",
        special_tokens=None
    )
}


class ModelPrompts:
    """Model-specific prompt templates."""

    def __init__(self, model_name: str = "default"):
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])
        self.cache = {}  # Prompt cache

    def get_action_prompt(self, context: Dict[str, Any]) -> str:
        """Get action prediction prompt for specific model."""

        if self.config.prompt_style == "minimal":
            return self._minimal_action_prompt(context)
        elif self.config.prompt_style == "structured":
            return self._structured_action_prompt(context)
        else:
            return self._verbose_action_prompt(context)

    def _minimal_action_prompt(self, ctx: Dict[str, Any]) -> str:
        """Ultra-minimal prompt for small models like gemma3n:e2b."""

        # Use abbreviations and codes
        elements = self._compress_elements(ctx.get('elements', []))

        prompt = f"""GOAL:{ctx['goal']}
ELS:{elements}
LAST:{ctx.get('last_action', 'none')}
ERR:{ctx.get('last_error', 'none')[:20]}

JSON:
{{"a":"action","s":"selector","v":"value"}}"""

        return prompt

    def _structured_action_prompt(self, ctx: Dict[str, Any]) -> str:
        """Structured prompt for medium/large models."""

        elements = self._format_elements_json(ctx.get('elements', []))

        prompt = f"""{{
  "goal": "{ctx['goal']}",
  "elements": {elements},
  "history": {{
    "last_success": "{ctx.get('last_success', '')}",
    "last_error": "{(ctx.get('last_error', '') or '')[:50]}"
  }},
  "request": "next_action"
}}

Response format:
{{
  "action": "click|fill|type|select|wait|check|pass",
  "selector": "#id or .class",
  "value": "if needed"
}}"""

        return prompt

    def _verbose_action_prompt(self, ctx: Dict[str, Any]) -> str:
        """Full verbose prompt (fallback)."""
        # Only used if specifically requested
        return self._structured_action_prompt(ctx)

    def _compress_elements(self, elements: List[Dict]) -> str:
        """Compress elements to minimal representation."""

        lines = []
        for el in elements[:15]:  # Limit for small models
            tag = el.get('tag', '?')[0]  # First letter only
            sel = el.get('selector', '')
            val = 'V' if el.get('value') else 'E'  # V=has value, E=empty
            req = 'R' if el.get('required') else ''  # R=required

            lines.append(f"{tag}:{sel}:{val}{req}")

        return ",".join(lines)

    def _format_elements_json(self, elements: List[Dict]) -> str:
        """Format elements as compact JSON."""

        els = []
        for el in elements[:25]:
            els.append({
                "t": el.get('tag', ''),
                "s": el.get('selector', ''),
                "v": bool(el.get('value')),
                "r": el.get('required', False)
            })

        return json.dumps(els, separators=(',', ':'))

    def parse_response(self, response: str, model_name: str = None) -> Dict[str, Any]:
        """Parse model response based on model type."""

        config = MODEL_CONFIGS.get(model_name, self.config)

        if config.prompt_style == "minimal":
            return self._parse_minimal_response(response)
        else:
            return self._parse_json_response(response)

    def _parse_minimal_response(self, response: str) -> Dict[str, Any]:
        """Parse minimal format response."""

        try:
            # Clean response first
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            # Try JSON first
            if '{' in response:
                # Extract just the first JSON object (gemma sometimes adds extra text)
                json_str = response[response.index('{'):response.rindex('}')+1]
                # Remove any trailing content after the JSON
                if '\n' in json_str:
                    json_str = json_str.split('\n')[0] + '}'
                data = json.loads(json_str)

                # Expand abbreviated keys and fix common issues
                action_map = {
                    'c': 'click', 'f': 'fill', 't': 'type',
                    's': 'select', 'w': 'wait', 'p': 'pass',
                    'set': 'fill',  # Common mistake
                    '': 'fill'  # Default to fill if empty
                }

                # Get action - default to fill if not specified
                action = data.get('a', '')
                if not action and data.get('v'):  # If there's a value, probably fill
                    action = 'fill'

                action = action_map.get(action, action)

                # Fix selector if missing # or .
                selector = data.get('s', '')
                if selector and not selector.startswith(('#', '.', '[')):
                    # Assume it's an ID if no prefix
                    selector = f"#{selector}"

                return {
                    'action': action if action else 'fill',
                    'selector': selector,
                    'value': data.get('v', '')
                }
        except Exception as e:
            print(f"Parse error: {e}")

        # Fallback to text parsing
        return {'action': 'pass', 'selector': '', 'value': ''}

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON format response."""

        try:
            # Extract JSON from response
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '{' in response:
                response = response[response.index('{'):response.rindex('}')+1]

            return json.loads(response)
        except:
            # Fallback
            return {'action': 'pass', 'selector': '', 'value': ''}


# Optimized prompts for gemma3n:e2b specifically
GEMMA3N_PROMPTS = {
    "action": """G:{goal}
E:{elements}
L:{last}
→""",

    "fill": """FILL:{selector}
VAL:{value}
→page.fill('{selector}','{value}')""",

    "click": """CLK:{selector}
→page.click('{selector}')""",

    "error_recovery": """ERR:{error}
TRY:{alternatives}
→""",

    "validation": """FORM:{fields}
REQ:{required}
→"""
}


class PromptCache:
    """Cache frequently used prompts to reduce generation time."""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size

    def get_or_create(self, key: str, generator: callable) -> str:
        """Get cached prompt or generate new one."""

        if key in self.cache:
            return self.cache[key]

        prompt = generator()

        # Evict old entries if cache full
        if len(self.cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[key] = prompt
        return prompt

    def clear(self):
        """Clear cache."""
        self.cache.clear()


class ActionBuilder:
    """Build Playwright actions from JSON responses."""

    @staticmethod
    def build(action_data: Dict[str, Any]) -> str:
        """Build Playwright command from action data."""

        action = action_data.get('action', '')
        selector = action_data.get('selector', '')
        value = action_data.get('value', '')

        # Map actions to Playwright commands
        if action == 'click':
            return f"page.click('{selector}')"
        elif action == 'fill':
            return f"page.fill('{selector}', '{value}')"
        elif action == 'type':
            return f"page.type('{selector}', '{value}')"
        elif action == 'select':
            return f"page.select_option('{selector}', '{value}')"
        elif action == 'check':
            return f"page.check('{selector}')"
        elif action == 'uncheck':
            return f"page.uncheck('{selector}')"
        elif action == 'wait':
            return f"page.wait_for_selector('{selector}')"
        elif action == 'pass':
            return "pass"
        else:
            # Try to use raw command if provided
            if 'command' in action_data:
                return action_data['command']
            return "pass"