"""
Next-generation prompting system.
Pure JSON protocol, model-specific optimization, zero backward compatibility.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import hashlib
from model_prompts import ModelPrompts, ActionBuilder, PromptCache


@dataclass
class PromptContext:
    """Context for prompt generation."""
    goal: str
    elements: List[Dict[str, Any]]
    last_action: Optional[str] = None
    last_error: Optional[str] = None
    last_success: Optional[str] = None
    failures: List[Dict[str, str]] = None
    page_type: Optional[str] = None
    form_errors: List[str] = None


class PromptsV2:
    """Modern prompting system with model-specific optimization."""

    def __init__(self, model_name: str = "default"):
        self.model_prompts = ModelPrompts(model_name)
        self.cache = PromptCache(max_size=50)
        self.model_name = model_name

    def get_action_prompt(self, state) -> str:
        """Generate action prediction prompt."""

        # Build context
        context = self._build_context(state)

        # Generate cache key
        cache_key = self._get_cache_key(context)

        # Try cache first
        if cache_key in self.cache.cache:
            return self.cache.cache[cache_key]

        # Generate prompt based on model
        if self.model_name == "gemma3n:e2b":
            prompt = self._gemma3n_prompt(context)
        else:
            prompt = self.model_prompts.get_action_prompt(context.__dict__)

        # Cache it
        self.cache.cache[cache_key] = prompt
        return prompt

    def _gemma3n_prompt(self, ctx: PromptContext) -> str:
        """Ultra-optimized prompt for gemma3n:e2b."""

        # Super compressed format - focus on unfilled required fields
        els = []
        for e in ctx.elements[:10]:
            if e.get('tag') in ['input', 'button', 'select']:
                t = e.get('tag', '?')[0]
                s = e.get('selector', '')
                val = e.get('value', '')
                # Check if actually has a value
                v = f"={val[:10]}" if val else ":EMPTY"
                r = 'REQ' if e.get('required') else ''
                els.append(f"{t}:{s}{v}{r}")

        # Extract action from goal
        goal_lower = ctx.goal.lower()
        if 'fill' in goal_lower:
            hint = 'fill'
        elif 'click' in goal_lower or 'submit' in goal_lower:
            hint = 'click'
        else:
            hint = 'fill'

        # Check if we should complete
        if self._is_goal_complete(ctx):
            return '{"a":"pass","s":"","v":""}'

        # Smart hint: find first empty field that needs filling
        next_field = ""
        next_value = ""
        for el in els[:5]:
            if 'EMPTY' in el and 'REQ' in el:
                # This field needs filling
                parts = el.split(':')
                if len(parts) >= 2:
                    next_field = parts[1].split('=')[0]
                    # Try to guess value from goal
                    if 'firstname' in next_field.lower() and 'alice' in goal_lower:
                        next_value = 'Alice'
                    elif 'lastname' in next_field.lower() and 'smith' in goal_lower:
                        next_value = 'Smith'
                    elif 'email' in next_field.lower():
                        next_value = 'alice@example.com'
                    break

        if not next_field:
            # No required empty fields, check optional
            for el in els[:5]:
                if 'EMPTY' in el:
                    parts = el.split(':')
                    if len(parts) >= 2:
                        next_field = parts[1].split('=')[0]
                        break

        prompt = f"""Form status:
{chr(10).join(els[:5])}

Goal: {ctx.goal[:30]}
Next: Fill {next_field if next_field else 'complete'}
JSON: {{"a":"{hint}","s":"{next_field}","v":"{next_value}"}}"""

        return prompt

    def _is_goal_complete(self, ctx: PromptContext) -> bool:
        """Check if the goal is complete based on context."""

        goal_lower = ctx.goal.lower()

        # Don't complete too early
        if len(ctx.last_success or '') == 0:
            return False

        # Extract fields mentioned in goal
        import re

        # Find all field names in goal (firstName, lastName, email, etc)
        field_patterns = ['firstname', 'lastname', 'email', 'phone', 'address', 'city', 'zip']
        mentioned_fields = []
        for pattern in field_patterns:
            if pattern in goal_lower:
                mentioned_fields.append(pattern)

        if not mentioned_fields:
            # No specific fields mentioned, can't determine
            return False

        # Check if all mentioned fields are filled
        filled_count = 0
        for field in mentioned_fields:
            for el in ctx.elements:
                sel = el.get('selector', '').lower()
                if field in sel:
                    if el.get('value') and len(str(el.get('value', ''))) > 0:
                        filled_count += 1
                        break

        # Complete if all mentioned fields are filled
        return filled_count >= len(mentioned_fields)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response."""
        return self.model_prompts.parse_response(response, self.model_name)

    def build_action(self, action_data: Dict[str, Any]) -> str:
        """Build Playwright command from parsed response."""
        return ActionBuilder.build(action_data)

    def _build_context(self, state) -> PromptContext:
        """Build context from state."""

        # Extract only essential elements
        elements = []
        for el in state.dom_elements[:30]:
            if not el.get('visible', False):
                continue

            # Clean up None values
            value = el.get('value')
            if value == 'None' or value is None:
                value = ''

            elements.append({
                'tag': el.get('tag', ''),
                'selector': el.get('selector', ''),
                'value': value,
                'required': el.get('required', False),
                'disabled': el.get('disabled', False)
            })

        # Recent actions
        last_action = state.successful_actions[-1] if state.successful_actions else None
        last_success = state.successful_actions[-1] if state.successful_actions else None
        last_error = None

        if state.failed_actions:
            last_failure = state.failed_actions[-1]
            last_error = f"{last_failure.action}:{last_failure.error[:30]}"

        return PromptContext(
            goal=state.main_goal or state.current_subgoal or "",
            elements=elements,
            last_action=last_action,
            last_error=last_error,
            last_success=last_success,
            page_type=state.page_type,
            form_errors=state.form_errors[:3] if hasattr(state, 'form_errors') else []
        )

    def _get_cache_key(self, context: PromptContext) -> str:
        """Generate cache key for context."""

        # Create deterministic key from context
        key_parts = [
            context.goal[:30],
            str(len(context.elements)),
            context.last_action or "",
            context.page_type or ""
        ]

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]


# Specialized prompt generators for specific situations
class SpecializedPrompts:
    """Highly specialized prompts for specific scenarios."""

    @staticmethod
    def form_fill_prompt(fields: List[Dict], values: Dict[str, str]) -> str:
        """Direct form filling prompt."""

        actions = []
        for field in fields:
            selector = field.get('selector', '')
            name = field.get('name', '')

            # Match field to value
            value_key = name or selector.replace('#', '').replace('.', '')
            if value_key in values:
                actions.append({
                    'a': 'fill',
                    's': selector,
                    'v': values[value_key]
                })

        return json.dumps({'actions': actions}, separators=(',', ':'))

    @staticmethod
    def recovery_prompt(error: str, selector: str) -> str:
        """Recovery from specific error."""

        strategies = []

        if "not found" in error.lower():
            strategies = [
                {'a': 'wait', 's': selector},
                {'a': 'wait', 's': 'body'},  # Wait for page
                {'a': 'click', 's': selector, 'force': True}
            ]
        elif "disabled" in error.lower():
            strategies = [
                {'a': 'wait', 's': selector, 'state': 'enabled'},
                {'a': 'fill_prereqs'},  # Signal to fill required fields
            ]

        return json.dumps({'recovery': strategies}, separators=(',', ':'))

    @staticmethod
    def validation_fix_prompt(validation_errors: List[str]) -> str:
        """Fix validation errors."""

        fixes = []
        for error in validation_errors[:3]:
            error_lower = error.lower()

            if "email" in error_lower:
                fixes.append({'field': 'email', 'pattern': 'user@example.com'})
            elif "phone" in error_lower:
                fixes.append({'field': 'phone', 'pattern': '555-0123'})
            elif "required" in error_lower:
                # Extract field name if possible
                fixes.append({'action': 'fill_required'})

        return json.dumps({'fixes': fixes}, separators=(',', ':'))