"""
Optimized prompt templates for LLM-driven web automation.

Key improvements:
- Structured output with JSON schema for action prediction
- Chain-of-thought reasoning for complex decisions
- Concise prompts optimized for smaller models
- Few-shot examples with correct Playwright syntax
- Better error recovery strategies
"""

from typing import List, Dict, Any, Optional
from state import State, FailedAction, analyze_failures
import json


def format_dom_elements_compact(elements: List[Dict[str, Any]], max_elements: int = 30) -> str:
    """Format DOM elements in compact, efficient format for smaller models."""

    if not elements:
        return "No elements found"

    lines = []
    shown = 0

    # Group elements by type for better organization
    inputs = []
    buttons = []
    links = []
    selects = []
    others = []

    for el in elements[:max_elements]:
        if not el.get('visible', False):
            continue

        tag = el.get('tag', '')
        selector = el.get('selector', '')
        value = el.get('value', '')
        text = (el.get('text', '') or '')[:30]
        disabled = el.get('disabled', False)
        required = el.get('required', False)
        el_type = el.get('type', '')

        # Build compact element representation
        info = f"{selector}"

        # Add state info
        if tag in ['input', 'textarea', 'select']:
            if value:
                info += f"={value}"
            elif required:
                info += "[EMPTY,REQUIRED]"
            else:
                info += "[EMPTY]"
        elif text:
            info += f":{text}"

        if disabled:
            info += "[DISABLED]"

        # Categorize
        if tag == 'input':
            inputs.append(f"  {el_type or 'text'}: {info}")
        elif tag == 'button':
            buttons.append(f"  {info}")
        elif tag == 'a':
            links.append(f"  {info}")
        elif tag == 'select':
            selects.append(f"  {info}")
        else:
            others.append(f"  {tag}: {info}")

        shown += 1

    # Format output
    if inputs:
        lines.append("INPUTS:\n" + "\n".join(inputs))
    if buttons:
        lines.append("BUTTONS:\n" + "\n".join(buttons))
    if selects:
        lines.append("SELECTS:\n" + "\n".join(selects))
    if links:
        lines.append("LINKS:\n" + "\n".join(links[:5]))  # Limit links

    return "\n".join(lines)


def format_recent_actions_compact(successful: List[str], failed: List[FailedAction]) -> str:
    """Format action history in compact format."""

    lines = []

    # Last successful actions (very brief)
    if successful:
        recent_success = successful[-3:]
        lines.append("RECENT SUCCESS:")
        for action in recent_success:
            # Simplify action display
            action_short = action.replace('page.', '').split('(')[0]
            lines.append(f"  ✓ {action_short}")

    # Recent failures with key info only
    if failed:
        recent_fail = failed[-3:]
        lines.append("RECENT FAILURES:")
        for f in recent_fail:
            action_short = f.action.replace('page.', '').split('(')[0]
            error_key = f.error.split(':')[0] if ':' in f.error else f.error[:30]
            lines.append(f"  ✗ {action_short}: {error_key}")

    return "\n".join(lines)


def build_action_prompt_optimized(state: State) -> str:
    """
    Optimized prompt for action prediction with structured output.

    Key optimizations:
    - Shorter, more focused prompt
    - Structured thinking process
    - Clear examples with correct syntax
    - JSON output for reliability
    """

    # Detect common patterns that need special handling
    error_pattern = detect_error_pattern(state.failed_actions)
    strategy_hint = get_strategy_hint(state, error_pattern)

    # Build compact state representation
    elements = format_dom_elements_compact(state.dom_elements)
    history = format_recent_actions_compact(state.successful_actions, state.failed_actions)

    # Core prompt with chain-of-thought structure
    prompt = f"""GOAL: {state.current_subgoal or state.main_goal}

PAGE: {state.url.split('/')[-1]}

ELEMENTS:
{elements}

{history}

{strategy_hint}

THINK STEP BY STEP:
1. What needs to be done based on the goal?
2. What element should I interact with?
3. What's the correct Playwright command?

EXAMPLES:
- Fill input: page.fill('#email', 'user@example.com')
- Click button: page.click('button[type="submit"]')
- Select option: page.select_option('#country', 'USA')
- Type slowly: page.type('#search', 'query text')
- Wait: page.wait_for_selector('.success')
- Complete: pass

OUTPUT (single line, executable Python):
"""

    return prompt


def build_structured_action_prompt(state: State) -> str:
    """
    Build prompt that enforces structured JSON output for better parsing.

    This approach is more reliable with smaller models.
    """

    elements = format_dom_elements_compact(state.dom_elements, max_elements=20)

    # Analyze situation
    situation = analyze_situation(state)

    prompt = f"""Web automation task. Analyze and respond with JSON.

GOAL: {state.main_goal}
URL: {state.url}

VISIBLE ELEMENTS:
{elements}

LAST ACTION: {state.successful_actions[-1] if state.successful_actions else 'none'}
LAST ERROR: {state.failed_actions[-1].error[:50] if state.failed_actions else 'none'}

Analyze the situation and provide next action.

Response format:
{{
  "reasoning": "brief explanation of what to do",
  "target": "selector or element to interact with",
  "action": "click|fill|type|select|wait|check|uncheck|goto|pass",
  "value": "value if needed for fill/type/select",
  "command": "exact Playwright command"
}}

Example:
{{
  "reasoning": "need to fill email field",
  "target": "#email",
  "action": "fill",
  "value": "user@example.com",
  "command": "page.fill('#email', 'user@example.com')"
}}

JSON response:"""

    return prompt


def build_recovery_prompt(state: State, repeated_failure: str) -> str:
    """
    Specialized prompt for recovering from repeated failures.

    Focuses on alternative strategies.
    """

    prompt = f"""The action "{repeated_failure}" has failed 3+ times.

GOAL: {state.main_goal}

FAILED APPROACHES:
{format_failed_actions_brief(state.failed_actions[-5:])}

AVAILABLE ELEMENTS:
{format_dom_elements_compact(state.dom_elements, max_elements=15)}

Generate an ALTERNATIVE approach. Consider:
- Different selector (id vs class vs text)
- Different method (click vs JavaScript)
- Waiting for element first
- Filling prerequisites
- Using keyboard navigation

Alternative action:"""

    return prompt


def build_validation_prompt(state: State) -> str:
    """
    Prompt for handling form validation errors.
    """

    # Extract validation messages
    validation_msgs = extract_validation_messages(state)

    prompt = f"""Form validation issues detected.

VALIDATION ERRORS:
{validation_msgs}

FORM FIELDS:
{format_form_fields(state.dom_elements)}

Fix validation by:
1. Check required field formats
2. Fill missing required fields
3. Correct invalid values

Next action to fix validation:"""

    return prompt


# Helper functions

def detect_error_pattern(failed_actions: List[FailedAction]) -> str:
    """Detect common error patterns from failures."""

    if not failed_actions:
        return "none"

    recent = failed_actions[-5:]
    errors = [f.error.lower() for f in recent]

    if any("not found" in e or "timeout" in e for e in errors):
        return "element_not_found"
    elif any("disabled" in e for e in errors):
        return "disabled_element"
    elif any("validation" in e for e in errors):
        return "validation_error"
    elif any("not visible" in e for e in errors):
        return "hidden_element"
    else:
        return "unknown"


def get_strategy_hint(state: State, error_pattern: str) -> str:
    """Get strategic hint based on error pattern."""

    hints = {
        "element_not_found": "HINT: Element not found. Try: 1) Different selector 2) Wait for element 3) Check if page loaded",
        "disabled_element": "HINT: Element disabled. Fill required fields first or check prerequisites.",
        "validation_error": "HINT: Validation failed. Check field formats and requirements.",
        "hidden_element": "HINT: Element hidden. Try: 1) Scroll to element 2) Click parent to expand 3) Wait for visibility",
        "none": "",
        "unknown": "HINT: If stuck, try a different approach or selector."
    }

    # Check for repeated failures
    if state.failed_actions:
        last_action = state.failed_actions[-1].action
        repeat_count = sum(1 for f in state.failed_actions[-3:] if f.action == last_action)
        if repeat_count >= 2:
            return f"HINT: '{last_action}' failed {repeat_count} times. Must try different approach!"

    return hints.get(error_pattern, "")


def analyze_situation(state: State) -> Dict[str, Any]:
    """Analyze current situation for better decision making."""

    situation = {
        "has_form": any(el.get('tag') in ['input', 'select', 'textarea']
                       for el in state.dom_elements),
        "has_required_empty": any(el.get('required') and not el.get('value')
                                 for el in state.dom_elements),
        "has_submit": any('submit' in str(el.get('type', '')).lower() or
                         'submit' in str(el.get('text', '')).lower()
                         for el in state.dom_elements),
        "stuck": len(state.failed_actions) >= 3 and len(set(f.action for f in state.failed_actions[-3:])) == 1
    }

    return situation


def format_failed_actions_brief(failed_actions: List[FailedAction]) -> str:
    """Brief format for failed actions."""

    if not failed_actions:
        return "None"

    lines = []
    for f in failed_actions:
        action_brief = f.action.split('(')[0]
        error_brief = f.error.split('.')[0]
        lines.append(f"- {action_brief}: {error_brief}")

    return "\n".join(lines)


def format_form_fields(elements: List[Dict[str, Any]]) -> str:
    """Format only form field elements."""

    form_fields = [el for el in elements
                  if el.get('tag') in ['input', 'select', 'textarea']
                  and el.get('visible')]

    if not form_fields:
        return "No form fields found"

    lines = []
    for el in form_fields[:15]:
        selector = el.get('selector', '')
        value = el.get('value', '')
        required = el.get('required', False)
        el_type = el.get('type', 'text')

        status = "FILLED" if value else ("REQUIRED" if required else "EMPTY")
        lines.append(f"- {el_type}: {selector} [{status}]")

    return "\n".join(lines)


def extract_validation_messages(state: State) -> str:
    """Extract validation messages from page."""

    # Look for common validation indicators in visible text
    text_lower = state.visible_text.lower()

    messages = []
    validation_keywords = ['required', 'invalid', 'must', 'should', 'error', 'wrong']

    lines = state.visible_text.split('\n')
    for line in lines[:20]:  # Check first 20 lines
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in validation_keywords):
            messages.append(line.strip())

    return "\n".join(messages[:5]) if messages else "No validation messages found"


def parse_structured_response(response: str) -> str:
    """
    Parse structured response to extract Playwright command.

    Handles both JSON and plain text responses.
    """

    # Try JSON parsing first
    try:
        # Clean response
        response = response.strip()
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]

        data = json.loads(response)

        # Extract command from JSON
        if 'command' in data:
            return data['command']

        # Build command from components
        action = data.get('action', '')
        target = data.get('target', '')
        value = data.get('value', '')

        if action == 'fill':
            return f"page.fill('{target}', '{value}')"
        elif action == 'click':
            return f"page.click('{target}')"
        elif action == 'type':
            return f"page.type('{target}', '{value}')"
        elif action == 'select':
            return f"page.select_option('{target}', '{value}')"
        elif action == 'wait':
            return f"page.wait_for_selector('{target}')"
        elif action == 'check':
            return f"page.check('{target}')"
        elif action == 'uncheck':
            return f"page.uncheck('{target}')"
        elif action == 'pass':
            return "pass"
        else:
            return extract_code_from_text(response)

    except (json.JSONDecodeError, KeyError):
        # Fall back to text extraction
        return extract_code_from_text(response)


def extract_code_from_text(response: str) -> str:
    """Extract Python code from text response."""

    # Remove code blocks if present
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]

    # Get first line that looks like code
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('page.') or line == 'pass':
            return line

    # Default if nothing found
    return response.strip().split('\n')[0]


# Few-shot examples for in-context learning

PLAYWRIGHT_EXAMPLES = """
COMMON PATTERNS:
1. Fill form field: page.fill('#username', 'john_doe')
2. Click button: page.click('button[type="submit"]')
3. Select dropdown: page.select_option('#country', 'USA')
4. Check checkbox: page.check('#agree-terms')
5. Type slowly: page.type('#search', 'search query')
6. Wait for element: page.wait_for_selector('.success-message')
7. Press key: page.press('#input', 'Enter')
8. Navigate: page.goto('https://example.com')
9. Hover: page.hover('.dropdown-trigger')
10. Complete: pass

SELECTOR PATTERNS:
- ID: #element-id
- Class: .class-name
- Name: [name="field-name"]
- Type: button[type="submit"]
- Text: button:has-text("Submit")
- Placeholder: [placeholder="Enter email"]
"""


# Main interface functions (backward compatible)

def format_dom_elements(elements: List[Dict[str, Any]], max_elements: int = 50) -> str:
    """Legacy function - redirects to compact version."""
    return format_dom_elements_compact(elements, max_elements)


def format_failed_actions(failed_actions: List[FailedAction], max_failures: int = 5) -> str:
    """Legacy function - maintained for compatibility."""
    return format_failed_actions_brief(failed_actions[:max_failures])


def format_successful_actions(actions: List[str], max_actions: int = 10) -> str:
    """Legacy function - maintained for compatibility."""
    if not actions:
        return "No actions taken yet"
    recent = actions[-max_actions:]
    return "\n".join(f"✓ {action}" for action in recent)


def build_action_prompt(state: State) -> str:
    """
    Main prompt builder - uses optimized version.

    Maintains backward compatibility while using new optimized approach.
    """
    # Use optimized prompt for better performance
    return build_action_prompt_optimized(state)


def build_initial_analysis_prompt(state: State) -> str:
    """
    Optimized initial analysis prompt.

    More concise for smaller models.
    """

    elements = format_dom_elements_compact(state.dom_elements, max_elements=20)

    prompt = f"""Analyze page for: {state.main_goal}

URL: {state.url}

ELEMENTS:
{elements}

TEXT (first 300 chars):
{state.visible_text[:300]}

Identify:
1. Page type (form/list/etc)
2. Required actions sequence
3. Potential blockers

Brief analysis (2-3 sentences):"""

    return prompt