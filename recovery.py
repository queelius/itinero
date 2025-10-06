"""
Recovery strategies for handling stuck or error states.

This module provides pluggable recovery functions that can be composed
with agents to handle various failure scenarios.
"""

from typing import Optional, List
from core import Action, Context


def default_recovery(context: Context) -> Optional[Action]:
    """
    Default recovery strategy when agent is stuck.

    Args:
        context: Current automation context

    Returns:
        Recovery action or None to give up
    """
    if not context.history:
        return None

    last_result = context.history[-1]

    # Analyze the error
    error = (last_result.error or '').lower()

    if 'not found' in error or 'timeout' in error:
        # Element not found - wait and retry
        return Action(type='wait', selector='body', value='')

    elif 'disabled' in error:
        # Element disabled - look for required fields to fill
        return _find_required_field(context)

    elif 'not visible' in error:
        # Element not visible - try scrolling
        selector = last_result.action.selector
        return Action(
            type='scroll',
            selector=selector,
            options={'command': f"page.evaluate('document.querySelector(\"{selector}\")?.scrollIntoView()')"}
        )

    # Default: wait briefly
    return Action(
        type='wait',
        selector='body',
        options={'command': 'page.wait_for_timeout(1000)'}
    )


def _find_required_field(context: Context) -> Optional[Action]:
    """Find the first required empty field to fill."""
    elements = context.page_state.get('elements', [])

    for el in elements:
        if (el.get('required') and
            el.get('tag') == 'input' and
            not el.get('value')):

            # Try to infer value from goal or field name
            selector = el.get('selector', '')
            value = _infer_value(selector, context.goal)

            if value:
                return Action(type='fill', selector=selector, value=value)

    return None


def _infer_value(selector: str, goal: str) -> str:
    """Infer appropriate value for a field from its selector and goal."""
    selector_lower = selector.lower()
    goal_lower = goal.lower()

    # Common field patterns
    if 'firstname' in selector_lower:
        return 'John'
    elif 'lastname' in selector_lower:
        return 'Doe'
    elif 'email' in selector_lower:
        return 'user@example.com'
    elif 'phone' in selector_lower:
        return '555-0123'
    elif 'address' in selector_lower:
        return '123 Main St'
    elif 'city' in selector_lower:
        return 'Boston'
    elif 'zip' in selector_lower or 'postal' in selector_lower:
        return '02134'

    return ''


def aggressive_recovery(context: Context) -> Optional[Action]:
    """
    More aggressive recovery that tries multiple strategies.

    Use this when default recovery isn't working.
    """
    # Try default first
    action = default_recovery(context)
    if action:
        return action

    # If still stuck, try refreshing the page
    if len(context.history) > 10:
        last_10 = context.history[-10:]
        failures = sum(1 for r in last_10 if not r.success)

        if failures > 7:  # More than 70% failure rate
            return Action(
                type='reload',
                selector='',
                options={'command': 'page.reload()'}
            )

    return None


def conservative_recovery(context: Context) -> Optional[Action]:
    """
    Conservative recovery that only waits, never takes risky actions.

    Use this for production environments where safety is paramount.
    """
    # Only return wait actions
    return Action(
        type='wait',
        selector='body',
        options={'command': 'page.wait_for_timeout(2000)'}
    )
