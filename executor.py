"""
Enhanced action execution with improved error handling and recovery.

Key improvements:
- Better error categorization and suggestions
- Automatic retry with different strategies
- Smart selector validation
- Enhanced Playwright method support
"""

import re
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Any, Tuple, List


@dataclass
class ExecutionResult:
    """Result of executing an action."""
    success: bool
    action: str
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None
    suggestion: Optional[str] = None
    execution_time: float = 0.0
    retry_suggestion: Optional[str] = None  # Specific retry strategy


def parse_action_components(action: str) -> Tuple[str, str, List[str]]:
    """
    Parse a Playwright action into method, selector, and arguments.

    Returns:
        Tuple of (method, selector, other_args)
    """

    # Match page.method(args...)
    match = re.match(r"page\.(\w+)\((.*)\)", action.strip())
    if not match:
        return "", "", []

    method = match.group(1)
    args_str = match.group(2)

    if not args_str:
        return method, "", []

    # Parse arguments - handle quoted strings properly
    args = []
    current_arg = ""
    in_quotes = False
    quote_char = None

    for char in args_str:
        if char in ["'", '"'] and not in_quotes:
            in_quotes = True
            quote_char = char
            current_arg += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            current_arg += char
            quote_char = None
        elif char == "," and not in_quotes:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char

    if current_arg:
        args.append(current_arg.strip())

    # Clean arguments
    cleaned_args = []
    for arg in args:
        arg = arg.strip()
        # Remove quotes if present
        if (arg.startswith("'") and arg.endswith("'")) or (arg.startswith('"') and arg.endswith('"')):
            arg = arg[1:-1]
        cleaned_args.append(arg)

    selector = cleaned_args[0] if cleaned_args else ""
    other_args = cleaned_args[1:] if len(cleaned_args) > 1 else []

    return method, selector, other_args


def validate_selector(page, selector: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a selector exists and is visible on the page.

    Returns:
        Tuple of (exists, error_message)
    """

    try:
        # Check if element exists
        count = page.locator(selector).count()
        if count == 0:
            return False, f"No elements found matching selector: {selector}"
        elif count > 1:
            # Multiple elements - might need more specific selector
            return True, f"Warning: {count} elements match selector {selector}"

        # Check if visible
        is_visible = page.locator(selector).first.is_visible()
        if not is_visible:
            return False, f"Element exists but is not visible: {selector}"

        return True, None

    except Exception as e:
        return False, f"Selector validation failed: {str(e)}"


def execute_with_retry(page, action: str, max_retries: int = 2) -> ExecutionResult:
    """
    Execute action with automatic retry using different strategies.
    """

    result = execute_action_internal(page, action)

    if result.success or max_retries == 0:
        return result

    # Try retry strategies based on error type
    retry_strategies = get_retry_strategies(result)

    for i, strategy in enumerate(retry_strategies[:max_retries]):
        if i > 0:
            time.sleep(0.5)  # Brief pause between retries

        retry_action = apply_retry_strategy(action, strategy, page)
        if retry_action != action:
            retry_result = execute_action_internal(page, retry_action)
            if retry_result.success:
                return retry_result

    return result  # Return original failure if all retries fail


def get_retry_strategies(result: ExecutionResult) -> List[str]:
    """
    Get retry strategies based on error type.
    """

    strategies = []

    if not result.error:
        return strategies

    error_lower = result.error.lower()

    if "timeout" in error_lower or "not found" in error_lower:
        strategies.extend(["wait_first", "different_selector", "javascript"])
    elif "disabled" in error_lower:
        strategies.extend(["wait_enable", "force_click"])
    elif "not visible" in error_lower:
        strategies.extend(["scroll_to", "wait_visible", "force_click"])
    elif "intercepted" in error_lower:
        strategies.extend(["wait_stable", "force_click"])

    return strategies


def apply_retry_strategy(action: str, strategy: str, page) -> str:
    """
    Apply a retry strategy to modify the action.
    """

    method, selector, args = parse_action_components(action)

    if strategy == "wait_first":
        # Add wait before action
        page.wait_for_selector(selector, timeout=3000)
        return action

    elif strategy == "different_selector":
        # Try to find alternative selector
        alt_selector = find_alternative_selector(page, selector)
        if alt_selector and alt_selector != selector:
            return action.replace(selector, alt_selector)

    elif strategy == "javascript":
        # Use JavaScript as fallback
        if method == "click":
            page.evaluate(f"document.querySelector('{selector}')?.click()")
            return "pass"  # Action completed via JS
        elif method == "fill" and args:
            value = args[0]
            page.evaluate(f"document.querySelector('{selector}').value = '{value}'")
            return "pass"

    elif strategy == "scroll_to":
        # Scroll element into view
        page.evaluate(f"document.querySelector('{selector}')?.scrollIntoView({{behavior: 'smooth', block: 'center'}})")
        time.sleep(0.5)
        return action

    elif strategy == "wait_visible":
        # Wait for visibility
        page.wait_for_selector(selector, state="visible", timeout=3000)
        return action

    elif strategy == "wait_enable":
        # Wait for element to be enabled
        page.wait_for_function(f"document.querySelector('{selector}')?.disabled === false", timeout=3000)
        return action

    elif strategy == "force_click":
        # Force click even if intercepted
        if method == "click":
            return f"page.locator('{selector}').click(force=True)"

    elif strategy == "wait_stable":
        # Wait for page to stabilize
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(500)
        return action

    return action


def find_alternative_selector(page, original_selector: str) -> Optional[str]:
    """
    Try to find an alternative selector for the same element.
    """

    try:
        # Get element properties to find alternatives
        element_data = page.evaluate(f"""
            (selector) => {{
                const el = document.querySelector(selector);
                if (!el) return null;
                return {{
                    id: el.id,
                    className: el.className,
                    name: el.name,
                    type: el.type,
                    text: el.textContent?.trim().substring(0, 50),
                    tagName: el.tagName.toLowerCase()
                }};
            }}
        """, original_selector)

        if not element_data:
            return None

        # Try alternative selectors
        alternatives = []

        if element_data.get('id'):
            alternatives.append(f"#{element_data['id']}")

        if element_data.get('name'):
            alternatives.append(f"[name='{element_data['name']}']")

        if element_data.get('className'):
            classes = element_data['className'].split()[:1]  # Use first class
            if classes:
                alternatives.append(f".{classes[0]}")

        # Try each alternative
        for alt in alternatives:
            if alt != original_selector:
                count = page.locator(alt).count()
                if count == 1:  # Unique match
                    return alt

    except:
        pass

    return None


def execute_action_internal(page, action: str) -> ExecutionResult:
    """
    Core action execution with enhanced error handling.
    """

    start_time = time.time()

    try:
        # Parse action components
        method, selector, args = parse_action_components(action)

        # Validate selector before execution (for methods that use selectors)
        if selector and method in ['click', 'fill', 'type', 'check', 'uncheck', 'hover', 'focus']:
            valid, error_msg = validate_selector(page, selector)
            if not valid:
                return ExecutionResult(
                    success=False,
                    action=action,
                    error=error_msg,
                    error_type="SelectorError",
                    suggestion="Check if selector is correct or element is loaded",
                    execution_time=time.time() - start_time
                )

        # Special handling for file:// protocol and about:blank issues with page.fill()
        if method == "fill" and (page.url.startswith("file://") or page.url == "about:blank"):
            try:
                # Use JavaScript as fallback
                value = args[0] if args else ""
                page.evaluate(f"""
                    const el = document.querySelector('{selector}');
                    if (el) {{
                        el.value = '{value}';
                        el.dispatchEvent(new Event('input', {{bubbles: true}}));
                        el.dispatchEvent(new Event('change', {{bubbles: true}}));
                    }}
                """)

                return ExecutionResult(
                    success=True,
                    action=action,
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                # Fall through to normal execution if JS fails
                pass

        # Create safe namespace for execution
        namespace = {
            'page': page,
            'time': time,
        }

        # Handle special methods that need parameter conversion
        if method == "wait_for_timeout":
            # Convert string to int for timeout
            if args:
                timeout = int(args[0]) if args[0].isdigit() else 1000
                page.wait_for_timeout(timeout)
            else:
                page.wait_for_timeout(1000)

            return ExecutionResult(
                success=True,
                action=action,
                execution_time=time.time() - start_time
            )

        # Execute the action
        exec(action.strip(), namespace)

        return ExecutionResult(
            success=True,
            action=action,
            execution_time=time.time() - start_time
        )

    except TimeoutError as e:
        return ExecutionResult(
            success=False,
            action=action,
            error=f"Timeout waiting for element: {str(e)}",
            error_type="TimeoutError",
            error_traceback=traceback.format_exc(),
            suggestion="Element may not exist or selector may be wrong. Try wait_for_selector first or use different selector.",
            retry_suggestion="wait_first",
            execution_time=time.time() - start_time
        )

    except AttributeError as e:
        return ExecutionResult(
            success=False,
            action=action,
            error=f"Method not found: {str(e)}",
            error_type="AttributeError",
            error_traceback=traceback.format_exc(),
            suggestion="Check Playwright method name and syntax. Common methods: click, fill, type, select_option",
            execution_time=time.time() - start_time
        )

    except ValueError as e:
        error_str = str(e).lower()
        if "disabled" in error_str:
            return ExecutionResult(
                success=False,
                action=action,
                error=f"Element is disabled: {str(e)}",
                error_type="DisabledError",
                error_traceback=traceback.format_exc(),
                suggestion="Element is disabled. Fill required fields or meet preconditions first.",
                retry_suggestion="wait_enable",
                execution_time=time.time() - start_time
            )
        else:
            return ExecutionResult(
                success=False,
                action=action,
                error=f"Invalid value or parameter: {str(e)}",
                error_type="ValueError",
                error_traceback=traceback.format_exc(),
                suggestion="Check value format and method parameters.",
                execution_time=time.time() - start_time
            )

    except Exception as e:
        # Enhanced error analysis
        error_msg = str(e)
        error_lower = error_msg.lower()
        suggestion = "Check command syntax and element state."
        retry_suggestion = None

        # Provide specific suggestions based on error patterns
        if "not found" in error_lower or "no node" in error_lower:
            suggestion = "Element not found. Try: 1) Different selector 2) wait_for_selector 3) Check if dynamically loaded"
            retry_suggestion = "different_selector"
        elif "not visible" in error_lower or "hidden" in error_lower:
            suggestion = "Element not visible. Try: 1) Scroll to element 2) Check if hidden by CSS 3) Wait for visibility"
            retry_suggestion = "scroll_to"
        elif "not a <select>" in error_lower:
            suggestion = "Element is not a dropdown. For custom dropdowns, use click() to open then click() option."
        elif "validation" in error_lower or "constraint" in error_lower:
            suggestion = "Form validation failed. Check: 1) Required formats 2) Field constraints 3) Error messages on page"
        elif "intercepted" in error_lower or "obscured" in error_lower:
            suggestion = "Click intercepted by another element. Try: 1) Wait for overlays to disappear 2) Scroll to element 3) Use force=True"
            retry_suggestion = "force_click"
        elif "permission" in error_lower or "denied" in error_lower:
            suggestion = "Permission denied. Check: 1) Login status 2) User permissions 3) Security restrictions"
        elif "stale" in error_lower or "detached" in error_lower:
            suggestion = "Element no longer in DOM. Page may have changed. Re-select element or wait for stable state."
            retry_suggestion = "wait_stable"

        return ExecutionResult(
            success=False,
            action=action,
            error=error_msg,
            error_type=type(e).__name__,
            error_traceback=traceback.format_exc(),
            suggestion=suggestion,
            retry_suggestion=retry_suggestion,
            execution_time=time.time() - start_time
        )


def execute_action(page, action: str) -> ExecutionResult:
    """
    Main entry point for action execution with retry logic.

    Args:
        page: Playwright page object
        action: Python code string to execute (e.g., "page.click('#submit')")

    Returns:
        ExecutionResult with success status and error details if failed
    """

    # Skip retry for certain actions
    no_retry_methods = ['wait_for_timeout', 'goto', 'reload', 'go_back', 'go_forward']
    method = parse_action_components(action)[0]

    if method in no_retry_methods:
        return execute_action_internal(page, action)

    # Execute with retry for other actions
    return execute_with_retry(page, action, max_retries=1)


def safe_execute_sequence(page, actions: List[str]) -> List[ExecutionResult]:
    """
    Execute a sequence of actions, with smart continuation on failures.

    Args:
        page: Playwright page object
        actions: List of action strings to execute

    Returns:
        List of ExecutionResults for each attempted action
    """

    results = []
    consecutive_failures = 0

    for i, action in enumerate(actions):
        result = execute_action(page, action)
        results.append(result)

        if not result.success:
            consecutive_failures += 1

            # Stop if too many consecutive failures
            if consecutive_failures >= 3:
                break

            # Try to recover based on error type
            if result.retry_suggestion == "wait_stable":
                page.wait_for_timeout(1000)
        else:
            consecutive_failures = 0

        # Small delay between actions to be more human-like
        if i < len(actions) - 1:  # Don't delay after last action
            time.sleep(0.3)

    return results


def parse_fill_command(action: str) -> Tuple[str, str]:
    """
    Legacy function for backward compatibility.
    Parse a page.fill() command to extract selector and value.
    """

    method, selector, args = parse_action_components(action)
    if method == "fill" and args:
        return selector, args[0]

    raise ValueError(f"Could not parse fill command: {action}")