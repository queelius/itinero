"""
Enhanced state management for LLM web automation.

Key improvements:
- Better failure pattern detection
- Enhanced DOM element extraction with more context
- Improved strategy suggestions based on failure patterns
- Smarter visible text extraction
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter


@dataclass
class FailedAction:
    """Record of a failed action with error details."""
    action: str
    error: str
    error_type: str
    timestamp: float
    traceback: Optional[str] = None
    suggestion: Optional[str] = None
    selector: Optional[str] = None  # Track which selector failed


@dataclass
class State:
    """Complete state of the web automation environment."""

    # Current page state
    url: str
    html: str
    dom_elements: List[Dict[str, Any]]
    visible_text: str

    # Action history
    successful_actions: List[str] = field(default_factory=list)
    failed_actions: List[FailedAction] = field(default_factory=list)

    # Goal tracking
    main_goal: str = ""
    completed_subgoals: List[str] = field(default_factory=list)
    current_subgoal: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    step_count: int = 0
    needs_strategy_change: bool = False

    # Additional context
    form_errors: List[str] = field(default_factory=list)  # Track form validation errors
    page_type: Optional[str] = None  # Detected page type (form, list, etc.)


def extract_dom_elements(page) -> List[Dict[str, Any]]:
    """
    Enhanced DOM element extraction with better context and attributes.
    """

    return page.evaluate("""
        () => {
            // Helper to get best selector for element
            function getBestSelector(el) {
                if (el.id) return `#${el.id}`;
                if (el.name) return `[name="${el.name}"]`;

                // Try data attributes
                for (let attr of el.attributes) {
                    if (attr.name.startsWith('data-') && attr.value) {
                        return `[${attr.name}="${attr.value}"]`;
                    }
                }

                // Use class if unique enough
                if (el.className) {
                    const classes = el.className.split(' ').filter(c => c.length > 0);
                    if (classes.length > 0) {
                        return `.${classes[0]}`;
                    }
                }

                // Fallback to tag with index
                const siblings = Array.from(el.parentNode?.children || []);
                const index = siblings.indexOf(el);
                return `${el.tagName.toLowerCase()}:nth-child(${index + 1})`;
            }

            // Helper to check if element is actually visible
            function isVisible(el) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                return rect.width > 0 &&
                       rect.height > 0 &&
                       style.display !== 'none' &&
                       style.visibility !== 'hidden' &&
                       style.opacity !== '0' &&
                       el.offsetParent !== null;
            }

            // Helper to get element's label
            function getLabel(el) {
                // Check for associated label
                if (el.id) {
                    const label = document.querySelector(`label[for="${el.id}"]`);
                    if (label) return label.textContent.trim();
                }

                // Check for parent label
                const parentLabel = el.closest('label');
                if (parentLabel) {
                    return parentLabel.textContent.trim().replace(el.textContent || '', '').trim();
                }

                // Check for aria-label
                if (el.getAttribute('aria-label')) {
                    return el.getAttribute('aria-label');
                }

                // Check for placeholder as last resort
                return el.placeholder || '';
            }

            // Extract all interactive elements
            const elements = Array.from(document.querySelectorAll(
                'input, button, select, textarea, a, [role="button"], [onclick], [role="link"]'
            )).slice(0, 150).map(el => {
                const rect = el.getBoundingClientRect();
                const label = getLabel(el);

                // Check for validation state
                let validationState = null;
                if (el.validity) {
                    if (!el.validity.valid) {
                        validationState = el.validationMessage || 'invalid';
                    }
                }

                // Get parent form if exists
                const form = el.closest('form');
                const formId = form?.id || form?.name || null;

                return {
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    id: el.id || null,
                    name: el.name || null,
                    className: el.className || null,
                    value: el.value || null,
                    text: el.textContent?.trim().substring(0, 100) || null,
                    placeholder: el.placeholder || null,
                    disabled: el.disabled || false,
                    readonly: el.readOnly || false,
                    required: el.required || false,
                    checked: el.checked || false,
                    visible: isVisible(el),
                    href: el.href || null,
                    selector: getBestSelector(el),
                    label: label,
                    validationState: validationState,
                    formId: formId,
                    ariaLabel: el.getAttribute('aria-label') || null,
                    ariaDescribedby: el.getAttribute('aria-describedby') || null,
                    dataTestId: el.getAttribute('data-testid') || el.getAttribute('data-test-id') || null,
                    position: {
                        top: rect.top,
                        left: rect.left,
                        width: rect.width,
                        height: rect.height
                    },
                    // Additional attributes for better selection
                    pattern: el.pattern || null,
                    maxLength: el.maxLength || null,
                    minLength: el.minLength || null,
                    min: el.min || null,
                    max: el.max || null,
                    step: el.step || null,
                    autocomplete: el.autocomplete || null
                };
            });

            // Also get page title and any error messages
            const errorMessages = Array.from(document.querySelectorAll(
                '.error, .alert, .validation-error, [role="alert"], .invalid-feedback'
            )).map(el => el.textContent.trim()).filter(text => text.length > 0);

            return {
                elements: elements,
                title: document.title,
                errorMessages: errorMessages,
                formCount: document.querySelectorAll('form').length,
                hasPasswordField: elements.some(el => el.type === 'password'),
                hasSubmitButton: elements.some(el =>
                    el.type === 'submit' ||
                    (el.tag === 'button' && (el.text || '').toLowerCase().includes('submit'))
                )
            };
        }
    """)


def get_visible_text(page, max_length: int = 3000) -> str:
    """
    Enhanced visible text extraction with better filtering.
    """

    text = page.evaluate("""
        () => {
            // Helper to check if element is visible
            function isVisible(el) {
                const style = window.getComputedStyle(el);
                return style.display !== 'none' &&
                       style.visibility !== 'hidden' &&
                       style.opacity !== '0';
            }

            // Get text but exclude script, style, and hidden content
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                {
                    acceptNode: function(node) {
                        const parent = node.parentElement;

                        // Skip script, style, noscript tags
                        if (parent.tagName === 'SCRIPT' ||
                            parent.tagName === 'STYLE' ||
                            parent.tagName === 'NOSCRIPT') {
                            return NodeFilter.FILTER_REJECT;
                        }

                        // Skip hidden elements
                        if (!isVisible(parent)) {
                            return NodeFilter.FILTER_REJECT;
                        }

                        // Accept non-empty text
                        if (node.nodeValue.trim().length > 0) {
                            return NodeFilter.FILTER_ACCEPT;
                        }

                        return NodeFilter.FILTER_REJECT;
                    }
                }
            );

            let text = '';
            let node;
            const seen = new Set();

            while (node = walker.nextNode()) {
                const value = node.nodeValue.trim();
                // Avoid duplicates
                if (!seen.has(value)) {
                    text += value + ' ';
                    seen.add(value);
                }
            }

            // Also specifically look for error messages
            const errors = Array.from(document.querySelectorAll(
                '.error-message, .error, .alert-danger, [role="alert"]'
            )).map(el => el.textContent.trim()).filter(t => t.length > 0);

            if (errors.length > 0) {
                text = errors.join(' | ') + ' | ' + text;
            }

            return text.trim();
        }
    """)

    return text[:max_length] if len(text) > max_length else text


def create_state(page, main_goal: str = "", step_count: int = 0) -> State:
    """
    Create an enhanced state snapshot from the current page.
    """

    # Extract DOM data
    dom_data = extract_dom_elements(page)

    # Separate elements and metadata
    if isinstance(dom_data, dict):
        elements = dom_data.get('elements', [])
        error_messages = dom_data.get('errorMessages', [])
        page_type = detect_page_type(dom_data)
    else:
        # Fallback for backward compatibility
        elements = dom_data
        error_messages = []
        page_type = None

    state = State(
        url=page.url,
        html=page.content()[:10000],  # Truncate for prompt limits
        dom_elements=elements,
        visible_text=get_visible_text(page),
        main_goal=main_goal,
        step_count=step_count,
        timestamp=time.time(),
        form_errors=error_messages,
        page_type=page_type
    )

    return state


def update_state(page, previous_state: State) -> State:
    """
    Update state with new page information while preserving history.
    """

    new_state = create_state(page, previous_state.main_goal, previous_state.step_count + 1)

    # Preserve history
    new_state.successful_actions = previous_state.successful_actions.copy()
    new_state.failed_actions = previous_state.failed_actions.copy()
    new_state.completed_subgoals = previous_state.completed_subgoals.copy()
    new_state.current_subgoal = previous_state.current_subgoal
    new_state.needs_strategy_change = previous_state.needs_strategy_change

    return new_state


def detect_page_type(dom_data: Dict[str, Any]) -> str:
    """
    Detect the type of page based on DOM structure.
    """

    if dom_data.get('formCount', 0) > 0:
        if dom_data.get('hasPasswordField'):
            return 'login_form'
        elif dom_data.get('hasSubmitButton'):
            return 'input_form'
        else:
            return 'partial_form'
    elif len([el for el in dom_data.get('elements', []) if el.get('tag') == 'a']) > 10:
        return 'navigation_page'
    else:
        return 'content_page'


def analyze_failures(state: State) -> str:
    """
    Enhanced failure analysis with better pattern detection and suggestions.
    """

    if not state.failed_actions:
        return ""

    recent = state.failed_actions[-5:]

    # Analyze failure patterns
    failure_patterns = detect_failure_patterns(recent)

    # Generate strategic hints based on patterns
    if failure_patterns.get('repeated_action'):
        action = failure_patterns['repeated_action']
        count = failure_patterns['repeat_count']
        return f"Action '{action}' failed {count} times. Must try different selector or approach."

    if failure_patterns.get('selector_issues'):
        return "Multiple selector failures. Try: 1) Wait for page load 2) Use different selector strategy 3) Check if elements are dynamically loaded"

    if failure_patterns.get('validation_errors'):
        return "Form validation failing. Check: 1) Required field formats 2) Field dependencies 3) Error messages on page"

    if failure_patterns.get('disabled_elements'):
        return "Elements are disabled. Need to: 1) Fill required fields first 2) Check prerequisites 3) Enable parent controls"

    if failure_patterns.get('timing_issues'):
        return "Timing issues detected. Try: 1) Add wait_for_selector 2) Increase delays 3) Check for async loading"

    if failure_patterns.get('permission_issues'):
        return "Permission/state issues. Check: 1) Login status 2) Page navigation flow 3) Required preconditions"

    # Fallback to simple pattern matching
    return get_simple_failure_hint(recent)


def detect_failure_patterns(failures: List[FailedAction]) -> Dict[str, Any]:
    """
    Detect complex failure patterns from recent failures.
    """

    patterns = {}

    if not failures:
        return patterns

    # Check for repeated action
    action_counts = Counter(f.action for f in failures)
    most_common = action_counts.most_common(1)[0] if action_counts else (None, 0)
    if most_common[1] >= 2:
        patterns['repeated_action'] = most_common[0]
        patterns['repeat_count'] = most_common[1]

    # Check error types
    error_types = [f.error_type for f in failures]
    errors = [f.error.lower() for f in failures]

    # Selector issues
    if sum(1 for e in errors if 'not found' in e or 'timeout' in e or 'no element' in e) >= 2:
        patterns['selector_issues'] = True

    # Validation errors
    if sum(1 for e in errors if 'validation' in e or 'invalid' in e or 'required' in e) >= 2:
        patterns['validation_errors'] = True

    # Disabled elements
    if sum(1 for e in errors if 'disabled' in e or 'not enabled' in e) >= 2:
        patterns['disabled_elements'] = True

    # Timing issues
    if sum(1 for t in error_types if 'Timeout' in t) >= 2:
        patterns['timing_issues'] = True

    # Permission issues
    if sum(1 for e in errors if 'permission' in e or 'unauthorized' in e or 'forbidden' in e) >= 1:
        patterns['permission_issues'] = True

    return patterns


def get_simple_failure_hint(failures: List[FailedAction]) -> str:
    """
    Simple failure hint generation as fallback.
    """

    if not failures:
        return ""

    last_error = failures[-1].error.lower()

    if "not found" in last_error:
        return "Element not found. Try different selector or wait for element."
    elif "disabled" in last_error:
        return "Element disabled. Fill required fields or check prerequisites."
    elif "validation" in last_error or "invalid" in last_error:
        return "Validation error. Check field format and requirements."
    elif "timeout" in last_error:
        return "Timeout error. Page may be slow or element doesn't exist."
    elif "not visible" in last_error:
        return "Element not visible. May need scrolling or expanding parent."
    else:
        return "Consider trying a different approach or selector."


def extract_selector_from_action(action: str) -> Optional[str]:
    """
    Extract the selector from a Playwright action string.
    """

    import re

    # Match patterns like page.click('selector') or page.fill('selector', ...)
    patterns = [
        r"page\.\w+\(['\"]([^'\"]+)['\"]",  # Single or double quoted
        r"page\.\w+\(([^,\)]+)",  # Unquoted (less common)
    ]

    for pattern in patterns:
        match = re.search(pattern, action)
        if match:
            return match.group(1).strip().strip("'\"")

    return None


def get_failure_suggestions(state: State) -> List[str]:
    """
    Generate a list of suggestions based on current failures.
    """

    suggestions = []

    if not state.failed_actions:
        return suggestions

    patterns = detect_failure_patterns(state.failed_actions[-5:])

    if patterns.get('selector_issues'):
        suggestions.extend([
            "Use wait_for_selector before interacting",
            "Try selecting by different attribute (id, class, text)",
            "Check if element is in an iframe",
            "Verify element is not dynamically loaded"
        ])

    if patterns.get('validation_errors'):
        suggestions.extend([
            "Read error messages on the page",
            "Check field format requirements",
            "Fill all required fields before submitting",
            "Verify field values match expected patterns"
        ])

    if patterns.get('disabled_elements'):
        suggestions.extend([
            "Fill prerequisite fields first",
            "Check if parent element needs to be activated",
            "Look for enable/unlock controls",
            "Verify form is in edit mode"
        ])

    if patterns.get('timing_issues'):
        suggestions.extend([
            "Add explicit waits for elements",
            "Check for loading indicators",
            "Wait for network requests to complete",
            "Use wait_for_timeout if page uses animations"
        ])

    return suggestions[:3]  # Return top 3 most relevant suggestions