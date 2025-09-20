"""
Enhanced LLM-driven web automation agent with improved prompting.

Key improvements:
- Multiple prompting strategies (optimized, structured, recovery)
- Better error pattern detection and recovery
- Adaptive strategy selection based on context
- Support for JSON structured output parsing
"""

import time
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass

from state import State, FailedAction, create_state, update_state
from executor import execute_action, ExecutionResult
from prompts import (
    build_action_prompt_optimized,
    build_structured_action_prompt,
    build_recovery_prompt,
    build_validation_prompt,
    build_initial_analysis_prompt,
    parse_structured_response,
    detect_error_pattern,
    extract_code_from_text
)


@dataclass
class AgentConfig:
    """Configuration for the LLM agent."""
    max_steps: int = 30
    max_repeated_failures: int = 3
    action_delay: float = 0.5
    verbose: bool = True
    analyze_on_start: bool = False
    use_structured_output: bool = False  # Use JSON structured output
    adaptive_prompting: bool = True  # Adapt prompt style based on situation


class SimpleLLMAgent:
    """
    Enhanced LLM-based web automation agent.

    Features adaptive prompting, structured output, and intelligent error recovery.
    """

    def __init__(self, llm_api: Callable[[str], str], config: Optional[AgentConfig] = None):
        """
        Initialize the agent.

        Args:
            llm_api: Function that takes a prompt and returns LLM response
            config: Agent configuration
        """
        self.llm_api = llm_api
        self.config = config or AgentConfig()
        self.repeated_failures: Dict[str, int] = {}
        self.consecutive_failures = 0
        self.last_error_pattern = "none"

    def predict_next_action(self, state: State) -> str:
        """
        Predict the next Playwright action based on current state.

        Uses adaptive prompting strategy based on context and error patterns.

        Args:
            state: Current environment state

        Returns:
            Playwright command string (e.g., "page.click('#submit')")
        """

        # Select prompting strategy based on situation
        prompt = self._select_prompt_strategy(state)

        # Get LLM response
        response = self.llm_api(prompt)

        # Parse response based on expected format
        if self.config.use_structured_output or "json" in prompt.lower():
            action = parse_structured_response(response)
        else:
            action = self._extract_code(response)

        # Validate and clean action
        action = self._validate_action(action, state)

        if self.config.verbose:
            print(f"Step {state.step_count}: {action}")

        return action

    def _select_prompt_strategy(self, state: State) -> str:
        """
        Select the best prompting strategy based on current context.

        Args:
            state: Current environment state

        Returns:
            Selected prompt string
        """

        if not self.config.adaptive_prompting:
            # Use default optimized prompt
            return build_action_prompt_optimized(state)

        # Detect patterns and select strategy
        error_pattern = detect_error_pattern(state.failed_actions)

        # Check for repeated failures on same action
        if state.failed_actions and self._is_stuck_on_action(state):
            # Use recovery prompt for stuck situations
            repeated_action = state.failed_actions[-1].action
            if self.config.verbose:
                print(f"  [Using recovery strategy for stuck action]")
            return build_recovery_prompt(state, repeated_action)

        # Check for validation errors
        if error_pattern == "validation_error":
            if self.config.verbose:
                print(f"  [Using validation-focused strategy]")
            return build_validation_prompt(state)

        # Use structured output for complex forms or if configured
        if self.config.use_structured_output or self._is_complex_form(state):
            if self.config.verbose:
                print(f"  [Using structured JSON output]")
            return build_structured_action_prompt(state)

        # Default to optimized prompt
        return build_action_prompt_optimized(state)

    def _is_stuck_on_action(self, state: State) -> bool:
        """Check if agent is stuck repeating the same failed action."""

        if len(state.failed_actions) < 2:
            return False

        last_action = state.failed_actions[-1].action
        recent_failures = state.failed_actions[-3:]

        return sum(1 for f in recent_failures if f.action == last_action) >= 2

    def _is_complex_form(self, state: State) -> bool:
        """Detect if current page is a complex form."""

        # Count form elements
        form_elements = [el for el in state.dom_elements
                        if el.get('tag') in ['input', 'select', 'textarea']]

        # Complex if many form fields or has validation errors
        return (len(form_elements) > 5 or
                any("validation" in f.error.lower() for f in state.failed_actions[-3:]))

    def _validate_action(self, action: str, state: State) -> str:
        """
        Validate and potentially correct the predicted action.

        Args:
            action: Predicted action string
            state: Current state

        Returns:
            Validated/corrected action string
        """

        # Clean whitespace
        action = action.strip()

        # Check for common syntax errors
        if action.startswith('page.') or action == 'pass':
            # Validate quotes are balanced
            single_quotes = action.count("'")
            double_quotes = action.count('"')

            if single_quotes % 2 != 0 or double_quotes % 2 != 0:
                # Try to fix unbalanced quotes
                if single_quotes % 2 != 0:
                    action += "'"
                if double_quotes % 2 != 0:
                    action += '"'

            # Ensure parentheses are balanced
            if action.count('(') != action.count(')'):
                if action.count('(') > action.count(')'):
                    action += ')'
                else:
                    action = 'page.' + action

        # Check if action was recently failed multiple times
        if action in [f.action for f in state.failed_actions[-3:]]:
            failure_count = sum(1 for f in state.failed_actions[-3:] if f.action == action)
            if failure_count >= 2 and self.config.verbose:
                print(f"  ⚠ Warning: Action '{action}' has failed {failure_count} times recently")

        return action

    def analyze_page(self, state: State) -> str:
        """
        Analyze the page for strategic planning.

        Args:
            state: Current environment state

        Returns:
            Analysis text from LLM
        """
        prompt = build_initial_analysis_prompt(state)
        return self.llm_api(prompt)

    def run(self, page, goal: str, context: Optional[Dict[str, Any]] = None) -> State:
        """
        Run automation to achieve the goal with enhanced error recovery.

        Args:
            page: Playwright page object
            goal: Main goal to achieve
            context: Optional context data (e.g., form values)

        Returns:
            Final state after automation
        """
        # Initialize state
        state = create_state(page, main_goal=goal)

        # Optional initial analysis
        if self.config.analyze_on_start:
            if self.config.verbose:
                print("\n=== Initial Analysis ===")
                analysis = self.analyze_page(state)
                print(analysis)
                print("=" * 50 + "\n")

        # Reset tracking variables
        self.repeated_failures.clear()
        self.consecutive_failures = 0

        # Main automation loop
        for step in range(self.config.max_steps):
            state.step_count = step

            # Predict next action
            action = self.predict_next_action(state)

            # Check if done
            if action.strip().lower() == "pass":
                if self.config.verbose:
                    print("\n✅ Agent completed task")
                break

            # Execute action
            result = execute_action(page, action)

            # Update state based on result
            if result.success:
                state.successful_actions.append(action)
                self.repeated_failures.clear()  # Reset on success
                self.consecutive_failures = 0

                if self.config.verbose:
                    print(f"  ✓ Success")
            else:
                # Record failure
                failed = FailedAction(
                    action=action,
                    error=result.error,
                    error_type=result.error_type,
                    timestamp=time.time(),
                    traceback=result.error_traceback,
                    suggestion=result.suggestion
                )
                state.failed_actions.append(failed)
                self.consecutive_failures += 1

                if self.config.verbose:
                    print(f"  ✗ Failed: {result.error}")
                    if result.suggestion:
                        print(f"    Hint: {result.suggestion}")

                # Track repeated failures
                self.repeated_failures[action] = self.repeated_failures.get(action, 0) + 1

                # Trigger strategy change if stuck
                if self.repeated_failures[action] >= self.config.max_repeated_failures:
                    state.needs_strategy_change = True
                    if self.config.verbose:
                        print(f"  ⚠ Action failed {self.repeated_failures[action]} times - switching strategy")

                # Emergency recovery after many consecutive failures
                if self.consecutive_failures >= 5:
                    if self.config.verbose:
                        print(f"  🔄 Too many consecutive failures - attempting recovery")
                    # Force a wait to let page stabilize
                    page.wait_for_timeout(2000)
                    self.consecutive_failures = 0

            # Update state from page
            state = update_state(page, state)

            # Adaptive delay based on failures
            delay = self.config.action_delay
            if self.consecutive_failures > 0:
                # Increase delay after failures to allow page to stabilize
                delay = min(delay * (1 + self.consecutive_failures * 0.5), 3.0)

            if delay > 0:
                time.sleep(delay)

        return state

    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.

        Enhanced version with better parsing logic.
        """
        return extract_code_from_text(response)


def check_goal_completion(page, goal: str, state: State) -> bool:
    """
    Check if the goal has been achieved.

    Enhanced with better success detection.
    """
    goal_lower = goal.lower()
    visible_text_lower = state.visible_text.lower()

    # Success indicators
    success_indicators = [
        'success', 'successful', 'successfully',
        'thank you', 'thanks for',
        'completed', 'complete',
        'submitted', 'submit successful',
        'confirmation', 'confirmed',
        'done', 'finished',
        'registered', 'registration successful'
    ]

    # Check for success indicators
    if any(indicator in visible_text_lower for indicator in success_indicators):
        return True

    # Check URL change for success pages
    if any(success_page in state.url.lower()
           for success_page in ['success', 'thank', 'confirm', 'complete']):
        return True

    # Goal-specific checks
    if "submit" in goal_lower or "complete" in goal_lower:
        # Form likely submitted if we see success indicators
        return any(indicator in visible_text_lower for indicator in success_indicators[:6])

    if "fill" in goal_lower and "form" in goal_lower:
        # Check if all required fields have values
        required_empty = [el for el in state.dom_elements
                          if el.get('required') and not el.get('value')]

        # Form filled if no required fields are empty and submit was successful
        if not required_empty and state.successful_actions:
            last_action = state.successful_actions[-1].lower()
            return "submit" in last_action or "click" in last_action

    return False


# Backward compatibility
def build_action_prompt(state: State) -> str:
    """Legacy function maintained for backward compatibility."""
    return build_action_prompt_optimized(state)