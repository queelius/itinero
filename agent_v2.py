"""
Modern LLM agent with pure JSON communication and model-specific optimization.
No backward compatibility - pure performance.
"""

import time
import json
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field

from state import State, FailedAction, create_state, update_state
from executor import execute_action
from prompts_v2 import PromptsV2, SpecializedPrompts
from model_prompts import MODEL_CONFIGS


@dataclass
class AgentConfigV2:
    """Modern agent configuration."""
    model_name: str = "default"
    max_steps: int = 30
    max_retries: int = 2
    delay_ms: int = 300
    verbose: bool = True
    pure_json: bool = True
    cache_prompts: bool = True
    adaptive_delay: bool = True
    emergency_recovery_after: int = 5  # Consecutive failures


class ModernAgent:
    """
    Next-gen web automation agent.
    Pure JSON protocol, model-specific optimization, no legacy code.
    """

    def __init__(self, llm_api: Callable[[str], str], config: Optional[AgentConfigV2] = None):
        self.llm_api = llm_api
        self.config = config or AgentConfigV2()
        self.prompts = PromptsV2(self.config.model_name)
        self.model_config = MODEL_CONFIGS.get(self.config.model_name, MODEL_CONFIGS["default"])

        # Tracking
        self.metrics = {
            'total_actions': 0,
            'successful': 0,
            'failed': 0,
            'cache_hits': 0,
            'recovery_attempts': 0
        }
        self.consecutive_failures = 0
        self.action_history = []

    def predict_action(self, state: State) -> str:
        """Predict next action using pure JSON protocol."""

        # Check if stuck and need recovery
        if self._is_stuck(state):
            return self._recovery_action(state)

        # Generate prompt
        prompt = self.prompts.get_action_prompt(state)

        # Call LLM with model-specific settings
        response = self._call_llm(prompt)

        # Parse JSON response
        action_data = self.prompts.parse_response(response)

        # Build Playwright command
        action = self.prompts.build_action(action_data)

        # Validate action
        action = self._validate_action(action, state)

        if self.config.verbose:
            self._log_action(state.step_count, action, action_data)

        return action

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with optimized settings."""

        # For gemma3n:e2b, we might want to add special formatting
        if self.config.model_name == "gemma3n:e2b":
            # Gemma likes clear JSON examples
            prompt = prompt + "\nExample: {\"a\":\"click\",\"s\":\"#submit\",\"v\":\"\"}"

        response = self.llm_api(prompt)

        # Track metrics
        if hasattr(self.prompts.cache, 'cache') and prompt in self.prompts.cache.cache:
            self.metrics['cache_hits'] += 1

        return response

    def _is_stuck(self, state: State) -> bool:
        """Detect if agent is stuck."""

        if len(state.failed_actions) < 3:
            return False

        # Check last 3 actions
        recent = state.failed_actions[-3:]
        actions = [f.action for f in recent]

        # Stuck if same action failed 3 times
        return len(set(actions)) == 1

    def _recovery_action(self, state: State) -> str:
        """Generate recovery action when stuck."""

        self.metrics['recovery_attempts'] += 1

        if self.config.verbose:
            print(f"  🔄 Recovery mode (attempt {self.metrics['recovery_attempts']})")

        # Get last failure
        last_failure = state.failed_actions[-1]

        # Use specialized recovery prompt
        recovery_prompt = SpecializedPrompts.recovery_prompt(
            last_failure.error,
            last_failure.selector or ""
        )

        response = self.llm_api(recovery_prompt)

        try:
            data = json.loads(response)
            strategies = data.get('recovery', [])
            if strategies:
                first_strategy = strategies[0]
                return self.prompts.build_action(first_strategy)
        except:
            pass

        # Fallback: wait and retry
        return "page.wait_for_timeout(1000)"

    def _validate_action(self, action: str, state: State) -> str:
        """Validate and fix common issues."""

        # Fix quote imbalance
        if action.count("'") % 2 != 0:
            action += "'"
        if action.count('"') % 2 != 0:
            action += '"'

        # Fix parentheses
        if action.count('(') != action.count(')'):
            if action.count('(') > action.count(')'):
                action += ')'

        # Warn about repeated failures
        if self.action_history[-3:].count(action) >= 2:
            if self.config.verbose:
                print(f"  ⚠ Action repeated multiple times: {action[:30]}")

        return action

    def _log_action(self, step: int, action: str, data: Dict[str, Any]):
        """Log action with details."""

        # Compact logging for gemma3n:e2b
        if self.config.model_name == "gemma3n:e2b":
            action_type = data.get('a', data.get('action', ''))
            selector = data.get('s', data.get('selector', ''))[:20]
            print(f"[{step:2d}] {action_type}:{selector}")
        else:
            print(f"Step {step}: {action}")

    def run(self, page, goal: str, context: Optional[Dict[str, Any]] = None) -> State:
        """Execute automation with modern approach."""

        # Initialize
        state = create_state(page, main_goal=goal)
        self.consecutive_failures = 0
        self.action_history = []

        if self.config.verbose:
            print(f"\n🎯 Goal: {goal}")
            print(f"🤖 Model: {self.config.model_name}")
            print("-" * 50)

        # Main loop
        for step in range(self.config.max_steps):
            state.step_count = step

            # Predict action
            action = self.predict_action(state)
            self.action_history.append(action)

            # Check completion
            if action == "pass":
                if self.config.verbose:
                    print("\n✅ Task completed")
                    self._print_metrics()
                break

            # Execute
            result = execute_action(page, action)
            self.metrics['total_actions'] += 1

            # Handle result
            if result.success:
                self.metrics['successful'] += 1
                state.successful_actions.append(action)
                self.consecutive_failures = 0

                if self.config.verbose:
                    print(f"  ✓")
            else:
                self.metrics['failed'] += 1
                self.consecutive_failures += 1

                # Record failure
                failed = FailedAction(
                    action=action,
                    error=result.error,
                    error_type=result.error_type,
                    timestamp=time.time(),
                    traceback=result.error_traceback,
                    suggestion=result.suggestion,
                    selector=self._extract_selector(action)
                )
                state.failed_actions.append(failed)

                if self.config.verbose:
                    print(f"  ✗ {result.error[:40]}")

                # Emergency recovery
                if self.consecutive_failures >= self.config.emergency_recovery_after:
                    if self.config.verbose:
                        print(f"  🚨 Emergency recovery")
                    page.wait_for_timeout(2000)
                    self.consecutive_failures = 0

            # Update state
            state = update_state(page, state)

            # Adaptive delay
            delay = self._calculate_delay()
            if delay > 0:
                time.sleep(delay / 1000)  # Convert to seconds

        return state

    def _calculate_delay(self) -> int:
        """Calculate adaptive delay in milliseconds."""

        base_delay = self.config.delay_ms

        if not self.config.adaptive_delay:
            return base_delay

        # Increase delay after failures
        if self.consecutive_failures > 0:
            return min(base_delay * (1 + self.consecutive_failures), 2000)

        return base_delay

    def _extract_selector(self, action: str) -> Optional[str]:
        """Extract selector from action."""

        import re
        match = re.search(r"['\"]([^'\"]+)['\"]", action)
        return match.group(1) if match else None

    def _print_metrics(self):
        """Print performance metrics."""

        total = self.metrics['total_actions']
        if total > 0:
            success_rate = (self.metrics['successful'] / total) * 100
            print(f"\n📊 Metrics:")
            print(f"  Actions: {total}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Cache hits: {self.metrics['cache_hits']}")
            print(f"  Recoveries: {self.metrics['recovery_attempts']}")


class BatchAgent(ModernAgent):
    """
    Agent that can execute multiple actions in batch.
    Useful for form filling where we know all the fields.
    """

    def fill_form(self, page, form_data: Dict[str, str]) -> State:
        """Fill a form with known data in batch."""

        state = create_state(page, main_goal="Fill form")

        # Get form fields from page
        form_fields = [el for el in state.dom_elements
                      if el.get('tag') in ['input', 'select', 'textarea']]

        # Generate batch fill prompt
        prompt = SpecializedPrompts.form_fill_prompt(form_fields, form_data)
        response = self.llm_api(prompt)

        try:
            data = json.loads(response)
            actions = data.get('actions', [])

            # Execute all actions
            for action_data in actions:
                action = self.prompts.build_action(action_data)
                result = execute_action(page, action)

                if result.success:
                    state.successful_actions.append(action)
                else:
                    print(f"Failed: {action} - {result.error[:50]}")

        except Exception as e:
            print(f"Batch fill error: {e}")

        return state