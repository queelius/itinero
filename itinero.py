"""
itinero - Elegant web automation through LLM composition

This is the main public API. It provides a clean, composable interface for
building web automation agents.

Example usage:

    # Simple usage with defaults
    from itinero import automate

    result = automate(page, "Fill the registration form")

    # Custom configuration
    from itinero import agent

    my_agent = (agent()
                .model("gemma3n:e2b")
                .with_retry(max_attempts=3)
                .with_recovery()
                .verbose()
                .build())

    result = my_agent.run(page, "Complete checkout")

    # Advanced composition
    from itinero import agent, strategies

    custom_strategy = strategies.chain([
        strategies.llm("claude-3"),
        strategies.fallback(strategies.heuristic())
    ])

    my_agent = agent().strategy(custom_strategy).build()
"""

from typing import Optional, Callable, List, Dict
from dataclasses import dataclass

from core import Agent, Strategy, AgentBuilder, Context
from adapters import OllamaLLM, PlaywrightExecutor, CallableLLM
from recovery import default_recovery, aggressive_recovery, conservative_recovery


# ============================================================================
# High-level convenience functions
# ============================================================================

def automate(page, goal: str, model: str = "gemma3n:e2b", **kwargs) -> Context:
    """
    Simplest way to run automation - one function call.

    Args:
        page: Playwright page object
        goal: What to accomplish (e.g., "Fill the registration form")
        model: LLM model to use
        **kwargs: Additional configuration

    Returns:
        Context with execution history

    Example:
        result = automate(page, "Click the submit button")
        if result.history[-1].success:
            print("Success!")
    """
    builder = agent().model(model)

    # Apply common defaults
    if kwargs.get('verbose'):
        builder = builder.verbose()
    if kwargs.get('retry', True):
        builder = builder.with_retry()
    if kwargs.get('recovery', True):
        builder = builder.with_recovery()

    my_agent = builder.build()

    # Set page on executor
    if hasattr(my_agent.executor, 'set_page'):
        my_agent.executor.set_page(page)

    return my_agent.run(page, goal, max_steps=kwargs.get('max_steps', 30))


def agent() -> 'FluentAgentBuilder':
    """
    Start building a custom agent with fluent interface.

    Returns:
        Builder for composing agent with desired capabilities

    Example:
        my_agent = (agent()
                    .model("gemma3n:e2b")
                    .with_retry(max_attempts=5)
                    .verbose()
                    .build())
    """
    return FluentAgentBuilder()


# ============================================================================
# Fluent Builder - Makes configuration read like natural language
# ============================================================================

class FluentAgentBuilder:
    """
    Fluent builder that creates agents through method chaining.

    This provides an elegant, discoverable API that reads like English.
    """

    def __init__(self):
        self._builder = AgentBuilder()
        self._model_name = "gemma3n:e2b"
        self._verbose = False
        self._max_steps = 30

    def model(self, name: str) -> 'FluentAgentBuilder':
        """
        Set the LLM model to use.

        Args:
            name: Model name (e.g., "gemma3n:e2b", "gpt-4", "claude-3")

        Returns:
            Self for chaining
        """
        self._model_name = name
        self._builder.with_model_config(name)
        return self

    def llm(self, llm_callable: Callable[[str], str]) -> 'FluentAgentBuilder':
        """
        Use a custom LLM callable.

        Args:
            llm_callable: Function that takes prompt and returns response

        Returns:
            Self for chaining
        """
        self._builder.with_llm(CallableLLM(llm_callable))
        return self

    def with_retry(self, max_attempts: int = 3) -> 'FluentAgentBuilder':
        """
        Enable automatic retry on failures.

        Args:
            max_attempts: Maximum retry attempts

        Returns:
            Self for chaining
        """
        self._builder.with_retry(max_attempts)
        return self

    def with_recovery(self, strategy: str = "default") -> 'FluentAgentBuilder':
        """
        Enable recovery from stuck states.

        Args:
            strategy: "default", "aggressive", or "conservative"

        Returns:
            Self for chaining
        """
        recovery_map = {
            "default": default_recovery,
            "aggressive": aggressive_recovery,
            "conservative": conservative_recovery
        }

        if strategy not in recovery_map:
            raise ValueError(f"Unknown recovery strategy: {strategy}")

        self._builder.with_recovery()
        return self

    def verbose(self, enabled: bool = True) -> 'FluentAgentBuilder':
        """
        Enable verbose logging.

        Args:
            enabled: Whether to enable verbose output

        Returns:
            Self for chaining
        """
        self._verbose = enabled
        return self

    def max_steps(self, steps: int) -> 'FluentAgentBuilder':
        """
        Set maximum automation steps.

        Args:
            steps: Maximum steps to execute

        Returns:
            Self for chaining
        """
        self._max_steps = steps
        return self

    def strategy(self, custom_strategy: Strategy) -> 'FluentAgentBuilder':
        """
        Use a completely custom strategy.

        This is an escape hatch for advanced users who want full control.

        Args:
            custom_strategy: Custom strategy implementation

        Returns:
            Self for chaining
        """
        # Create executor
        executor = PlaywrightExecutor()
        self._agent = Agent(custom_strategy, executor)
        self._custom_strategy = True
        return self

    def build(self) -> Agent:
        """
        Build the agent with configured options.

        Returns:
            Configured agent ready to run
        """
        # If custom strategy was set, return that agent
        if hasattr(self, '_custom_strategy'):
            return self._agent

        # Auto-create LLM if not provided
        if self._builder._llm is None:
            llm = OllamaLLM(self._model_name)
            self._builder.with_llm(llm)

        # Build the agent
        agent_instance = self._builder.build()

        # Wrap with verbose logging if requested
        if self._verbose:
            agent_instance = VerboseAgent(agent_instance)

        return agent_instance


# ============================================================================
# Strategy Combinators - Compose complex strategies from simple ones
# ============================================================================

class strategies:
    """
    Namespace for strategy composition functions.

    This enables building complex decision-making logic by combining
    simple strategies.
    """

    @staticmethod
    def chain(strategy_list: List[Strategy]) -> Strategy:
        """
        Chain strategies - try each until one succeeds.

        Args:
            strategy_list: List of strategies to try in order

        Returns:
            Combined strategy

        Example:
            strategy = strategies.chain([
                strategies.llm("gpt-4"),
                strategies.heuristic()
            ])
        """
        return ChainStrategy(strategy_list)

    @staticmethod
    def llm(model: str) -> Strategy:
        """
        Create an LLM-based strategy.

        Args:
            model: Model name to use

        Returns:
            Strategy that uses the specified model
        """
        from core import LLMStrategy
        from adapters import create_prompt_builder, create_parser

        llm = OllamaLLM(model)
        prompt_builder = create_prompt_builder(model)
        parser = create_parser(model)

        return LLMStrategy(llm, prompt_builder, parser)

    @staticmethod
    def heuristic() -> Strategy:
        """
        Create a heuristic-based strategy (no LLM).

        Returns:
            Strategy that uses simple heuristics

        Example:
            # Use heuristics as fallback
            strategy = strategies.chain([
                strategies.llm("gemma3n:e2b"),
                strategies.heuristic()
            ])
        """
        return HeuristicStrategy()

    @staticmethod
    def fallback(fallback_strategy: Strategy) -> Strategy:
        """
        Create a fallback wrapper.

        This is a specialized version of chain() for the common case
        of having one main strategy with a fallback.

        Args:
            fallback_strategy: Strategy to use if main fails

        Returns:
            Decorator that adds fallback behavior
        """
        return lambda main: ChainStrategy([main, fallback_strategy])


# ============================================================================
# Internal Strategy Implementations
# ============================================================================

class ChainStrategy(Strategy):
    """Try multiple strategies in sequence until one works."""

    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies

    def next_action(self, context: Context) -> Optional['Action']:
        for strategy in self.strategies:
            try:
                action = strategy.next_action(context)
                if action is not None:
                    return action
            except Exception:
                continue  # Try next strategy
        return None


class HeuristicStrategy(Strategy):
    """Simple rule-based strategy without LLM."""

    def next_action(self, context: Context) -> Optional['Action']:
        from core import Action

        elements = context.page_state.get('elements', [])

        # Look for required empty fields
        for el in elements:
            if (el.get('required') and
                el.get('tag') == 'input' and
                not el.get('value')):

                selector = el.get('selector', '')
                value = self._guess_value(selector, el)

                if value:
                    return Action(type='fill', selector=selector, value=value)

        # Look for submit button
        for el in elements:
            if (el.get('tag') == 'button' and
                el.get('type') == 'submit'):
                return Action(type='click', selector=el.get('selector', ''))

        # Give up
        return None

    def _guess_value(self, selector: str, element: Dict) -> str:
        """Guess appropriate value for a field."""
        selector_lower = selector.lower()
        label = (element.get('label') or '').lower()

        if 'email' in selector_lower or 'email' in label:
            return 'test@example.com'
        elif 'name' in selector_lower or 'name' in label:
            return 'John Doe'
        elif 'phone' in selector_lower or 'phone' in label:
            return '555-0123'

        return 'test'


# ============================================================================
# Verbose Wrapper - Adds logging to any agent
# ============================================================================

class VerboseAgent:
    """
    Decorator that adds verbose logging to any agent.

    This demonstrates the Decorator pattern for adding cross-cutting concerns.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.executor = agent.executor  # Expose for compatibility

    def run(self, page, goal: str, max_steps: int = 30) -> Context:
        """Run with verbose logging."""
        print(f"\n🎯 Goal: {goal}")
        print(f"📊 Max steps: {max_steps}")
        print("-" * 60)

        # Set page on executor
        if hasattr(self.agent.executor, 'set_page'):
            self.agent.executor.set_page(page)

        # Run with logging
        context = None
        step = 0

        # Call original run but intercept to log
        original_executor = self.agent.executor.execute

        def logged_execute(action_str: str):
            nonlocal step
            step += 1

            # Parse action for display
            import re
            match = re.match(r"page\.(\w+)\((.*)\)", action_str)
            if match:
                method = match.group(1)
                args = match.group(2)[:30]
                print(f"[{step:2d}] {method}({args}...)", end=" ")
            else:
                print(f"[{step:2d}] {action_str[:40]}", end=" ")

            # Execute
            result = original_executor(action_str)

            # Log result
            if result.success:
                print("✓")
            else:
                error = result.error[:40] if result.error else "Unknown error"
                print(f"✗ {error}")

            return result

        # Temporarily replace executor
        self.agent.executor.execute = logged_execute

        try:
            context = self.agent.run(page, goal, max_steps)
        finally:
            # Restore original
            self.agent.executor.execute = original_executor

        # Print summary
        print("-" * 60)
        if context:
            total = len(context.history)
            successful = sum(1 for r in context.history if r.success)
            print(f"✅ Completed: {successful}/{total} actions successful")

        return context


# ============================================================================
# Batch Operations - High-level utilities for common tasks
# ============================================================================

@dataclass
class FormFillResult:
    """Result of form filling operation."""
    success: bool
    filled_fields: List[str]
    failed_fields: List[str]
    context: Context


def fill_form(page, data: dict, model: str = "gemma3n:e2b") -> FormFillResult:
    """
    Fill a form with provided data in one shot.

    This is optimized for the common case of filling known data into a form.

    Args:
        page: Playwright page object
        data: Dictionary mapping field names/IDs to values
        model: LLM model to use

    Returns:
        FormFillResult with success status and details

    Example:
        result = fill_form(page, {
            "firstName": "Alice",
            "lastName": "Smith",
            "email": "alice@example.com"
        })

        if result.success:
            print(f"Filled {len(result.filled_fields)} fields")
    """
    # Build goal from data
    goal = f"Fill form: {', '.join(f'{k}={v}' for k, v in list(data.items())[:3])}"

    # Run automation
    my_agent = agent().model(model).with_retry().build()

    if hasattr(my_agent.executor, 'set_page'):
        my_agent.executor.set_page(page)

    context = my_agent.run(page, goal, max_steps=len(data) * 2 + 5)

    # Analyze results
    filled = []
    failed = []

    for field_name in data.keys():
        try:
            # Check if field was filled
            value = page.evaluate(f"document.querySelector('#{field_name}')?.value || "
                                f"document.querySelector('[name=\"{field_name}\"]')?.value || ''")
            if value == data[field_name]:
                filled.append(field_name)
            else:
                failed.append(field_name)
        except:
            failed.append(field_name)

    return FormFillResult(
        success=len(failed) == 0,
        filled_fields=filled,
        failed_fields=failed,
        context=context
    )


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    # Main functions
    'automate',
    'agent',
    'fill_form',

    # Building blocks
    'strategies',

    # Types (for advanced users)
    'Agent',
    'Strategy',
    'Context',
    'FormFillResult',
]
