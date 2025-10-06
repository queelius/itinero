"""
Core abstractions for itinero web automation.

This module defines the fundamental building blocks that compose to create
web automation agents. Each component does one thing well and can be combined
with others to build complex behaviors.

Philosophy:
- Simple: Each class has a single, clear responsibility
- Composable: Components work together through clean interfaces
- Testable: Pure functions and dependency injection throughout
- Pythonic: Natural, idiomatic Python code
"""

from dataclasses import dataclass
from typing import Protocol, Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod


# ============================================================================
# Core Protocols - Define what things can do, not what they are
# ============================================================================

class LLM(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        ...


class Executor(Protocol):
    """Protocol for action execution."""

    def execute(self, action: str) -> 'ExecutionResult':
        """Execute an action and return result."""
        ...


class PromptBuilder(Protocol):
    """Protocol for building prompts."""

    def build(self, context: Dict[str, Any]) -> str:
        """Build a prompt from context."""
        ...


class ActionParser(Protocol):
    """Protocol for parsing LLM responses into actions."""

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse response into structured action data."""
        ...


# ============================================================================
# Value Objects - Immutable data containers
# ============================================================================

@dataclass(frozen=True)
class Action:
    """An immutable action to be executed."""
    type: str  # click, fill, type, etc.
    selector: str
    value: str = ""
    options: Dict[str, Any] = None

    def to_playwright(self) -> str:
        """Convert to Playwright command string."""
        if self.type == 'click':
            return f"page.click('{self.selector}')"
        elif self.type == 'fill':
            return f"page.fill('{self.selector}', '{self.value}')"
        elif self.type == 'type':
            return f"page.type('{self.selector}', '{self.value}')"
        elif self.type == 'select':
            return f"page.select_option('{self.selector}', '{self.value}')"
        elif self.type == 'check':
            return f"page.check('{self.selector}')"
        elif self.type == 'wait':
            return f"page.wait_for_selector('{self.selector}')"
        elif self.type == 'pass':
            return "pass"
        else:
            # Fallback to raw command if provided
            return self.options.get('command', 'pass') if self.options else 'pass'


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing an action."""
    success: bool
    action: Action
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass(frozen=True)
class Context:
    """Automation context - all information needed for decision making."""
    goal: str
    page_state: Dict[str, Any]
    history: List[ExecutionResult]
    metadata: Dict[str, Any] = None


# ============================================================================
# Strategy Interface - How to decide what to do next
# ============================================================================

class Strategy(ABC):
    """
    Strategy for deciding what action to take next.

    This is the core abstraction that makes agents composable.
    Different strategies can be swapped to change agent behavior.
    """

    @abstractmethod
    def next_action(self, context: Context) -> Optional[Action]:
        """
        Decide what action to take next given the current context.

        Returns:
            Action to execute, or None if goal is complete
        """
        pass


# ============================================================================
# Composable Strategies - Build complex from simple
# ============================================================================

class LLMStrategy(Strategy):
    """Strategy that uses an LLM to decide actions."""

    def __init__(self, llm: LLM, prompt_builder: PromptBuilder, parser: ActionParser):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.parser = parser

    def next_action(self, context: Context) -> Optional[Action]:
        # Build prompt from context
        prompt = self.prompt_builder.build({
            'goal': context.goal,
            'state': context.page_state,
            'history': context.history
        })

        # Get LLM response
        response = self.llm.generate(prompt)

        # Parse into action
        action_data = self.parser.parse(response)

        # Check for completion
        if action_data.get('type') == 'pass':
            return None

        return Action(
            type=action_data.get('type', 'pass'),
            selector=action_data.get('selector', ''),
            value=action_data.get('value', ''),
            options=action_data.get('options')
        )


class RetryStrategy(Strategy):
    """
    Decorator strategy that adds retry logic.

    Example of composing strategies - wrap any strategy with retry behavior.
    """

    def __init__(self, base: Strategy, max_retries: int = 3):
        self.base = base
        self.max_retries = max_retries
        self._retry_count = 0

    def next_action(self, context: Context) -> Optional[Action]:
        # Check if we should retry based on history
        if context.history and not context.history[-1].success:
            self._retry_count += 1
            if self._retry_count >= self.max_retries:
                # Give up, ask base for different action
                self._retry_count = 0
                return self.base.next_action(context)
        else:
            self._retry_count = 0

        return self.base.next_action(context)


class RecoveryStrategy(Strategy):
    """
    Decorator that adds recovery from stuck states.

    Another example of composition - add recovery without modifying base strategy.
    """

    def __init__(self, base: Strategy, recovery_fn: Callable[[Context], Optional[Action]]):
        self.base = base
        self.recovery_fn = recovery_fn

    def next_action(self, context: Context) -> Optional[Action]:
        # Check if stuck (same action failed multiple times)
        if self._is_stuck(context):
            return self.recovery_fn(context)

        return self.base.next_action(context)

    def _is_stuck(self, context: Context) -> bool:
        if len(context.history) < 3:
            return False

        recent_failures = [r for r in context.history[-3:] if not r.success]
        if len(recent_failures) < 3:
            return False

        # Stuck if same action failed 3 times
        actions = [r.action.to_playwright() for r in recent_failures]
        return len(set(actions)) == 1


# ============================================================================
# Agent - The orchestrator
# ============================================================================

class Agent:
    """
    Agent orchestrates the automation loop.

    This class is intentionally minimal - it just coordinates the pieces.
    All complex behavior comes from composing strategies and executors.
    """

    def __init__(self, strategy: Strategy, executor: Executor):
        """
        Create an agent.

        Args:
            strategy: How to decide what to do next
            executor: How to execute actions
        """
        self.strategy = strategy
        self.executor = executor

    def run(self, page, goal: str, max_steps: int = 30) -> Context:
        """
        Run automation to achieve a goal.

        Args:
            page: Playwright page object
            goal: What to accomplish
            max_steps: Maximum steps to take

        Returns:
            Final context with complete history
        """
        # Import here to avoid circular dependency
        from state import create_state

        # Initialize context
        page_state = self._extract_state(page)
        context = Context(
            goal=goal,
            page_state=page_state,
            history=[],
            metadata={'step': 0}
        )

        # Automation loop
        for step in range(max_steps):
            # Decide next action
            action = self.strategy.next_action(context)

            # Check if complete
            if action is None:
                break

            # Execute action
            result = self.executor.execute(action.to_playwright())

            # Update context with result
            context = Context(
                goal=context.goal,
                page_state=self._extract_state(page),
                history=context.history + [ExecutionResult(
                    success=result.success,
                    action=action,
                    error=result.error,
                    error_type=result.error_type,
                    metadata=result.metadata
                )],
                metadata={'step': step + 1}
            )

        return context

    def _extract_state(self, page) -> Dict[str, Any]:
        """Extract page state for context."""
        from state import extract_dom_elements, get_visible_text

        dom_data = extract_dom_elements(page)

        return {
            'url': page.url,
            'elements': dom_data.get('elements', []) if isinstance(dom_data, dict) else dom_data,
            'text': get_visible_text(page, max_length=1000),
            'metadata': {
                'form_count': dom_data.get('formCount', 0) if isinstance(dom_data, dict) else 0,
                'error_messages': dom_data.get('errorMessages', []) if isinstance(dom_data, dict) else []
            }
        }


# ============================================================================
# Builder - Fluent API for constructing agents
# ============================================================================

class AgentBuilder:
    """
    Fluent builder for creating agents with composable strategies.

    Example:
        agent = (AgentBuilder()
                 .with_llm(ollama)
                 .with_model_config("gemma3n:e2b")
                 .with_retry(max_attempts=3)
                 .with_recovery()
                 .build())
    """

    def __init__(self):
        self._llm = None
        self._model_config = None
        self._retry_config = None
        self._recovery = False
        self._executor = None
        self._prompt_builder = None
        self._parser = None

    def with_llm(self, llm: LLM) -> 'AgentBuilder':
        """Set the LLM to use."""
        self._llm = llm
        return self

    def with_model_config(self, model_name: str) -> 'AgentBuilder':
        """Configure for specific model."""
        self._model_config = model_name
        return self

    def with_retry(self, max_attempts: int = 3) -> 'AgentBuilder':
        """Enable retry logic."""
        self._retry_config = {'max_attempts': max_attempts}
        return self

    def with_recovery(self) -> 'AgentBuilder':
        """Enable automatic recovery from stuck states."""
        self._recovery = True
        return self

    def with_executor(self, executor: Executor) -> 'AgentBuilder':
        """Set custom executor."""
        self._executor = executor
        return self

    def build(self) -> Agent:
        """Build the agent with configured components."""
        # Create base strategy
        if self._llm is None:
            raise ValueError("LLM is required")

        # Auto-create prompt builder and parser if not provided
        if self._prompt_builder is None:
            from adapters import create_prompt_builder
            self._prompt_builder = create_prompt_builder(self._model_config or "default")

        if self._parser is None:
            from adapters import create_parser
            self._parser = create_parser(self._model_config or "default")

        strategy = LLMStrategy(self._llm, self._prompt_builder, self._parser)

        # Wrap with decorators based on configuration
        if self._retry_config:
            strategy = RetryStrategy(strategy, self._retry_config['max_attempts'])

        if self._recovery:
            from recovery import default_recovery
            strategy = RecoveryStrategy(strategy, default_recovery)

        # Create executor if not provided
        if self._executor is None:
            from adapters import PlaywrightExecutor
            self._executor = PlaywrightExecutor()

        return Agent(strategy, self._executor)
