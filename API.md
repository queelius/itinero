# itinero API Documentation

**Elegant web automation through LLM composition**

## Philosophy

itinero follows three core principles:

1. **Do one thing well** (Unix) - Each component has a single, clear responsibility
2. **There should be one obvious way** (Python) - Clean, intuitive APIs
3. **Composition over complexity** (Generic Programming) - Build powerful systems from simple pieces

## Quick Start

```python
from itinero import automate

# Simplest usage - one function call
result = automate(page, "Fill the registration form")
```

That's it! For most use cases, `automate()` is all you need.

## API Levels

itinero provides three API levels, from simple to advanced:

### Level 1: Simple Functions (90% of use cases)

```python
from itinero import automate, fill_form

# General automation
result = automate(page, "Click the submit button", verbose=True)

# Form filling with known data
result = fill_form(page, {
    "firstName": "Alice",
    "lastName": "Smith",
    "email": "alice@example.com"
})
```

**When to use:**
- Quick scripts
- One-off automations
- Testing forms
- Simple workflows

### Level 2: Fluent Builder (Custom configuration)

```python
from itinero import agent

# Build customized agent
my_agent = (agent()
            .model("gemma3n:e2b")
            .with_retry(max_attempts=3)
            .with_recovery()
            .verbose()
            .max_steps(20)
            .build())

# Use it
result = my_agent.run(page, "Complete checkout")
```

**When to use:**
- Need custom configuration
- Reusing agent across multiple pages
- Fine-tuning behavior
- Production deployments

### Level 3: Strategy Composition (Advanced)

```python
from itinero import agent, strategies

# Compose sophisticated strategies
my_strategy = strategies.chain([
    strategies.llm("gpt-4"),
    strategies.llm("gemma3n:e2b"),  # Fallback to faster model
    strategies.heuristic()           # Ultimate fallback
])

my_agent = agent().strategy(my_strategy).build()
```

**When to use:**
- Complex decision logic
- Multiple LLM fallbacks
- Hybrid LLM + rule-based approaches
- Research and experimentation

## Core Concepts

### 1. Actions

Actions are immutable value objects representing what to do:

```python
from core import Action

action = Action(
    type="fill",
    selector="#email",
    value="user@example.com"
)
```

Types: `click`, `fill`, `type`, `select`, `check`, `wait`, `pass`

### 2. Context

Context contains everything needed for decision making:

```python
from core import Context

context = Context(
    goal="Fill the form",
    page_state={...},
    history=[...],
    metadata={...}
)
```

Immutable - each step creates new context with updated history.

### 3. Strategies

Strategies decide what action to take next:

```python
from core import Strategy

class MyStrategy(Strategy):
    def next_action(self, context: Context) -> Optional[Action]:
        # Your logic here
        return Action(type="click", selector="#submit")
```

Strategies are composable - wrap them to add behavior.

### 4. Agents

Agents orchestrate the automation loop:

```python
from core import Agent

agent = Agent(strategy=my_strategy, executor=my_executor)
result = agent.run(page, goal="Fill form")
```

Agents are minimal - complexity comes from composed strategies.

## Fluent Builder API

The builder provides a readable, self-documenting API:

```python
agent()
  .model(name: str)              # Set LLM model
  .llm(callable)                  # Use custom LLM
  .with_retry(max_attempts=3)     # Enable retry
  .with_recovery(strategy="default")  # Enable recovery
  .verbose(enabled=True)          # Enable logging
  .max_steps(steps: int)          # Set step limit
  .strategy(custom_strategy)      # Use custom strategy
  .build()                        # Build the agent
```

All methods return `self` for chaining.

## Strategy Composition

Build complex strategies from simple ones:

```python
from itinero import strategies

# Chain - try each until one succeeds
combined = strategies.chain([
    strategies.llm("gpt-4"),
    strategies.heuristic()
])

# LLM strategy
llm_strat = strategies.llm("gemma3n:e2b")

# Heuristic (rule-based, no LLM)
rule_strat = strategies.heuristic()

# Fallback wrapper
safe_strat = strategies.fallback(heuristic_strategy)
```

### Built-in Strategy Decorators

Strategies can be wrapped to add behavior:

```python
from core import RetryStrategy, RecoveryStrategy

# Add retry logic
retry_strat = RetryStrategy(base_strategy, max_retries=3)

# Add recovery from stuck states
recovery_strat = RecoveryStrategy(base_strategy, recovery_fn)
```

## Custom LLM Integration

Integrate any LLM using a simple callable:

```python
def my_llm(prompt: str) -> str:
    # Call your LLM API
    response = openai.chat.completions.create(...)
    return response.choices[0].message.content

my_agent = agent().llm(my_llm).build()
```

Or implement the `LLM` protocol:

```python
from core import LLM

class MyLLM:
    def generate(self, prompt: str) -> str:
        # Your implementation
        return response

my_agent = agent().llm(MyLLM()).build()
```

## Recovery Strategies

Handle stuck states and errors:

```python
from recovery import (
    default_recovery,      # Balanced approach
    aggressive_recovery,   # Try harder, take risks
    conservative_recovery  # Safe, only waits
)

# Use built-in
agent().with_recovery(strategy="aggressive")

# Or create custom
def my_recovery(context: Context) -> Optional[Action]:
    # Your recovery logic
    return Action(type="wait", selector="body")

from core import RecoveryStrategy
my_strat = RecoveryStrategy(base_strategy, my_recovery)
```

## Model Configurations

Pre-configured for popular models:

```python
# Small, fast models
agent().model("gemma3n:e2b")

# Large, capable models
agent().model("gpt-4")
agent().model("claude-3")
agent().model("llama-70b")
```

Each model gets optimized prompts and parsing based on its capabilities.

## Results and Metrics

All functions return a `Context` with execution history:

```python
result = automate(page, "Fill form")

# Check success
last_result = result.history[-1]
if last_result.success:
    print("Success!")

# Analyze history
total = len(result.history)
successful = sum(1 for r in result.history if r.success)
success_rate = successful / total

# Access metadata
step_count = result.metadata.get('step', 0)
```

For form filling, get detailed results:

```python
result = fill_form(page, data)

print(f"Success: {result.success}")
print(f"Filled: {result.filled_fields}")
print(f"Failed: {result.failed_fields}")
```

## Testing and Mocking

Clean interfaces make testing easy:

```python
from core import Strategy, Context, Action

# Mock strategy for testing
class MockStrategy(Strategy):
    def next_action(self, context):
        return Action(type="pass", selector="")

# Mock LLM
class MockLLM:
    def generate(self, prompt):
        return '{"type":"pass","selector":"","value":""}'

# Test with mocks
agent = Agent(MockStrategy(), MockExecutor())
```

## Performance Considerations

### Prompt Caching

Builders automatically cache prompts:

```python
agent().model("gemma3n:e2b")  # Enables caching
```

Cache hits are transparent and automatic.

### Model Selection

Choose model based on your needs:

- `gemma3n:e2b` - Fast, good for simple forms (50 tokens)
- `gpt-4` - Best accuracy, slower (500 tokens)
- `claude-3` - Great balance (500 tokens)

### Batch Operations

Reuse agents for efficiency:

```python
my_agent = agent().model("gemma3n:e2b").build()

for page in pages:
    result = my_agent.run(page, goal)
```

## Error Handling

Errors are captured in results, not raised:

```python
result = automate(page, "Click button")

if not result.history[-1].success:
    error = result.history[-1].error
    error_type = result.history[-1].error_type
    print(f"{error_type}: {error}")
```

Use recovery strategies to automatically handle errors:

```python
agent().with_recovery()  # Automatic recovery
```

## Best Practices

### 1. Start Simple

```python
# Start here
result = automate(page, goal)

# Only customize when needed
my_agent = agent().model("gemma3n:e2b").build()
```

### 2. Compose from Small Pieces

```python
# Good - small, testable strategies
strategy = strategies.chain([
    strategies.llm("gpt-4"),
    strategies.heuristic()
])

# Avoid - monolithic custom strategy
# (unless you really need it)
```

### 3. Use Type Hints

```python
from core import Agent, Strategy, Context
from typing import Optional

def create_agent() -> Agent:
    return agent().model("gemma3n:e2b").build()
```

### 4. Leverage Immutability

```python
# Context is immutable
original_context = Context(...)
new_context = update_context(original_context)
# original_context unchanged
```

### 5. Prefer Protocols Over Classes

```python
from core import LLM, Executor  # Protocols

# Any object with generate() method works as LLM
# Any object with execute() method works as Executor
```

## Architecture

itinero uses **Ports & Adapters** (Hexagonal Architecture):

```
┌─────────────────────────────────────┐
│         Public API (itinero.py)     │  ← What users see
├─────────────────────────────────────┤
│      Core Domain (core.py)          │  ← Pure business logic
├─────────────────────────────────────┤
│      Adapters (adapters.py)         │  ← Connect to outside world
└─────────────────────────────────────┘
```

**Core** - Pure domain logic, no dependencies
**Adapters** - Bridge core to Playwright, LLMs, etc.
**API** - Friendly interface built on core abstractions

## Extending itinero

### Add Custom Strategy

```python
from core import Strategy, Context, Action
from typing import Optional

class MyStrategy(Strategy):
    def next_action(self, context: Context) -> Optional[Action]:
        # Your logic
        return Action(...)

# Use it
agent().strategy(MyStrategy()).build()
```

### Add Custom Executor

```python
from core import Executor, ExecutionResult

class MyExecutor:
    def execute(self, action_str: str) -> ExecutionResult:
        # Your execution logic
        return ExecutionResult(success=True, ...)

# Use it
from core import Agent
agent = Agent(my_strategy, MyExecutor())
```

### Add Custom Prompt Builder

```python
from core import PromptBuilder

class MyPromptBuilder:
    def build(self, context: dict) -> str:
        # Your prompt generation
        return prompt_string

# Use with LLMStrategy
from core import LLMStrategy
strategy = LLMStrategy(llm, MyPromptBuilder(), parser)
```

## Examples

See `examples.py` for complete working examples:

```bash
# Run all examples
python examples.py

# Run specific example
python examples.py 1  # Simple
python examples.py 2  # Fluent API
python examples.py 3  # Form filling
python examples.py 4  # Strategy composition
python examples.py 5  # Custom LLM
python examples.py 6  # Reusable agent
```

## Migration from v2

Old code:
```python
from agent_v2 import ModernAgent, AgentConfigV2

config = AgentConfigV2(model_name="gemma3n:e2b")
agent = ModernAgent(llm_api, config)
result = agent.run(page, goal)
```

New code:
```python
from itinero import agent

my_agent = agent().model("gemma3n:e2b").llm(llm_api).build()
result = my_agent.run(page, goal)
```

Or even simpler:
```python
from itinero import automate

result = automate(page, goal, model="gemma3n:e2b")
```

## Summary

**Simplest:**
```python
automate(page, goal)
```

**Custom config:**
```python
agent().model("X").with_retry().build().run(page, goal)
```

**Advanced composition:**
```python
agent().strategy(custom_strategy).build().run(page, goal)
```

Choose the level that fits your needs. Start simple, compose when needed.
