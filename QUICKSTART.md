# itinero Quick Start

## Install

```bash
pip3 install playwright requests
playwright install chromium
```

## 30-Second Example

```python
from playwright.sync_api import sync_playwright
from itinero import automate

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com/form")

    # That's it!
    result = automate(page, "Fill the registration form")

    browser.close()
```

## Common Patterns

### 1. Simple Automation

```python
from itinero import automate

result = automate(page, "Click the submit button")
```

### 2. Form Filling

```python
from itinero import fill_form

result = fill_form(page, {
    "firstName": "Alice",
    "lastName": "Smith",
    "email": "alice@example.com"
})

print(f"Success: {result.success}")
print(f"Filled: {result.filled_fields}")
```

### 3. Custom Configuration

```python
from itinero import agent

my_agent = (agent()
            .model("gemma3n:e2b")
            .with_retry(max_attempts=3)
            .with_recovery()
            .verbose()
            .build())

result = my_agent.run(page, "Complete the checkout process")
```

### 4. Reusable Agent

```python
my_agent = agent().model("gemma3n:e2b").build()

# Use multiple times
for url in urls:
    page.goto(url)
    result = my_agent.run(page, "Extract product info")
```

### 5. Custom LLM

```python
def my_llm(prompt):
    # Call any LLM API
    response = openai.chat.completions.create(...)
    return response.choices[0].message.content

my_agent = agent().llm(my_llm).build()
```

### 6. Strategy Composition

```python
from itinero import agent, strategies

# Chain strategies with fallback
combined = strategies.chain([
    strategies.llm("gpt-4"),      # Try first
    strategies.heuristic()        # Fallback
])

my_agent = agent().strategy(combined).build()
```

## API Cheat Sheet

### Functions

```python
# Simple automation
automate(page, goal, model="gemma3n:e2b", verbose=False)

# Form filling
fill_form(page, data_dict, model="gemma3n:e2b")

# Get builder
agent()
```

### Builder Methods

```python
agent()
  .model(name)                    # Set model
  .llm(callable)                  # Custom LLM
  .with_retry(max_attempts=3)     # Enable retry
  .with_recovery(strategy="default")  # Enable recovery
  .verbose(enabled=True)          # Logging
  .max_steps(steps)               # Step limit
  .strategy(custom_strategy)      # Custom strategy
  .build()                        # Build agent
```

### Strategy Composition

```python
strategies.chain([s1, s2])      # Try each in order
strategies.llm(model)            # LLM strategy
strategies.heuristic()           # Rule-based
strategies.fallback(strategy)    # Wrapper
```

## Results

### automate() / agent.run()

Returns `Context`:
```python
result = automate(page, goal)

# Check success
if result.history[-1].success:
    print("Success!")

# Analyze
total = len(result.history)
successful = sum(1 for r in result.history if r.success)
```

### fill_form()

Returns `FormFillResult`:
```python
result = fill_form(page, data)

print(f"Success: {result.success}")
print(f"Filled: {result.filled_fields}")
print(f"Failed: {result.failed_fields}")
```

## Models

Pre-configured models:
- `gemma3n:e2b` - Fast, small (default)
- `gpt-4` - Best accuracy
- `claude-3` - Great balance
- `llama-70b` - Large, capable

## Recovery Strategies

```python
.with_recovery(strategy="default")      # Balanced
.with_recovery(strategy="aggressive")   # Try harder
.with_recovery(strategy="conservative") # Safe
```

## Error Handling

Errors captured in results, not raised:
```python
result = automate(page, goal)

if not result.history[-1].success:
    print(f"Error: {result.history[-1].error}")
    print(f"Type: {result.history[-1].error_type}")
```

## Examples

Run included examples:
```bash
python examples.py      # List all
python examples.py 1    # Simple
python examples.py 2    # Fluent API
python examples.py 3    # Form filling
python examples.py 4    # Strategies
```

## Debugging

Enable verbose mode:
```python
# Function
automate(page, goal, verbose=True)

# Builder
agent().verbose().build()
```

Output:
```
🎯 Goal: Fill form
📊 Max steps: 30
──────────────────────────────────────
[ 1] fill(#firstName...) ✓
[ 2] fill(#lastName...) ✓
[ 3] click(#submit...) ✓
──────────────────────────────────────
✅ Completed: 3/3 actions successful
```

## Common Issues

### LLM not responding

Check Ollama is running:
```bash
ollama serve
ollama pull gemma3n:e2b
```

### Element not found

Use recovery:
```python
agent().with_recovery().build()
```

### Wrong values filled

Try different model:
```python
agent().model("gpt-4").build()
```

## Next Steps

1. Read [API.md](API.md) for complete docs
2. Review [examples.py](examples.py) for patterns
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) for internals

## Philosophy

- **Start simple**: Use `automate()` first
- **Customize gradually**: Add options as needed
- **Compose when advanced**: Use strategies for complex logic

**Simple → Custom → Advanced**

That's it! Happy automating! 🚀
