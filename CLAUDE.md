# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Simple LLM Web Automation Agent that uses Playwright for browser automation and integrates with LLMs (specifically Ollama) to intelligently navigate and interact with web pages. The agent can understand web page context, predict actions, learn from failures, and adapt its strategy to complete automation goals.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

1. **State Management** (`state.py`): Captures complete web page state including DOM elements, visible text, and maintains action history (successful and failed). The `State` dataclass tracks goals, subgoals, and failure patterns to enable learning.

2. **Action Execution** (`executor.py`): Safely executes Playwright commands with comprehensive error handling. Provides detailed failure analysis with suggestions for recovery. Special handling for file:// protocol issues.

3. **Prompt Engineering** (`prompts.py`): Constructs context-aware prompts for the LLM including current page state, action history, and strategic hints based on failure analysis.

4. **Agent Core** (`simple_llm_agent.py`): Orchestrates the automation loop - predicts actions via LLM, executes them, updates state, and adapts strategy based on failures. Implements failure tracking and strategy change triggers.

5. **Test Harness** (`test_agent.py`): Demonstrates agent usage with Ollama integration for testing on registration forms.

## Development Commands

### Install Dependencies
```bash
pip3 install -r requirements.txt
# Note: playwright is listed but not currently installed - install if needed:
pip3 install playwright
playwright install chromium
```

### Run Test Agent
```bash
# Ensure Ollama is running first:
ollama serve
ollama pull gemma3n:e2b

# Run the test
python3 test_agent.py
```

### Testing Individual Components
```bash
# Test state extraction on a live page
python3 -c "from state import *; from playwright.sync_api import sync_playwright; p = sync_playwright().start(); browser = p.chromium.launch(); page = browser.new_page(); page.goto('https://example.com'); state = create_state(page); print(state.dom_elements[:5])"

# Test executor with mock actions
python3 -c "from executor import *; from playwright.sync_api import sync_playwright; p = sync_playwright().start(); browser = p.chromium.launch(); page = browser.new_page(); page.goto('https://example.com'); result = execute_action(page, 'page.wait_for_timeout(1000)'); print(result)"
```

## Key Design Patterns

1. **Failure-Aware Learning**: The agent tracks failed actions with detailed error information and automatically suggests alternative approaches when actions fail repeatedly.

2. **State-Driven Prompting**: Each LLM prompt includes complete context - current DOM state with element values, action history, and strategic hints derived from failure patterns.

3. **Safe Execution**: All Playwright commands are executed in a sandboxed namespace with comprehensive exception handling and error categorization.

4. **Strategy Adaptation**: When an action fails multiple times (configurable via `max_repeated_failures`), the agent triggers a strategy change to avoid getting stuck.

## Configuration

The `AgentConfig` dataclass in `simple_llm_agent.py` controls behavior:
- `max_steps`: Maximum automation steps (default: 30)
- `max_repeated_failures`: Failures before strategy change (default: 3)
- `action_delay`: Delay between actions in seconds (default: 0.5)
- `verbose`: Enable step-by-step logging (default: True)
- `analyze_on_start`: Perform initial page analysis (default: False)

## Integration Points

The agent is designed to work with any LLM via a simple callback interface:
```python
def llm_api(prompt: str) -> str:
    # Your LLM integration here
    return llm_response

agent = SimpleLLMAgent(llm_api)
```

Currently configured for Ollama with the gemma3n:e2b model, but easily adaptable to OpenAI, Anthropic, or other LLM providers.