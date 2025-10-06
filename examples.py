#!/usr/bin/env python3
"""
Examples demonstrating the itinero API.

These examples show the progression from simple to advanced usage,
demonstrating the composability and elegance of the API.
"""

import time
from playwright.sync_api import sync_playwright


# ============================================================================
# Example 1: Simplest possible usage
# ============================================================================

def example_simple():
    """
    The simplest way to automate - just one function call.

    This is perfect for quick scripts or one-off automations.
    """
    print("=" * 70)
    print("Example 1: Simple automation with defaults")
    print("=" * 70)

    from itinero import automate

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("file:///home/spinoza/github/alpha/itinero/integrations/user-registration-forms/index.html")

        # That's it! One function call.
        result = automate(
            page,
            "Fill firstName with Alice",
            verbose=True
        )

        # Check result
        if result.history[-1].success:
            print("\n✅ Automation succeeded!")
        else:
            print("\n❌ Automation failed")

        time.sleep(2)
        browser.close()


# ============================================================================
# Example 2: Custom configuration with fluent API
# ============================================================================

def example_fluent():
    """
    Build a customized agent using the fluent builder API.

    This reads like natural language and is self-documenting.
    """
    print("\n" + "=" * 70)
    print("Example 2: Fluent API for custom configuration")
    print("=" * 70)

    from itinero import agent

    # Build agent with desired capabilities
    my_agent = (agent()
                .model("gemma3n:e2b")
                .with_retry(max_attempts=3)
                .with_recovery()
                .verbose()
                .max_steps(20)
                .build())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("file:///home/spinoza/github/alpha/itinero/integrations/user-registration-forms/index.html")

        # Use the agent
        result = my_agent.run(page, "Fill firstName with Bob and lastName with Jones")

        print(f"\n📊 Executed {len(result.history)} actions")

        time.sleep(2)
        browser.close()


# ============================================================================
# Example 3: Form filling utility
# ============================================================================

def example_form_fill():
    """
    Use the specialized form-filling API for structured data entry.

    This is optimized for the common case of filling forms with known data.
    """
    print("\n" + "=" * 70)
    print("Example 3: High-level form filling")
    print("=" * 70)

    from itinero import fill_form

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("file:///home/spinoza/github/alpha/itinero/integrations/user-registration-forms/index.html")

        # Define data to fill
        form_data = {
            "firstName": "Carol",
            "lastName": "Williams",
            "email": "carol@example.com",
            "phone": "555-9876"
        }

        # Fill the form
        result = fill_form(page, form_data)

        # Check results
        print(f"\n✅ Filled: {', '.join(result.filled_fields)}")
        if result.failed_fields:
            print(f"❌ Failed: {', '.join(result.failed_fields)}")

        print(f"\nSuccess: {result.success}")

        time.sleep(2)
        browser.close()


# ============================================================================
# Example 4: Strategy composition
# ============================================================================

def example_strategies():
    """
    Compose complex decision-making from simple strategies.

    This demonstrates the power of composition - build sophisticated
    behavior from simple, testable components.
    """
    print("\n" + "=" * 70)
    print("Example 4: Strategy composition")
    print("=" * 70)

    from itinero import agent, strategies

    # Create a strategy that chains LLM with heuristic fallback
    combined_strategy = strategies.chain([
        strategies.llm("gemma3n:e2b"),
        strategies.heuristic()  # Fallback if LLM fails
    ])

    # Build agent with custom strategy
    my_agent = (agent()
                .strategy(combined_strategy)
                .verbose()
                .build())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("file:///home/spinoza/github/alpha/itinero/integrations/user-registration-forms/index.html")

        result = my_agent.run(page, "Fill the form with test data")

        print(f"\n📊 Total actions: {len(result.history)}")

        time.sleep(2)
        browser.close()


# ============================================================================
# Example 5: Custom LLM integration
# ============================================================================

def example_custom_llm():
    """
    Integrate with any LLM using a simple callable.

    This shows how easy it is to adapt the framework to different LLMs.
    """
    print("\n" + "=" * 70)
    print("Example 5: Custom LLM integration")
    print("=" * 70)

    from itinero import agent

    # Define your own LLM callable
    def my_llm_function(prompt: str) -> str:
        """
        This could call any LLM API - OpenAI, Anthropic, local model, etc.
        For demo purposes, we'll return a simple action.
        """
        print(f"LLM received: {prompt[:50]}...")

        # Return JSON action
        return '{"type":"fill","selector":"#firstName","value":"Demo"}'

    # Build agent with custom LLM
    my_agent = (agent()
                .llm(my_llm_function)
                .verbose()
                .build())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("file:///home/spinoza/github/alpha/itinero/integrations/user-registration-forms/index.html")

        result = my_agent.run(page, "Fill firstName", max_steps=1)

        print(f"\n📊 Actions executed: {len(result.history)}")

        time.sleep(2)
        browser.close()


# ============================================================================
# Example 6: Reusable agent across multiple pages
# ============================================================================

def example_reusable():
    """
    Create an agent once and reuse it for multiple automations.

    This is efficient for batch processing or testing multiple pages.
    """
    print("\n" + "=" * 70)
    print("Example 6: Reusable agent")
    print("=" * 70)

    from itinero import agent

    # Build agent once
    my_agent = (agent()
                .model("gemma3n:e2b")
                .with_retry()
                .build())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        # Use same agent for multiple pages
        for i, name in enumerate(["Alice", "Bob", "Carol"], 1):
            print(f"\n--- Test {i}: {name} ---")

            page = browser.new_page()
            page.goto("file:///home/spinoza/github/alpha/itinero/integrations/user-registration-forms/index.html")

            result = my_agent.run(page, f"Fill firstName with {name}")

            success = result.history[-1].success if result.history else False
            print(f"Result: {'✅' if success else '❌'}")

            page.close()
            time.sleep(1)

        browser.close()


# ============================================================================
# Main - Run all examples
# ============================================================================

if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Simple", example_simple),
        "2": ("Fluent API", example_fluent),
        "3": ("Form Fill", example_form_fill),
        "4": ("Strategies", example_strategies),
        "5": ("Custom LLM", example_custom_llm),
        "6": ("Reusable", example_reusable),
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning: {name}\n")
            func()
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}: {name}")
        print("\nUsage: python examples.py <number>")
        print("   or: python examples.py 1  # Run example 1")
