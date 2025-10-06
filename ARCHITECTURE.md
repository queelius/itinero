# itinero Architecture

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER CODE                                  │
│                                                                     │
│  from itinero import automate, agent, fill_form, strategies       │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       PUBLIC API LAYER                              │
│                        (itinero.py)                                 │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  automate()  │  │   agent()    │  │ fill_form()  │            │
│  │              │  │              │  │              │            │
│  │ One-liner    │  │ Fluent       │  │ Specialized  │            │
│  │ for simple   │  │ builder      │  │ for forms    │            │
│  │ tasks        │  │ for custom   │  │              │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────┐          │
│  │           strategies namespace                       │          │
│  │                                                       │          │
│  │  chain()  llm()  heuristic()  fallback()            │          │
│  └─────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       CORE DOMAIN LAYER                             │
│                          (core.py)                                  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │                    Protocols                            │        │
│  │                                                         │        │
│  │  LLM          Executor      PromptBuilder              │        │
│  │  ┌─────┐     ┌─────┐      ┌─────┐                    │        │
│  │  │gen()│     │exec()│      │build│                    │        │
│  │  └─────┘     └─────┘      └─────┘                    │        │
│  │                                                         │        │
│  │  ActionParser                                          │        │
│  │  ┌─────┐                                               │        │
│  │  │parse│                                               │        │
│  │  └─────┘                                               │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │                 Value Objects                           │        │
│  │                                                         │        │
│  │  Action          Context         ExecutionResult       │        │
│  │  ┌────────┐     ┌────────┐      ┌────────┐           │        │
│  │  │ type   │     │ goal   │      │success │           │        │
│  │  │selector│     │ state  │      │ error  │           │        │
│  │  │ value  │     │history │      │metadata│           │        │
│  │  └────────┘     └────────┘      └────────┘           │        │
│  │  (immutable)   (immutable)     (immutable)           │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │                   Strategies                            │        │
│  │                                                         │        │
│  │  Strategy (ABC)                                        │        │
│  │       ↑                                                 │        │
│  │       ├─── LLMStrategy                                 │        │
│  │       ├─── RetryStrategy (decorator)                   │        │
│  │       └─── RecoveryStrategy (decorator)                │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │                     Agent                               │        │
│  │                                                         │        │
│  │  Agent(strategy, executor)                             │        │
│  │    - run(page, goal) → Context                         │        │
│  │    - Minimal: just orchestrates                        │        │
│  └────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       ADAPTER LAYER                                 │
│                      (adapters.py)                                  │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  OllamaLLM   │  │ CallableLLM  │  │Playwright    │            │
│  │              │  │              │  │Executor      │            │
│  │ Implements   │  │ Wraps any    │  │              │            │
│  │ LLM protocol │  │ callable     │  │ Implements   │            │
│  │              │  │              │  │ Executor     │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                               │
│  │ModelSpecific │  │JSON Action   │                               │
│  │PromptBuilder │  │Parser        │                               │
│  │              │  │              │                               │
│  │ Implements   │  │ Implements   │                               │
│  │ PromptBuilder│  │ ActionParser │                               │
│  └──────────────┘  └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                             │
│                   (existing, unchanged)                             │
│                                                                     │
│  state.py          executor.py         model_prompts.py            │
│  ┌────────┐       ┌────────┐          ┌────────┐                  │
│  │ State  │       │execute │          │ MODEL_ │                  │
│  │ Failed │       │ action │          │ CONFIGS│                  │
│  │ Action │       │        │          │        │                  │
│  └────────┘       └────────┘          └────────┘                  │
│                                                                     │
│  recovery.py                                                        │
│  ┌────────────────────────────────┐                                │
│  │ default_recovery()             │                                │
│  │ aggressive_recovery()          │                                │
│  │ conservative_recovery()        │                                │
│  └────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent Composition                      │
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │              Agent                             │        │
│  │                                                │        │
│  │  ┌──────────────┐       ┌──────────────┐     │        │
│  │  │   Strategy   │◄─────►│   Executor   │     │        │
│  │  └──────────────┘       └──────────────┘     │        │
│  │         ▲                                     │        │
│  │         │                                     │        │
│  │         │ next_action()                      │        │
│  │         │                                     │        │
│  │         ▼                                     │        │
│  │  ┌──────────────┐                            │        │
│  │  │   Context    │                            │        │
│  │  │              │                            │        │
│  │  │ - goal       │                            │        │
│  │  │ - page_state │                            │        │
│  │  │ - history    │                            │        │
│  │  └──────────────┘                            │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Strategy Composition Pattern

```
┌─────────────────────────────────────────────────────────────┐
│               Decorator Pattern for Strategies              │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │        RecoveryStrategy                      │          │
│  │  ┌────────────────────────────────┐         │          │
│  │  │                                │         │          │
│  │  │     RetryStrategy              │         │          │
│  │  │  ┌──────────────────────┐     │         │          │
│  │  │  │                      │     │         │          │
│  │  │  │   LLMStrategy        │     │         │          │
│  │  │  │  ┌────────────┐     │     │         │          │
│  │  │  │  │    LLM     │     │     │         │          │
│  │  │  │  │  Prompt    │     │     │         │          │
│  │  │  │  │  Parser    │     │     │         │          │
│  │  │  │  └────────────┘     │     │         │          │
│  │  │  └──────────────────────┘     │         │          │
│  │  └────────────────────────────────┘         │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  Each layer wraps the next, adding behavior                │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Automation Loop                           │
│                                                             │
│  1. Extract page state                                     │
│     ┌─────────┐                                            │
│     │ Page    │──────► extract_dom_elements()              │
│     └─────────┘              │                             │
│                              ▼                             │
│                       ┌──────────────┐                     │
│                       │  page_state  │                     │
│                       └──────────────┘                     │
│                              │                             │
│  2. Build context            ▼                             │
│                       ┌──────────────┐                     │
│                       │   Context    │                     │
│                       │              │                     │
│                       │ goal + state │                     │
│                       │  + history   │                     │
│                       └──────────────┘                     │
│                              │                             │
│  3. Decide action            ▼                             │
│                       Strategy.next_action()               │
│                              │                             │
│                              ▼                             │
│                       ┌──────────────┐                     │
│                       │   Action     │                     │
│                       │              │                     │
│                       │ type+selector│                     │
│                       │    +value    │                     │
│                       └──────────────┘                     │
│                              │                             │
│  4. Execute                  ▼                             │
│                       Executor.execute()                   │
│                              │                             │
│                              ▼                             │
│                       ┌──────────────┐                     │
│                       │   Result     │                     │
│                       │              │                     │
│                       │success+error │                     │
│                       └──────────────┘                     │
│                              │                             │
│  5. Update context           ▼                             │
│                       new Context(                         │
│                         history + [result]                 │
│                       )                                    │
│                              │                             │
│  6. Repeat or Complete       ▼                             │
│                       Check if done                        │
└─────────────────────────────────────────────────────────────┘
```

## LLM Strategy Internals

```
┌─────────────────────────────────────────────────────────────┐
│              LLMStrategy Decision Flow                      │
│                                                             │
│  Context                                                    │
│     │                                                       │
│     ▼                                                       │
│  PromptBuilder.build()                                     │
│     │                                                       │
│     ▼                                                       │
│  "GOAL: Fill form                                          │
│   ELEMENTS: input#email:EMPTY:REQ                          │
│   JSON: {type:fill,selector:#email,value:...}"            │
│     │                                                       │
│     ▼                                                       │
│  LLM.generate()                                            │
│     │                                                       │
│     ▼                                                       │
│  "{\"type\":\"fill\",\"selector\":\"#email\",              │
│    \"value\":\"user@example.com\"}"                        │
│     │                                                       │
│     ▼                                                       │
│  ActionParser.parse()                                      │
│     │                                                       │
│     ▼                                                       │
│  {"type": "fill",                                          │
│   "selector": "#email",                                    │
│   "value": "user@example.com"}                             │
│     │                                                       │
│     ▼                                                       │
│  Action(type="fill",                                       │
│         selector="#email",                                 │
│         value="user@example.com")                          │
└─────────────────────────────────────────────────────────────┘
```

## Builder Pattern Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Fluent Builder Construction                    │
│                                                             │
│  agent()                                                    │
│     │                                                       │
│     ▼                                                       │
│  FluentAgentBuilder()                                      │
│     │                                                       │
│     ▼                                                       │
│  .model("gemma3n:e2b")                                     │
│     │ ─────► stores model config                           │
│     ▼                                                       │
│  .with_retry(max_attempts=3)                               │
│     │ ─────► stores retry config                           │
│     ▼                                                       │
│  .with_recovery()                                          │
│     │ ─────► sets recovery flag                            │
│     ▼                                                       │
│  .verbose()                                                │
│     │ ─────► sets verbose flag                             │
│     ▼                                                       │
│  .build()                                                  │
│     │                                                       │
│     ├─────► Create LLM (OllamaLLM)                         │
│     ├─────► Create PromptBuilder                           │
│     ├─────► Create Parser                                  │
│     ├─────► Create LLMStrategy                             │
│     ├─────► Wrap with RetryStrategy                        │
│     ├─────► Wrap with RecoveryStrategy                     │
│     ├─────► Create Executor                                │
│     ├─────► Create Agent                                   │
│     └─────► Wrap with VerboseAgent if verbose              │
│                │                                            │
│                ▼                                            │
│             Agent (ready to use)                           │
└─────────────────────────────────────────────────────────────┘
```

## Extension Points

```
┌─────────────────────────────────────────────────────────────┐
│                  How to Extend itinero                      │
│                                                             │
│  1. Custom Strategy                                        │
│     ┌─────────────────────────────────┐                    │
│     │  class MyStrategy(Strategy):    │                    │
│     │    def next_action(context):    │                    │
│     │      # your logic               │                    │
│     │      return Action(...)         │                    │
│     └─────────────────────────────────┘                    │
│                                                             │
│  2. Custom LLM                                             │
│     ┌─────────────────────────────────┐                    │
│     │  class MyLLM:                   │                    │
│     │    def generate(prompt):        │                    │
│     │      # call your LLM            │                    │
│     │      return response            │                    │
│     └─────────────────────────────────┘                    │
│                                                             │
│  3. Custom Executor                                        │
│     ┌─────────────────────────────────┐                    │
│     │  class MyExecutor:              │                    │
│     │    def execute(action_str):     │                    │
│     │      # your execution           │                    │
│     │      return ExecutionResult()   │                    │
│     └─────────────────────────────────┘                    │
│                                                             │
│  4. Custom Prompt Builder                                  │
│     ┌─────────────────────────────────┐                    │
│     │  class MyPromptBuilder:         │                    │
│     │    def build(context):          │                    │
│     │      # generate prompt          │                    │
│     │      return prompt_string       │                    │
│     └─────────────────────────────────┘                    │
│                                                             │
│  5. Strategy Decorator                                     │
│     ┌─────────────────────────────────┐                    │
│     │  class LoggingStrategy(Strategy)│                    │
│     │    def __init__(base):          │                    │
│     │      self.base = base           │                    │
│     │    def next_action(context):    │                    │
│     │      log("deciding...")         │                    │
│     │      return self.base.next...() │                    │
│     └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Patterns Used

1. **Protocol Pattern** - Define interfaces without inheritance
2. **Strategy Pattern** - Pluggable algorithms for decision-making
3. **Decorator Pattern** - Wrap strategies to add behavior
4. **Builder Pattern** - Fluent construction of complex objects
5. **Adapter Pattern** - Connect domain to external dependencies
6. **Ports & Adapters** - Hexagonal architecture for clean separation

## Dependencies

```
itinero.py
   │
   ├─► core.py (no external dependencies)
   │      └─► typing, abc, dataclasses (stdlib only)
   │
   ├─► adapters.py
   │      ├─► core.py
   │      ├─► model_prompts.py
   │      ├─► executor.py (existing)
   │      └─► requests (for Ollama)
   │
   └─► recovery.py
          └─► core.py

Core has ZERO external dependencies - pure domain logic
Adapters handle all external integrations
```

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Testing Layers                           │
│                                                             │
│  Unit Tests                                                │
│  ┌─────────────────────────────────┐                       │
│  │ - Test each Strategy in isolation│                       │
│  │ - Mock LLM, Executor            │                       │
│  │ - Test value objects            │                       │
│  │ - Test builder                  │                       │
│  └─────────────────────────────────┘                       │
│                                                             │
│  Integration Tests                                         │
│  ┌─────────────────────────────────┐                       │
│  │ - Test Strategy composition     │                       │
│  │ - Test with real adapters       │                       │
│  │ - Test end-to-end flows         │                       │
│  └─────────────────────────────────┘                       │
│                                                             │
│  E2E Tests                                                 │
│  ┌─────────────────────────────────┐                       │
│  │ - Test with real browser        │                       │
│  │ - Test with real LLM            │                       │
│  │ - Examples as tests             │                       │
│  └─────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

- **LLM Call**: ~500ms (depends on model)
- **Action Execution**: ~50-200ms (depends on page)
- **State Extraction**: ~10-50ms
- **Prompt Building**: ~1-5ms (cached after first)
- **Response Parsing**: ~1ms

**Total per action**: ~600ms average

**Memory**: Low - immutable objects, no leaks

**Scalability**: Can run multiple agents in parallel (thread-safe)
