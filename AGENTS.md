# AI Agent Code Guidelines

## Core Directives

- **Tooling**: `uv` (manager), `ty` (checker), Python 3.11+.
- **Format**: 2-space indent, single quotes, 88 chars, `snake_case`, `PascalCase` classes.
- **Typing**: Strict `ty` checking. Pydantic `BaseModel` for interfaces/schemas. `dataclasses` only for internal perf.
- **Docs**: One-line docstrings. Type hints > text. Comments explain *why*, not *what*.

## Architecture & Patterns

- **Functional Core, Imperative Shell**: Classes for state/lifecycle, pure functions for logic.
- **Protocols**: Duck-typing over inheritance (`typing.Protocol`).
- **Config**: Frozen dataclasses/Pydantic models with factories (`from_env`).
- **Resource Safety**: Context managers (`with`), reference counting, explicit ownership.
- **Data Flow**: Streaming `Iterator[T]`. RORO (Receive Object, Return Object). Pipeline pattern.
- **Errors**: Custom domain exceptions. No broad catches.

## Quality Standards

- **Complexity**: Max 25 lines per method. Break large functions into small, focused helpers.
- **Readability**: Optimize for review. Explicit dependencies over global state. No magic values.
- **Testing**: Test behaviors, not implementation details. High coverage on core logic.
- **Reviewability**: Code should be obvious. If it needs a long comment, refactor it.

## Anti-Patterns

- **Verbosity**: Long docstrings, narrating comments, redundant type info in docs.
- **Vague Naming**: `manager`, `handler`, `util`, `data`. Be specific.
- **Defensive Coding**: Validate at boundaries, trust internals. No "just in case" checks.
- **Deep Nesting**: Use guard clauses and early returns.
- **Over-Abstraction**: No single-method classes or speculative layers.
