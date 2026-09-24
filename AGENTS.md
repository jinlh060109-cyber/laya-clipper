# AGENTS.md

## Testing

- Use end-to-end tests as the only way to test. Use them to prove complex features work.
- Never write unit tests after writing the code.
- Run the tests once the implementation is finished, not while you are still building it.
- Do not settle for the simplest scenario that passes. Test a medium-to-hard, realistic scenario.
- Every E2E run must leave behind an artifact that someone can check and reproduce.
- If a part really has to be tested on its own, first write down every way it could fail, then write the code.

## Design

- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Study how established products solve the problem before designing a solution. Adopt their proven patterns and conventions rather than inventing an approach from scratch.

## Dependencies

- Use the dependencies already in the project before writing your own code or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Otherwise, prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
