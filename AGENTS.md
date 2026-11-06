# MTVA implementation standards

- Keep implementations small, direct, readable, deterministic, and easy to audit.
- Add an abstraction only for actual reuse or to isolate meaningful complexity.
- Avoid redundant helpers, defensive exception handling, explanatory inline comments, and decorative naming.
- Keep control flow linear. Catch exceptions only at a boundary that can recover or add actionable context.
- Prefer precise domain names and typed boundaries. Fail with actionable validation errors.
- Keep benchmark behaviour data-driven and plug-compatible. Do not hardcode scenario, tool, language, provider, model, or rubric content; only versioned scoring weights may be fixed in code.
- Keep prompts detailed and explicit; keep runtime code and tests lean.
- Test the essential contract and failure modes without duplicating implementation details.
- Keep documentation short and high-signal.
- Agent personality selection is out of scope.
