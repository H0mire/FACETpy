# AI Agent Playbook

This playbook standardizes how FACETpy contributors work with coding agents.

## Goals

- Reduce prompt ambiguity and context drift.
- Keep processor changes aligned with project rules.
- Make reviews faster and more consistent.

## Standard Workflow

1. Define a narrow task and explicit acceptance criteria.
2. Prepare context with the Projects guidelines.
3. Send prompt using one template from `AI_PROMPT_TEMPLATES.md`.
4. Implement in small steps (prefer incremental diffs).
5. Run local checks (`ruff`, `pytest`, docs build when needed).
6. If Codex is installed, run VSCode task `Review: Uncommitted Changes (Codex)` before manual review.
7. Open PR and complete the agent checklist in the PR template.

## Mandatory Rule

Always include `docs/PROCESSOR_GUIDELINES.md` in agent context for processor-related work.

## Recommended Prompt Inputs

- One-sentence task objective.
- In-scope files and out-of-scope files.
- Functional requirements and edge cases.
- Definition of done (tests/docs/review expectations).
- Repository constraints (`AGENTS.md` when relevant).

## Quality Gates

Run these before opening a PR:

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run pytest
```

For docs changes:

```bash
poetry run sphinx-build -b html docs/source docs/build/html
```

## Optional Team Extensions

- Add domain-specific prompt variants for common processor types.
- Add a lightweight "agent output rubric" (correctness, tests, readability).
- Track recurring agent mistakes and convert them into checklist items.
