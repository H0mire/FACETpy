AI Agents For Processor Development
===================================

This page describes a pragmatic workflow for implementing, refactoring, and reviewing FACETpy processors with AI agents.

Required Prompt Context
-----------------------

When using an AI agent for processor code, always provide
``docs/PROCESSOR_GUIDELINES.md`` in the prompt (or as an attached context file).
Without these guidelines, style and architecture drift is very likely.

Recommended minimum context for each prompt:

- Processor goal in one sentence
- Affected files (for example ``src/facet/correction/...`` and ``tests/...``)
- Acceptance criteria (functional and technical)
- ``docs/PROCESSOR_GUIDELINES.md`` as mandatory rules

Prompt Template (Short)
-----------------------

.. code-block:: text

   Task: [short description]
   Scope: [files/modules]
   Requirements:
   - Follow docs/PROCESSOR_GUIDELINES.md strictly.
   - Write/update matching tests.
   - Keep existing APIs and naming conventions.
   Definition of Done:
   - Tests pass locally.
   - Documentation is updated for behavior changes.

Tips And Practices
------------------

- Work in small, explicit steps instead of large monolithic prompts.
- Request a concrete diff plan before the agent edits multiple files.
- Ask the agent to write or extend tests first, then implement.
- Specify edge cases explicitly (for example missing triggers, invalid parameters, empty data).
- For performance-critical changes, require a short runtime/memory impact rationale.

Review Workflow With Codex
--------------------------

If Codex is installed, run the VS Code task
``Review: Uncommitted Changes (Codex)`` before manual review
(``Tasks: Run Task`` in VS Code).
For branch comparisons, use ``Review: Against Branch (Codex)``.

Useful Additional Context Files
-------------------------------

Depending on the change, these files are often useful extra context:

- ``AGENTS.md`` (repository-specific working rules)
- ``docs/source/development/contributing.rst`` (local workflow and checks)
- ``.vscode/tasks.json`` (standard build/test/review tasks)
- Affected API or user-guide pages under ``docs/source/``

If your team uses agents frequently, maintain a dedicated playbook such as
``AI_AGENT_PLAYBOOK.md`` with team-specific templates and do/don't rules.

Optional Skill Ideas
--------------------

For recurring workflows, define dedicated agent skills, for example:

- ``processor-implementation``: create a new processor and baseline tests following the guidelines.
- ``processor-review-check``: validate diffs against ``docs/PROCESSOR_GUIDELINES.md``.
- ``processor-doc-sync``: verify API/user-guide docs are updated for code changes.

Starter Kit In This Repository
------------------------------

This repository includes a small starter kit for agent workflows:

- ``AI_AGENT_PLAYBOOK.md`` (standardized workflow)
- ``AI_PROMPT_TEMPLATES.md`` (implementation/refactor/review templates)
- ``scripts/agent_context.sh`` (builds a context bundle for prompts)
- ``.github/pull_request_template.md`` (PR checklist including agent checks)
