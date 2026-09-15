Working on processors with coding agents
========================================

Give the agent a concrete task, the affected paths, relevant edge cases and
acceptance criteria. Include ``AGENTS.md``, :doc:`contributing` and the processor
contract in :doc:`../user_guide/custom_processors`.

Keep validation separate from execution. Preserve units, trigger indices and
context ownership. For a refactor, identify the behavior that must remain stable
and compare it on matching inputs. Tests should check the public behavior and
meaningful failure cases rather than restating the implementation.

Review correctness, API compatibility, test coverage and documentation. Report
findings with a file location, the triggering condition and a concrete remedy.
Use the repository's standard checks:

.. code-block:: console

   uv run ruff check src tests
   uv run ruff format --check src tests
   uv run pytest
   uv run sphinx-build -W -b html docs/source docs/build

Define approval and publication boundaries in the task. A local code review does
not authorize a push, release or remote training run.
