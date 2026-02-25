Contributing to FACETpy
=======================

Thanks for contributing. This guide documents the current development workflow.

Quick Setup
-----------

1. Fork ``https://github.com/H0mire/facetpy``.
2. Clone your fork.

.. code-block:: bash

   git clone https://github.com/<your-username>/facetpy.git
   cd facetpy

3. Create a virtual environment and install all dev dependencies.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"

4. (Optional) Also install docs dependencies when working on documentation.

.. code-block:: bash

   pip install -e ".[docs]"

5. Verify your environment.

.. code-block:: bash

   pytest -m "not slow"

.. note::

   **Poetry users** — ``poetry install`` continues to work as before.
   Use ``poetry run pytest`` etc. as you normally would.

Development Workflow
--------------------

1. Create a topic branch.

.. code-block:: bash

   git checkout -b codex/<short-topic>

2. Implement the change in ``src/facet``.
3. Add or update tests in ``tests``.
4. Update docs in ``docs/source`` when behavior or API changes.
5. Run checks locally.

.. code-block:: bash

   pytest

6. Build docs locally when docs changed.

.. code-block:: bash

   sphinx-build -b html docs/source docs/build

Code Style
----------

- Follow PEP 8 with four-space indentation.
- Use descriptive ``snake_case`` names for functions/variables/modules.
- Use ``PascalCase`` for classes and ``UPPER_CASE`` for constants.
- Prefer explicit imports from ``facet`` modules.
- Keep public APIs type hinted.
- Use ``loguru`` for diagnostic logging.

Linting and Formatting
----------------------

Ruff is the active linter configuration for this repository.

.. code-block:: bash

   ruff check src tests

(Optional auto-fix)

.. code-block:: bash

   ruff check --fix src tests

Testing
-------

- Place tests under ``tests/<module>/test_<feature>.py``.
- Use markers for heavy scenarios:

  - ``@pytest.mark.slow``
  - ``@pytest.mark.requires_data``
  - ``@pytest.mark.requires_c_extension``

- Use deterministic fixtures and sample data from ``examples`` when possible.

Run tests:

.. code-block:: bash

   pytest

Run a subset:

.. code-block:: bash

   pytest -m "not slow"

Documentation Style
-------------------

Use NumPy-style docstrings for public code.

.. code-block:: python

   def my_function(param1: str, param2: int = 10) -> bool:
       """One-line summary.

       Parameters
       ----------
       param1 : str
           Description of param1.
       param2 : int, optional
           Description of param2.

       Returns
       -------
       bool
           Description of return value.
       """

Pull Requests
-------------

Before opening a PR:

1. Ensure tests pass locally.
2. Ensure documentation is updated for user-facing changes.
3. Include a concise description of what changed and why.
4. Link related issues when applicable.
5. Mention skipped test markers (if any) with a reason.

Commit messages in this repository typically start with lowercase topic tags,
for example:

- ``docs, refactor: clarify pipeline flow``
- ``fix: handle missing trigger channel``
