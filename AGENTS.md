# Repository Guidelines

## Project Structure & Module Organization
The core library lives in `src/facet`, organized by domain (`io`, `correction`, `evaluation`, `preprocessing`, `helpers`). Shared data assets sit under `src/facet/resources`. Automated tests reside in `tests`, mirroring the package layout for quick cross-reference. Documentation sources are under `docs`, with rendered coverage reports in `htmlcov`. Example datasets and walkthrough notebooks are kept in `examples`, `example_simple_bids`, and `example_full_bids` for realistic pipelines.

## Build, Test, and Development Commands
- `poetry install` – set up the virtual environment and install all runtime and dev dependencies.
- `poetry run pytest` – execute the test suite (adds coverage data via the config in `pytest.ini`).
- `poetry run build-fastranc` – compile the C extension used in artifact correction.
- `poetry run sphinx-build -b html docs/source docs/build` – build the documentation locally for spot-checking narrative changes.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive, snake_case naming for functions, modules, and variables. Classes use PascalCase, and constants stay in ALL_CAPS. Prefer explicit imports from `facet` modules so call sites stay readable. Log diagnostic output with `loguru` for consistency. Run `poetry run pytest` before committing so we catch regressions and gather up-to-date coverage details.

## Testing Guidelines
Use `pytest` for all new tests, placing them in `tests/<module>/test_<feature>.py`. Mark slow or data-dependent scenarios with `@pytest.mark.slow` or `@pytest.mark.requires_data` so they can be gated in CI. Track coverage growth for new logic once baseline numbers exist (inspect `htmlcov/index.html` after a run). Favor fixtures that reuse BIDS samples from `examples` to keep setups deterministic.

## Commit & Pull Request Guidelines
Commits typically start with lowercase topic tags separated by commas, e.g., `docs, refactor: clarify pipeline flow`. Keep messages concise and focused on a single change set. For pull requests, link related issues, outline algorithmic or API changes, and include before/after outputs or screenshots when behavior shifts. Confirm the latest test run in the PR description and mention any skipped markers with rationale.

## Documentation & Examples
When updating guides, edit `docs/source` and regenerate HTML to confirm links and images. Keep notebook-driven tutorials in `examples` lightweight by preferring smaller bundled datasets; reference larger files from `example_full_bids` only when necessary to illustrate complete workflows.
