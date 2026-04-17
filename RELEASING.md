# Releasing FACETpy to PyPI

This repository publishes the package as `facetpy` (import name remains `facet`).

## 1) One-time setup on PyPI

1. In your PyPI account, go to **Account settings -> Publishing**.
2. Add a **pending Trusted Publisher** for project `facetpy`:
   - Owner: `H0mire`
   - Repository: `facetpy`
   - Workflow: `publish.yml`
   - Environment: leave empty unless you use one in GitHub Actions

You do not need to manually "prime" the project with a token upload first.
The first successful trusted publish creates the `facetpy` project on PyPI.

## 2) Local preflight checks

```bash
uv sync --locked
uv run pytest -m "not slow and not requires_data and not requires_c_extension" --tb=short -q
rm -f dist/*
uv build
uv run python -m pip install --upgrade twine
uv run python -m twine check dist/facetpy-*
```

## 3) Create a release tag

1. Bump version in:
   - `pyproject.toml`
   - `src/facet/__init__.py`
2. Commit and push.
3. Tag and push:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Pushing the tag triggers `.github/workflows/publish.yml`, which builds and uploads to PyPI.

## 4) Verify publication

Check:

- https://pypi.org/project/facetpy/
- `pip install facetpy`
