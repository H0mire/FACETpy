# FACETpy Documentation

This directory contains the Sphinx documentation for FACETpy v2.0.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `build/html/`. Open `build/html/index.html` in your browser.

### Clean Build

To clean previous builds:

```bash
make clean
make html
```

### Build PDF (Optional)

```bash
make latexpdf
```

## Documentation Structure

```
docs/source/
├── index.rst                 # Main index
├── getting_started/          # Getting started guides
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── tutorial.rst
│   └── examples.rst
├── user_guide/               # User guides
│   ├── architecture.rst
│   ├── pipelines.rst
│   ├── processors.rst
│   ├── parallel_processing.rst
│   └── custom_processors.rst
├── api/                      # API reference
│   ├── core.rst
│   ├── io.rst
│   ├── preprocessing.rst
│   ├── correction.rst
│   └── evaluation.rst
├── migration/                # Migration guides
│   ├── v2_migration_guide.rst
│   ├── legacy_api.rst
│   └── ...
└── development/              # Development docs
    ├── contributing.rst
    ├── changelog.rst
    └── roadmap.rst
```

## Documentation Standards

### ReStructuredText (RST)

- Use RST for documentation pages
- Follow [Sphinx RST primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

### Code Examples

All code examples should be:
- Executable (when possible)
- Self-contained
- Include imports
- Show expected output

Example:

```rst
.. code-block:: python

   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline("data.edf", "corrected.edf")
   result = pipeline.run()
   print(f"SNR: {result.context.metadata.custom['metrics']['snr']:.2f}")
```

### Docstrings

Use Google-style docstrings in code:

```python
def my_function(param1, param2):
    """Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Example:
        >>> my_function(1, 2)
        3
    """
    return param1 + param2
```

## Local Development

### Auto-rebuild on Changes

Install sphinx-autobuild:

```bash
pip install sphinx-autobuild
```

Run auto-rebuild server:

```bash
sphinx-autobuild source build/html
```

Navigate to http://127.0.0.1:8000 - the page will auto-reload on changes.

## Read the Docs Integration

This documentation is configured for Read the Docs:

- **Config file**: `.readthedocs.yaml` (in repository root)
- **Requirements**: `docs/requirements.txt`
- **Python version**: 3.8+

The documentation will be automatically built and published when changes are pushed to the repository.

## Contributing to Documentation

1. **Follow the structure** - Place new pages in appropriate directories
2. **Update index.rst** - Add new pages to the table of contents
3. **Test locally** - Always build and check before committing
4. **Check links** - Ensure all cross-references work
5. **Review examples** - Make sure code examples execute correctly

## Troubleshooting

### Import Errors

If you get import errors during build:

```bash
# Make sure FACETpy is installed in development mode
cd ..
pip install -e .

# Then rebuild docs
cd docs
make clean html
```

### Missing Modules

If autodoc can't find modules:

```bash
# Check that sys.path is correct in conf.py
# It should include: sys.path.insert(0, os.path.abspath('../../src/'))
```

### Theme Issues

If the RTD theme doesn't load:

```bash
pip install --upgrade sphinx-rtd-theme
make clean html
```

## Publishing

### To Read the Docs

Documentation is automatically published to Read the Docs when:
- Changes are pushed to main branch
- A new tag/release is created

### Manual Export

To export documentation for offline use:

```bash
make html
cd build/html
zip -r ../../facetpy-docs.zip .
```

## Support

For documentation issues:
- Open an issue on GitHub
- Label it with "documentation"
- Provide details about what's unclear or incorrect
