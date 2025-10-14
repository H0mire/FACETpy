Contributing to FACETpy
=======================

Thank you for your interest in contributing to FACETpy! This guide will help you
get started with contributing to the project.

Ways to Contribute
------------------

There are many ways to contribute:

- 🐛 **Report bugs** - Help us identify and fix issues
- 💡 **Suggest features** - Propose new functionality
- 📝 **Improve documentation** - Fix typos, add examples
- 🧪 **Write tests** - Increase test coverage
- 🔧 **Fix bugs** - Submit bug fixes
- ✨ **Add features** - Implement new processors or capabilities
- 🎨 **Improve code** - Refactor and optimize

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository**

   Visit https://github.com/your-org/facetpy and click "Fork"

2. **Clone your fork**

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/facetpy.git
      cd facetpy

3. **Set up development environment**

   .. code-block:: bash

      # Create virtual environment
      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # Install in development mode with dev dependencies
      pip install -e ".[dev]"

4. **Install pre-commit hooks**

   .. code-block:: bash

      pre-commit install

5. **Verify installation**

   .. code-block:: bash

      # Run tests
      pytest

      # Check code style
      flake8 src/facet

      # Build docs
      cd docs && make html

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a branch**

   .. code-block:: bash

      git checkout -b feature/my-new-feature
      # or
      git checkout -b fix/bug-description

2. **Make your changes**

   Write code, tests, and documentation

3. **Run tests**

   .. code-block:: bash

      pytest tests/

4. **Check code style**

   .. code-block:: bash

      flake8 src/facet
      black src/facet tests/
      isort src/facet tests/

5. **Commit changes**

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description"

6. **Push to your fork**

   .. code-block:: bash

      git push origin feature/my-new-feature

7. **Create Pull Request**

   Visit GitHub and create a PR from your branch

Code Guidelines
---------------

Style Guide
~~~~~~~~~~~

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Prefer double quotes for strings
- **Imports**: Use absolute imports
- **Type hints**: Required for all public functions

.. code-block:: python

   # ✓ Good
   from facet.core import Processor, ProcessingContext

   def process_data(raw: mne.io.Raw, param: float = 1.0) -> np.ndarray:
       """Process the data."""
       return raw.get_data(copy=False) * param

   # ✗ Bad
   from facet.core import *  # Don't use star imports

   def process_data(raw, param=1.0):  # Missing type hints
       return raw.get_data(copy=False) * param

Naming Conventions
~~~~~~~~~~~~~~~~~~

- **Classes**: PascalCase (e.g., ``MyProcessor``)
- **Functions**: snake_case (e.g., ``process_data``)
- **Constants**: UPPER_CASE (e.g., ``MAX_WINDOW_SIZE``)
- **Private**: Prefix with ``_`` (e.g., ``_internal_method``)

Documentation
~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def my_function(param1: str, param2: int = 10) -> bool:
       """
       One-line summary.

       Longer description explaining what the function does,
       when to use it, and any important notes.

       Parameters
       ----------
       param1 : str
           Description of param1
       param2 : int, optional
           Description of param2 (default: 10)

       Returns
       -------
       bool
           Description of return value

       Raises
       ------
       ValueError
           When param2 is negative

       Examples
       --------
       >>> result = my_function("test", param2=5)
       True

       See Also
       --------
       related_function : Related functionality

       Notes
       -----
       Any implementation notes or warnings.

       References
       ----------
       .. [1] Author et al. "Paper Title", Journal, 2024.
       """

Testing
-------

Writing Tests
~~~~~~~~~~~~~

All new code must include tests:

.. code-block:: python

   # tests/test_my_feature.py
   import pytest
   from facet.core import ProcessingContext

   @pytest.mark.unit
   class TestMyProcessor:
       """Tests for MyProcessor."""

       def test_initialization(self):
           """Test processor initialization."""
           processor = MyProcessor(param=10)
           assert processor.param == 10

       def test_process(self, sample_context):
           """Test processing."""
           processor = MyProcessor(param=10)
           result = processor.execute(sample_context)

           assert result is not None
           assert result.get_raw() is not None

       def test_validation_failure(self):
           """Test validation fails when required data missing."""
           processor = MyProcessor(param=10)
           context = ProcessingContext()  # Empty

           with pytest.raises(ProcessorValidationError):
               processor.execute(context)

Test Coverage
~~~~~~~~~~~~~

Set coverage expectations once the initial suite lands. In the meantime, capture coverage data when running locally:

.. code-block:: bash

   # Run with coverage
   pytest --cov=facet --cov-report=html

   # View coverage report
   open htmlcov/index.html

Test Markers
~~~~~~~~~~~~

Use markers to organize tests:

.. code-block:: python

   @pytest.mark.unit  # Fast unit tests
   @pytest.mark.integration  # Integration tests
   @pytest.mark.slow  # Slow tests
   @pytest.mark.requires_data  # Needs external data
   @pytest.mark.requires_c_extension  # Needs C extension

Run specific tests:

.. code-block:: bash

   pytest -m unit  # Run only unit tests
   pytest -m "not slow"  # Skip slow tests

Pull Request Process
--------------------

Before Submitting
~~~~~~~~~~~~~~~~~

1. **Run all tests**

   .. code-block:: bash

      pytest

2. **Check code style**

   .. code-block:: bash

      flake8 src/facet
      black --check src/facet tests/
      isort --check src/facet tests/

3. **Update documentation**

   Add or update docs for new features

4. **Add changelog entry**

   Add entry to ``CHANGELOG.md``

5. **Build docs locally**

   .. code-block:: bash

      cd docs && make html

PR Guidelines
~~~~~~~~~~~~~

**Title Format:**

- ``feat: Add new processor for X``
- ``fix: Correct bug in Y``
- ``docs: Update documentation for Z``
- ``test: Add tests for W``
- ``refactor: Improve performance of V``

**Description Should Include:**

- What changes were made
- Why the changes are needed
- How to test the changes
- Related issues (if any)

**Example:**

.. code-block:: text

   ## Summary
   Adds a new `CustomFilter` processor for applying custom filters.

   ## Motivation
   Users requested ability to apply custom filter designs (#123).

   ## Changes
   - Added `CustomFilter` class in `facet/preprocessing/filtering.py`
   - Added tests in `tests/test_preprocessing.py`
   - Updated documentation

   ## Testing
   - All existing tests pass
   - Added 5 new tests for `CustomFilter`
   - Manually tested with example data

   ## Closes
   #123

Review Process
~~~~~~~~~~~~~~

1. **Automated checks** run (tests, style, coverage)
2. **Maintainer reviews** code
3. **Feedback addressed** (if needed)
4. **PR merged** after approval

Code Review Checklist
~~~~~~~~~~~~~~~~~~~~~

Reviewers check for:

- ✓ Tests pass
- ✓ Code follows style guide
- ✓ Documentation updated
- ✓ Changelog updated
- ✓ No breaking changes (or properly documented)
- ✓ Performance impact considered

Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, include:

- **FACETpy version**: ``facet.__version__``
- **Python version**: ``python --version``
- **Operating system**: Windows/Mac/Linux
- **MNE version**: ``mne.__version__``
- **Minimal example** to reproduce
- **Expected vs actual behavior**
- **Error messages** (full traceback)

**Example:**

.. code-block:: text

   **Bug Description**
   AASCorrection fails with IndexError for short recordings

   **Environment**
   - FACETpy: 2.0.0
   - Python: 3.9.5
   - OS: Ubuntu 20.04
   - MNE: 1.5.0

   **Minimal Example**
   ```python
   from facet.correction import AASCorrection
   # ... code to reproduce
   ```

   **Error**
   ```
   IndexError: index 100 is out of bounds for axis 1 with size 50
   ```

   **Expected**
   Should handle short recordings gracefully or raise clear error

Feature Requests
~~~~~~~~~~~~~~~~

When requesting features, describe:

- **Use case**: What problem does it solve?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered?
- **Impact**: Who benefits from this feature?

Development Guidelines
----------------------

Adding a New Processor
~~~~~~~~~~~~~~~~~~~~~~

1. **Create processor class**

   .. code-block:: python

      # src/facet/correction/my_correction.py
      from facet.core import Processor, register_processor

      @register_processor
      class MyCorrection(Processor):
          name = "my_correction"
          description = "My correction algorithm"

          def __init__(self, param1, param2=10):
              self.param1 = param1
              self.param2 = param2
              super().__init__()

          def process(self, context):
              # Implementation
              pass

2. **Add tests**

   .. code-block:: python

      # tests/test_my_correction.py
      class TestMyCorrection:
          def test_initialization(self):
              # Tests...
              pass

3. **Update exports**

   .. code-block:: python

      # src/facet/correction/__init__.py
      from .my_correction import MyCorrection

      __all__ = ['MyCorrection']

4. **Add documentation**

   Update relevant docs and add API reference

Adding Dependencies
~~~~~~~~~~~~~~~~~~~

1. **Add to setup.py**

   .. code-block:: python

      install_requires=[
          'numpy>=1.20.0',
          'mne>=1.5.0',
          'new-package>=1.0.0'
      ]

2. **Document requirement**

   Add to ``docs/source/getting_started/installation.rst``

3. **Test compatibility**

   Test with minimum and latest versions

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

When optimizing code:

1. **Profile first**

   .. code-block:: python

      import cProfile
      cProfile.run('processor.execute(context)')

2. **Benchmark**

   Add benchmark tests:

   .. code-block:: python

      def test_performance(benchmark, sample_context):
          processor = MyProcessor()
          benchmark(processor.execute, sample_context)

3. **Document improvements**

   Note performance gains in PR

Community
---------

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and discussions
- **Email**: support@facetpy.org

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards others

Recognition
-----------

Contributors are recognized in:

- ``AUTHORS.md`` file
- Release notes
- Documentation credits

Thank you for contributing to FACETpy! 🎉
