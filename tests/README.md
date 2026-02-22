# FACETpy Test Suite

Comprehensive test suite for FACETpy v2.0.

## Overview

The test suite provides:
- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Fixtures** for test data generation
- **Coverage tracking** to ensure code quality

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and utilities
├── test_core_context.py     # ProcessingContext tests
├── test_core_processor.py   # Processor base class tests
├── test_core_pipeline.py    # Pipeline tests
├── test_preprocessing.py    # Preprocessing processor tests
├── test_correction.py       # Correction processor tests
├── test_evaluation.py       # Evaluation processor tests
└── README.md               # This file
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_core_context.py
```

### Run Specific Test Class

```bash
pytest tests/test_core_context.py::TestProcessingContext
```

### Run Specific Test

```bash
pytest tests/test_core_context.py::TestProcessingContext::test_initialization
```

### Run with Coverage

```bash
pytest --cov=facet --cov-report=html
```

View coverage report in `htmlcov/index.html`

### Run by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only tests that require data
pytest -m requires_data
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests for individual components
- `@pytest.mark.integration` - Integration tests for complete workflows
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.requires_data` - Tests needing external data files
- `@pytest.mark.requires_c_extension` - Tests needing C extension

## Test Fixtures

### Basic Fixtures

#### `sample_raw`
Creates a simple MNE Raw object with random data.

```python
def test_with_raw(sample_raw):
    assert sample_raw.info['sfreq'] == 250
```

#### `sample_raw_with_artifacts`
Creates Raw object with simulated fMRI artifacts.

```python
def test_correction(sample_raw_with_artifacts):
    # Test artifact removal
    pass
```

#### `sample_triggers`
Generates evenly-spaced trigger positions.

```python
def test_triggers(sample_triggers):
    assert len(sample_triggers) == 10
```

#### `sample_context`
Creates a complete ProcessingContext with triggers and metadata.

```python
def test_processor(sample_context):
    processor = MyProcessor()
    result = processor.execute(sample_context)
```

#### `sample_context_with_noise`
Context with estimated noise added.

```python
def test_anc(sample_context_with_noise):
    anc = ANCCorrection()
    result = anc.execute(sample_context_with_noise)
```

#### `sample_edf_file`
Creates a temporary EDF file with annotations.

```python
def test_loader(sample_edf_file):
    loader = Loader(path=str(sample_edf_file))
```

#### `temp_dir`
Provides a temporary directory for file operations.

```python
def test_export(temp_dir):
    output_path = temp_dir / "output.edf"
```

## Writing Tests

### Unit Test Template

```python
import pytest
from facet.core import Processor

@pytest.mark.unit
class TestMyProcessor:
    """Tests for MyProcessor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = MyProcessor(param1=10)
        assert processor.param1 == 10

    def test_execution(self, sample_context):
        """Test processor execution."""
        processor = MyProcessor()
        result = processor.execute(sample_context)

        assert result is not None
        assert result.get_raw() is not None

    def test_validation(self, sample_raw):
        """Test processor validation."""
        processor = MyProcessor()
        context = ProcessingContext(raw=sample_raw)

        with pytest.raises(ProcessorValidationError):
            processor.execute(context)
```

### Integration Test Template

```python
import pytest
from facet.core import Pipeline

@pytest.mark.integration
class TestMyWorkflow:
    """Integration tests for my workflow."""

    def test_complete_workflow(self, sample_edf_file):
        """Test complete processing workflow."""
        pipeline = Pipeline([
            Loader(path=str(sample_edf_file)),
            MyProcessor1(),
            MyProcessor2(),
            MyProcessor3()
        ])

        result = pipeline.run()

        assert result.success is True
        assert result.context is not None
```

## Helper Functions

### `assert_raw_equal(raw1, raw2, rtol=1e-5)`
Assert two Raw objects are approximately equal.

```python
def test_equality(sample_raw):
    raw2 = sample_raw.copy()
    assert_raw_equal(sample_raw, raw2)
```

### `assert_context_valid(context)`
Assert a ProcessingContext is valid.

```python
def test_context(sample_context):
    assert_context_valid(sample_context)
```

### `create_mock_processor(name, process_fn=None)`
Create a mock processor for testing.

```python
def test_pipeline():
    mock = create_mock_processor("mock")
    pipeline = Pipeline([mock])
    result = pipeline.run()
    assert mock.call_count == 1
```

## Test Coverage Goals

Target coverage and reporting will be defined once the initial test suite is executed.

## Continuous Integration

Automated test runs and CI configuration are pending and will be documented after the first successful pipeline execution.

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Install in development mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Missing Dependencies

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock
```

### Slow Tests

Skip slow tests during development:

```bash
pytest -m "not slow"
```

### Test Data

Test data is generated automatically. To use your own data:

```python
@pytest.fixture
def my_data():
    return load_my_test_data()

def test_with_my_data(my_data):
    # Test with your data
    pass
```

## Test Organization

### What to Test

**Unit Tests:**
- Initialization with various parameters
- Basic functionality
- Input validation
- Error handling
- Edge cases
- Boundary conditions

**Integration Tests:**
- Complete workflows
- Processor combinations
- Real data processing
- File I/O operations

### What Not to Test

- External library behavior (MNE, NumPy, etc.)
- Python built-ins
- Simple property getters/setters
- Obvious code paths

## Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** that describe what's being tested
3. **Use fixtures** instead of setup/teardown
4. **Test edge cases** and error conditions
5. **Keep tests independent** - no test should depend on another
6. **Use markers** to organize tests
7. **Mock external dependencies** when appropriate
8. **Test both success and failure paths**

## Running Specific Test Suites

```bash
# Core module tests
pytest tests/test_core_*.py

# Preprocessing tests
pytest tests/test_preprocessing.py

# Correction tests
pytest tests/test_correction.py

# Evaluation tests
pytest tests/test_evaluation.py

# Fast tests only
pytest -m unit

# With verbose output
pytest -v

# With print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

## Coverage Report

Generate detailed coverage report:

```bash
pytest --cov=facet --cov-report=html --cov-report=term-missing
```

View in browser:

```bash
open htmlcov/index.html
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD approach)
2. Record actual coverage results once the suite runs
3. Include both unit and integration tests
4. Add appropriate markers
5. Update this README if adding new fixtures

## Test Data

Test data is automatically generated and cleaned up. Parameters:

- **Sampling frequency:** 250 Hz
- **Number of channels:** 4 EEG channels
- **Duration:** 10 seconds
- **Number of triggers:** 10
- **Artifact length:** 50 samples

Customize in `tests/conftest.py`.

## Questions?

See the [main documentation](../docs/) or open an issue on GitHub.
