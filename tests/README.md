# COSET Pytest Tests

Pytest-compatible test suite for the COSET library, covering quantized vector operations and one-sided vLUT functionality.

## Test Files

### `test_quantized_vectors_pytest.py`
Tests for encoder/decoder with simulated quantized data.

**Test Classes:**
- `TestEncoderDecoderWithSimulatedData` - Zero reconstruction error validation
- `TestQuantizeWrapper` - Quantize wrapper function consistency
- `TestDifferentParameters` - Various lattice/parameter combinations
- `TestSimulatorValidation` - Simulator built-in validation
- `TestPerformance` - Performance benchmarks (marked as slow)

**Parametrized Tests:** Z2, D4, E8 lattices with various q and M values

### `test_one_sided_vlut_pytest.py`
Tests for one-sided vLUT dot product operations.

**Test Classes:**
- `TestOneSidedVLUTConstruction` - vLUT building and structure
- `TestOneSidedVLUTDotProductAccuracy` - Accuracy vs ground truth
- `TestOneSidedVLUTBatchProcessing` - Batch accuracy validation
- `TestOneSidedVLUTCaching` - Caching mechanism tests
- `TestOneSidedVLUTvsTraditional` - Comparison with decode-then-compute
- `TestOneSidedVLUTPerformance` - Performance benchmarks (marked as slow)

## Running Tests

### Run All Tests
```bash
cd /workspace/coset
python3 -m pytest tests/ -v
```

### Run Specific Test File
```bash
python3 -m pytest tests/test_quantized_vectors_pytest.py -v
python3 -m pytest tests/test_one_sided_vlut_pytest.py -v
```

### Run Specific Test Class
```bash
python3 -m pytest tests/test_quantized_vectors_pytest.py::TestEncoderDecoderWithSimulatedData -v
```

### Run Specific Test
```bash
python3 -m pytest tests/test_quantized_vectors_pytest.py::TestEncoderDecoderWithSimulatedData::test_zero_reconstruction_error -v
```

### Run Tests by Marker
```bash
# Skip slow tests
python3 -m pytest tests/ -m "not slow" -v

# Run only slow (performance) tests
python3 -m pytest tests/ -m "slow" -v
```

### Run Tests with Coverage (if pytest-cov installed)
```bash
python3 -m pytest tests/ --cov=coset --cov-report=html --cov-report=term
```

### Run Tests with Specific Output
```bash
# Short traceback
python3 -m pytest tests/ --tb=short

# Line-only traceback
python3 -m pytest tests/ --tb=line

# No traceback
python3 -m pytest tests/ --tb=no

# Show print statements
python3 -m pytest tests/ -s
```

### Run Tests in Parallel (if pytest-xdist installed)
```bash
python3 -m pytest tests/ -n auto
```

## Test Configuration

Configuration is in `pytest.ini` at the project root:

```ini
[pytest]
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
testpaths = tests
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

## Test Markers

- `@pytest.mark.slow` - Performance benchmarks and long-running tests
- `@pytest.mark.integration` - Integration tests (not used yet)
- `@pytest.mark.unit` - Unit tests (not used yet)

## Fixtures

### Quantized Vectors Tests

- `simulator_config` - Provides simulator and simulated vectors for different lattice configurations
  - Parametrized across Z2, D4, E8 lattices
  - Different q and M values

### vLUT Tests

- `vlut_setup` - Provides simulator and vLUT manager
  - Parametrized across Z2, D4, E8 lattices

## Expected Results

### All Non-Slow Tests
```
53 passed in ~60s
```

### Including Slow Tests
```
57 passed in ~90s
```

### Test Coverage

**Quantized Vectors:**
- ✅ Zero reconstruction error (5 configs)
- ✅ Encoding/decoding shapes (5 configs each)
- ✅ Quantize wrapper consistency (5 configs)
- ✅ Various parameter combinations (8 configs)
- ✅ Simulator validation
- ✅ Performance benchmarks (marked slow)

**One-Sided vLUT:**
- ✅ vLUT construction (3 configs)
- ✅ vLUT dtype and size validation
- ✅ Dot product accuracy (5 configs)
- ✅ Batch processing accuracy
- ✅ Caching mechanism (3 tests)
- ✅ Comparison with traditional method
- ✅ Performance benchmarks (marked slow)

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast tests only (for PR checks)
python3 -m pytest tests/ -m "not slow" -v --tb=short

# Full test suite (for main branch)
python3 -m pytest tests/ -v --tb=short --cov=coset --cov-report=xml
```

## Debugging Failed Tests

### View full traceback
```bash
python3 -m pytest tests/ -v --tb=long
```

### Run with pdb debugger
```bash
python3 -m pytest tests/ --pdb
```

### Run last failed tests only
```bash
python3 -m pytest tests/ --lf
```

### Run failed tests first
```bash
python3 -m pytest tests/ --ff
```

## Requirements

- Python 3.7+
- pytest >= 6.0
- torch
- numpy
- coset library

Optional:
- pytest-cov (for coverage reports)
- pytest-xdist (for parallel execution)

## Adding New Tests

1. Create test file: `test_<feature>_pytest.py`
2. Use class-based organization: `class Test<Feature>:`
3. Use fixtures for setup: `@pytest.fixture`
4. Use parametrize for multiple inputs: `@pytest.mark.parametrize`
5. Mark slow tests: `@pytest.mark.slow`
6. Follow naming convention: `test_<what_it_tests>`

Example:
```python
import pytest

@pytest.mark.parametrize("lattice_type,q,M", [
    ("D4", 3, 2),
    ("E8", 4, 2),
])
def test_my_feature(lattice_type, q, M):
    # Test implementation
    assert True
```

---

## License

Same as COSET project license.
