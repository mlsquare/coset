# Pytest Conversion Summary

Pytest-based test structure for COSET.

## Test Files

- `tests/legacy/test_quantized_vectors_pytest.py` - Quantized vector operations
- `tests/legacy/test_lattices.py` - Lattice tests
- `tests/legacy/test_nn.py` - Neural network layer tests
- `tests/legacy/test_quant.py` - Quantization function tests

## Running Tests

```bash
# All tests (skip slow)
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=coset --cov-report=html
```

## Markers

- `@pytest.mark.slow` - Performance benchmarks
- `@pytest.mark.integration` - Integration tests

See [tests/README.md](../tests/README.md) for full documentation.
