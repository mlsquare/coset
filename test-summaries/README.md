# Test Documentation Archive

Documentation for the COSET test suite.

## Contents

### Test Summaries

1. **[TESTING_SUMMARY](TESTING_SUMMARY.html)** - Testing overview
2. **[PYTEST_CONVERSION_SUMMARY](PYTEST_CONVERSION_SUMMARY.html)** - Pytest structure

## Quick Reference

### Test Structure

```
tests/
└── legacy/
    ├── test_lattices.py
    ├── test_nn.py
    ├── test_quant.py
    └── test_quantized_vectors_pytest.py
```

### Running Tests

```bash
pytest tests/ -v -m "not slow"
```

### Specific Tests

```bash
# Quantized vectors only
pytest tests/legacy/test_quantized_vectors_pytest.py -v

# Performance benchmarks
pytest tests/ -v -m "slow"
```

See [tests/README.md](../tests/README.md) for complete usage.

## Current Focus

The supported API focuses on:
- **QAT Layers** - `create_e8_hnlq_linear`, `HNLQLinear`
- **E8 and D4** lattice quantization
- **Quantized vectors** - Encoding/decoding validation

Legacy tests cover deprecated modules for backward compatibility.
