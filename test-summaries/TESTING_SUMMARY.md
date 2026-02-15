# COSET Testing Summary

Overview of testing for the COSET library.

## Overview

The test suite covers quantization operations and the supported API. The main supported API focuses on QAT layers (`create_e8_hnlq_linear`, `HNLQLinear`) for quantization-aware training.

## Test Suites

### Quantized Vector Testing

Validates encoder/decoder with simulated quantized data.

| Lattice | q | M | Zero Error Rate |
|---------|---|---|-----------------|
| Z2 | 3 | 2 | 100% |
| D4 | 3 | 2 | 100% |
| E8 | 3 | 2 | 95% |
| D4 | 4 | 2 | 100% |
| E8 | 4 | 3 | 100% |

### Legacy Tests

Legacy tests in `tests/legacy/` cover deprecated modules (lattices, quant, nn) for backward compatibility. These are not part of the current supported API.

## Running Tests

```bash
pytest tests/ -v -m "not slow"
```

See [tests/README.md](../tests/README.md) for full documentation.
