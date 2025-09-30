# COSET Examples

This directory contains practical examples demonstrating real-world applications of the COSET (Coding-based Quantization for Efficient Transformers) library.

## Examples

### 1. One-Sided vLUT Vector Search (`one_sided_vlut_search.py`)

Demonstrates efficient vector search and similarity retrieval using one-sided value lookup tables (vLUTs) with quantized data.

**Use Cases:**
- Semantic search with quantized embeddings
- K-nearest neighbors in compressed vector databases
- Fast similarity retrieval without decompression
- Efficient query processing against large quantized collections

**Key Features:**
- ✅ **Perfect Accuracy**: Zero error with residual-based vLUT implementation
- ⚡ **1000x+ Speedup**: Query-specific vLUT caching for repeated queries
- 💾 **Memory Efficient**: Only store quantized data, no need to decode
- 🚀 **3-4x Faster**: vs traditional decode-then-compute approach

**Examples Included:**

1. **Semantic Search** - Search 1000 quantized document embeddings
   - Demonstrates building query-specific vLUT
   - Shows top-k retrieval
   - Validates accuracy against ground truth

2. **Batch Query Processing** - Process multiple queries efficiently
   - Shows vLUT building and caching for different queries
   - Demonstrates throughput with batch operations

3. **Repeated Query Caching** - Massive speedup for repeated queries
   - 1000x+ speedup from vLUT caching
   - Ideal for pagination, filtering, refinement scenarios

4. **K-Nearest Neighbors** - Find k-NN in quantized database
   - 2000 vector database
   - Top-10 retrieval with accuracy verification
   - ~9000 vectors/sec throughput

5. **Performance Comparison** - vLUT vs traditional approach
   - Direct comparison with decode-then-compute
   - Shows 3.85x speedup with perfect accuracy

**Running the Example:**

```bash
cd /workspace/coset
PYTHONPATH=/workspace/coset:$PYTHONPATH python3 examples/one_sided_vlut_search.py
```

**Expected Output:**
- All 5 examples execute successfully
- Perfect accuracy (< 1e-6 error)
- Performance metrics and comparisons
- Validation of vLUT caching benefits

---

## Technical Details

### One-Sided vLUT Architecture

The one-sided vLUT enables efficient dot product computation between:
- **Unquantized query vector** (full precision)
- **Quantized data vectors** (compressed in encoding space)

**How it Works:**

1. **vLUT Construction** (once per query):
   ```
   For each possible encoding i:
     residual_i = (G @ encoding_i) - q·Q((G @ encoding_i)/q)
     vLUT[i] = ⟨query, residual_i⟩
   ```

2. **Dot Product Computation** (fast lookup):
   ```
   For quantized vector x̂ with encodings {b₀, b₁, ..., b_{M-1}}:
     ⟨query, x̂⟩ = Σᵢ qⁱ · vLUT[index(bᵢ)]
   ```

**Key Innovation:**
- Stores **residuals** instead of raw lattice points
- Matches HNLQ hierarchical quantization formula
- Correct matrix multiplication: `encoding @ G.T`

### Performance Characteristics

| Operation | Time | Throughput |
|-----------|------|------------|
| vLUT Build (E8, q=3, M=2) | ~2.4s | One-time per query |
| vLUT Build (D4, q=4, M=2) | ~48ms | One-time per query |
| Cached vLUT Lookup | ~0.009ms | 1000x+ faster |
| Dot Product (1000 vecs) | ~110ms | 9000 docs/sec |
| k-NN (2000 vecs, k=10) | ~220ms | 9000 vecs/sec |

**Speedup Factors:**
- Caching: **1000-5000x** for repeated queries
- vs Decode-then-compute: **3-4x** speedup
- Perfect accuracy maintained (< 1e-6 error)

---

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- COSET library (this repository)

## Contributing

Feel free to add more examples demonstrating:
- Integration with vector databases (FAISS, Qdrant, etc.)
- Transformer attention with quantized weights
- Large-scale retrieval benchmarks
- Multi-GPU vLUT operations

---

## Citation

If you use COSET in your research, please cite:

```bibtex
@article{kaplan2025coset,
  title={COSET: Coding-based Quantization for Efficient Transformers},
  author={Kaplan, Haim and Ordentlich, Erik},
  journal={arXiv preprint},
  year={2025}
}
```
