# CUDA Kernels for Hierarchical Nested Lattice Quantizer (HNLQ)

This document provides implementation-focused instructions for developers to build CUDA kernels and supporting infrastructure for:

1. **Encoding** a vector into hierarchical nested lattice quantizer (HNLQ) indices.
2. **Decoding** indices back to quantized vectors or digits.
3. **VLUT-based compute** (matrix multiply/accumulate) in the quantized index space.

---

## 0. Scope & Definitions

1. **Quantizer**: hierarchical nested lattice quantizer with parameters:
   - radix \(Q \ge 2\),
   - depth \(M \ge 1\),
   - dimension \(D \ge 1\) (contracting axis).

   At each hierarchy level \(m \in \{1,\dots,M\}\), a digit \(d_m \in \{0,\dots,Q-1\}^D\) is produced.

2. **Indexing/packing**: all \(M \times D\) base-\(Q\) digits are packed to integers used as indices for lookup tables. Several formats are supported (see §3).

3. **Contracting axis**: last dimension of size \(D\). Inputs/outputs have shape `[..., D]` for encoder and `[...]` for indices.

4. **VLUT compute**: matrix multiply-accumulate using lookup tables:
   - **One-sided**: one operand encoded, one dense.
   - **Two-sided**: both operands encoded.

---

## 1. Required Inputs & Precomputation

- **Runtime tensor inputs**:
  - Encoder input `X`: `[..., D]` (FP32/FP16/BF16).

- **Quantizer parameters**:
  - Integers: `Q`, `M`, `D`.

- **Lattice state** (device-resident):
  - Nearest-plane factors (e.g., triangular `R`).
  - Transform matrices `T_to_lat`, `T_from_lat`.
  - Per-level coset/modulo metadata.

- **Packing meta**:
  - Pack mode, limb counts, bit widths per digit, endianness.

- **VLUT tables**:
  - One-sided LUT: index → vector contribution.
  - Two-sided LUT: (index_left, index_right) → scalar/block contribution.

---

## 2. Encoder

### 2.1 Functional Contract

Input: `X[..., D]`  
Output: packed indices `I[...]`

Steps:
1. Transform vector into lattice coordinates.
2. For each level \(m = 1 \dots M\):
   - Compute digit vector \(d_m \in [0, Q-1]^D\).
   - Update residual for next level.
3. Pack all digits into indices per selected format.

### 2.2 Numerical Steps
1. \(u \leftarrow T_{\text{to\_lat}} \cdot x\).
2. Apply hierarchical modulo/coset operations to get digits.
3. Pack digits (see §3).

### 2.3 Parallelization
- One CUDA thread per vector along contracting axis.
- Grid covers all leading batch elements.
- Coalesced loads for `X[..., D]`.

### 2.4 Rounding
- Deterministic rounding (round-to-nearest, ties-to-even).

---

## 3. Index Pack Formats

### 3.1 LEVEL_PACKED
- Output **M integers**, each encoding \(D\) digits:
  \[
  \text{level\_index}_m = \sum_{j=0}^{D-1} d_m[j] \cdot Q^j
  \]
- Range: \([0, Q^D-1]\).  
- Shape: `[..., M]`.

### 3.2 FULL_PACKED_64
- Output **single 64-bit integer** if \(Q^{M\cdot D} \le 2^{64}\).  
- Packing order: per-level, per-digit.

### 3.3 MULTIWORD
- Output multiple 32-bit words (limbs).  
- Shape: `[..., L]`.

**Validation**: select pack mode based on representable range.

---

## 4. Decoder

Goal: reconstruct either
- digit vectors \(\{d_m\}\), or
- de-quantized approximation \(\hat{x}\).

Steps:
1. Unpack digits from index.
2. Hierarchically reconstruct lattice coordinate.
3. Apply inverse transform: \( \hat{x} = T_{\text{from\_lat}} \cdot \hat{u} \).

Parallelization: one thread per vector.

---

## 5. VLUT Compute Primitives

### 5.1 One-Sided VLUT
- Input: encoded indices (rows), dense FP matrix.  
- LUT maps index → contribution vector.  
- Multiply and accumulate in FP32.

### 5.2 Two-Sided VLUT
- Input: encoded indices for both sides.  
- LUT maps (left_index, right_index) → scalar/block.  
- Multiply and accumulate in FP32.

---

## 6. VLUT Table Specifications

- **One-sided LUT**:
  - Key: packed index.
  - Value: scalar/vector/tile contribution.
  - Layout: contiguous, 128-bit aligned.

- **Two-sided LUT**:
  - Key: pair of indices.
  - Value: scalar/block.
  - Storage: 2D table or blocked + hashed.

- **Precision**: FP16/BF16 storage, FP32 accumulation.

---

## 7. Kernel Design

### 7.1 Encoder
- Grid: `num_vecs = prod(leading_dims)`.
- Each thread:
  - Load `D` scalars.
  - Transform, compute digits, pack.
  - Write index.

### 7.2 Decoder
- Inverse of encoder.

### 7.3 One-Sided VLUT MAC
- Tile dense operand into shared memory.
- Fetch LUT entries per index.
- FMA accumulate.

### 7.4 Two-Sided VLUT MAC
- Load both indices.
- Form key.
- Fetch LUT entry.
- FMA accumulate.

---

## 8. Memory & Alignment

- Align arrays to 128-bit.  
- Prefer structure-of-arrays for indices.  
- For multiword, contiguous limbs.  
- Stage small constants in constant memory.

---

## 9. Numeric Stability

- Deterministic rounding convention.  
- Accumulate in FP32 minimum.  
- Optional compensated summation.

---

## 10. Constraints

- If \(Q^{M\cdot D} > 2^{64}\), disallow FULL_PACKED_64.  
- If \(Q^D > 2^{32}\), disallow 32-bit LEVEL_PACKED.  
- If LUT exceeds memory, require blocked processing.

---

## 11. Testing Plan

- **Unit tests**:
  - Encoder/decoder round-trip.
  - Pack/unpack identity.
  - Tie cases.

- **Property tests**:
  - Batch reshaping invariance.
  - Parameter validation.

- **VLUT correctness**:
  - Compare to dense baseline.

- **Performance checks**:
  - Bandwidth, occupancy, GFLOPs/GB.

---

## 12. Integration Hooks

- Expose C ABI wrappers.  
- Accept raw pointers and strides.  
- Provide opaque `EncState` / `VLUTState` handle management.  
- Stream-aware launches.

---

## 13. Developer Checklist

1. Implement pack/unpack utilities.  
2. Implement CPU reference encoder/decoder.  
3. Port to CUDA encoder.  
4. Build LUT key layouts.  
5. Implement two-sided MAC.  
6. Add blocked LUT path.  
7. Add profiling hooks.  
8. Document parameter limits.

---