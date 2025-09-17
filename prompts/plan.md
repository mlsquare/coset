# Plan for High-Performance PyTorch Codebase: Hierarchical Nested-Lattice Quantization

This document outlines a step-by-step plan to implement a **PyTorch-based library** that enables quantization-aware training (QAT) using hierarchical nested-lattice quantization (Kaplan & Ordentlich, 2025). The goal is to build a performant and extensible system with CUDA acceleration, gradient compression hooks for distributed training, and clean testing infrastructure.

---

## 0) Repo scaffolding & standards

**Package name:** `hlqtorch` (Hierarchical Lattice Quantization for Torch)

**Directory structure:**
```
coset/
  __init__.py
  lattices/
    __init__.py
    zd.py        # Z^d: baseline NN quantizer
    d4.py        # D4: fast NN quantizer
    e8.py        # E8: later
  quant/
    __init__.py
    functional.py   # encode, decode, quantize, inner_product_lut, add_modq
    lut.py          # LUT builders (two-sided & one-sided)
    params.py       # lattice params (d, q, M), scaling β, dithers
  nn/
    __init__.py
    qlinear.py      # QAT-ready Linear layer
  dist/
    __init__.py
    comm_hooks.py   # Gradient compression hooks
  cuda/
    CMakeLists.txt / setup.py
    bindings.cpp
    encode_kernel.cu
    decode_kernel.cu
    lut_innerprod.cu
    modq_arith.cu
  tests/
    test_encode_decode.py
    test_lut_ip.py
    test_qlinear_qat.py
    test_comm_hook.py
    test_kernels_correctness.py
  benchmarks/
    bench_encode_decode.py
    bench_lut_innerprod.py
    bench_qlinear_throughput.py
  examples/
    mnist_qat.py
    cifar_qat.py
pyproject.toml / setup.cfg
```

**Coding standards:**  
- Type hints  
- `ruff` + `black`  
- `pytest` for testing  
- CI with CPU tests (GPU optional)  

---

## 1) Core data model & shapes

- **Axis convention:** For tensor `X[..., d]`, quantize only the last dimension (`d`).  
- **Parameters:**  
  - `d` ∈ {2,4,8}  
  - `q` (odd or power of 2)  
  - `M` depth (1–4 typical)  
- **Encodings:** `b ∈ [q]^d`, shape `[..., M, d]`, dtype `uint8` or `int16`.  
- **A_q elements:** derived from `(G, Q_L)`.  
- **Scales & dithers:** `β` (scaling), `z ∈ q^{-1}A_q` (structured dither).  

---

## 2) Lattice layer: nearest-neighbor Q_L

Implement lattices:

- `Z^d`: trivial rounding.  
- `D4`: checkerboard NN logic.  
- `E8`: reference implementation provided in /ref/*

**API:**
```python
class Lattice:
    name: str
    d: int
    G: torch.Tensor
    G_inv: torch.Tensor
    def Q(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Tests:**  
- Idempotence: `Q(Q(x)) == Q(x)`  
- Golden vectors for D4/E8  

---

## 3) Algorithms 1 & 2 (reference implementation)

**functional.py:**
```python
def encode(x, lattice, q: int, M: int, *, beta=1.0, dither=None):
    # Algorithm 1
    ...

def decode(b, lattice, q: int, M: int):
    # Algorithm 2
    ...

def quantize(x, lattice, q, M, *, beta=1.0, dither=None):
    ...
```

**Tests:**  
- `Z^d`: check `decode(encode(x)) == Q_L(x)`  
- D4: verify telescoping identity  

---

## 4) LUT builders (inner product)

**lut.py**
- Two-sided LUT: size `q^{2d}`  
- One-sided LUT (per-query): size `q^d`

**APIs:**
```python
def build_two_sided_LUT(lattice, q): ...
def build_one_sided_LUT(y, lattice, q): ...
```

**Tests:**  
- Compare LUT IP with explicit decode & dot product  

---

## 5) Mod-q arithmetic & compressed-domain ops

**add_modq:**  
```python
def add_modq(b1, b2, q: int):
    return (b1 + b2) % q
```

**Note:** compressed-domain addition is exact in `A_q`, but decoded sums may not match float addition unless no overload.  

---

## 6) CUDA & C++ extensions

Implement kernels:  
1. `encode_kernel`  
2. `decode_kernel`  
3. `lut_innerprod`  
4. `modq_arith`  

**Bindings:** use `torch.utils.cpp_extension`.  
**Tests:** check GPU vs CPU reference.  

---

## 7) QAT Linear module (`nn.QLinear`)

**Design:**  
- Maintain FP32 shadow weights.  
- Quantize at interval (`quantize_every`).  
- Support quantizing activations, weights, or both.  
- Option for LUT-based matmul.  
- STE for QAT.

**API sketch:**
```python
class QLinear(nn.Module):
    def __init__(..., d=4, q=4, M=2, quantize_weights=True,
                 quantize_activations=False, use_lut_mac=False, ...):
        ...

    def forward(self, x):
        ...
```

**Tests:**  
- Shape parity with `nn.Linear`  
- Forward match with float baseline  
- MNIST QAT smoke test  

---

## 8) Distributed gradient compression

Implement DDP comm hook.

**Hook contract:**
```python
def hlq_ddp_comm_hook(state, bucket):
    grad = bucket.get_tensor()
    b, overload = encode(grad_flattened, ...)
    grad_hat = decode(b, ...)
    fut = dist.all_reduce(grad_hat, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0] / world_size)
```

**Tests:**  
- Numerical equivalence vs baseline  
- Bandwidth savings  

---

## 9) Product-code tiling

For non-multiples of `d`, tile last dim.  

**Helpers:**
```python
def tile_last_dim(x, d): ...
def untile_last_dim(x_tiled, pad_info): ...
```

---

## 10) Overload avoidance

Add `β-ramp` (geometric). Retry until no overload.  
Expose `beta0`, `alpha`, `max_retries`, `track_T`.  

---

## 11) Benchmarks

- Encode/Decode throughput  
- LUT IP vs dequantized matmul  
- QLinear throughput  
- DDP bandwidth savings  

---

## 12) Documentation & examples

- Colab demo  
- Example scripts (MNIST, CIFAR)  
- Clear notes on compressed-domain arithmetic & overload handling  

---

## Minimal code stubs

**functional.py**
```python
def encode(x, lattice, q, M, *, beta=1.0, dither=None): ...
def decode(b, lattice, q, M, *, beta=1.0, dither=None): ...
def quantize(x, lattice, q, M, **kw): ...
```

**nn/qlinear.py**
```python
class QLinear(nn.Module):
    def __init__(..., d=4, q=4, M=2, quantize_every=1, ...):
        ...
    def forward(self, x):
        ...
```

---

## Validation roadmap

1. Encode/Decode correctness  
2. LUT IP matches decode-based dot products  
3. CUDA vs reference parity  
4. QLinear ≈ nn.Linear when β large  
5. QAT converges on MNIST  
6. DDP hook reduces bytes sent  

---

## Practical defaults

- Start with `D4`, `d=4`, `q=4`, `M=2`  
- `β=1.0`, no dither initially  
- Use dequantized matmul first, LUT MAC later  

---
