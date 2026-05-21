# EXL3 Quantization Format — Technical Specification

> Based on exllamav3 v0.0.32 (https://github.com/turboderp-org/exllamav3), Qwen3-0.6B-EXL3-5bpw
> Reference files (exllamav3 root `exllamav3/`):
> - `modules/quant/exl3.py` — LinearEXL3 forward, unpack_bf, get_weight_tensor
> - `modules/quant/exl3_lib/quantize.py` — Quantizer, preapply_had_l/r, regularize
> - `exllamav3_ext/quant/hadamard.cu` — `had_r_128` host wrapper
> - `exllamav3_ext/quant/hadamard_inner.cuh` — CUDA kernel (`had_hf_r_128_inner`)
> - `exllamav3_ext/quant/exl3_dq.cuh` — Trellis unpack + codebook decode
> - `util/hadamard.py` — Hadamard matrix generation

---

## 1. Overview

EXL3 is a post-training quantization format that uses:

- **Trellis Coded Quantization (TCQ)** with a bitshift trellis to encode weights
- **Procedural codebooks** (LCG + LOP3) — no stored codebook, computed on-the-fly
- **Incoherence processing** via random Hadamard transforms on activations

A linear layer's weights are replaced by three tensors: `trellis`, `suh`, `svh`.

---

## 2. Tensor Layout

### 2.1 trellis — Packed Quantized Indices

```
Shape:  [tiles_k, tiles_n, tile_packed_size]
Dtype:  int16   (signed, stored int16 but values are packed uint16)
Where:
  tiles_k         = ceil(in_features / 16)
  tiles_n         = ceil(out_features / 16)
  tile_packed_size = 256 * K // 16    (K = bitrate, e.g. 5 → 80)
```

The weight matrix (in_features, out_features) is divided into 16×16 tiles.
Each tile of 256 weights is quantized to K-bit indices, then packed into `tile_packed_size` uint16 words.

A tile at position `(tk, tn)` in the weight matrix is stored at `trellis[tk, tn, :]`.

**Packing format (see `pack.cu`, `exl3_dq.cuh`):**

Each tile has 256 K-bit values. These are packed MSB-first into a uint16 array.

For K=5 (5bpw): 256 × 5 = 1280 bits = 80 uint16 words.

The packing arranges values across 16 spans of 16 values each. Each span writes 16*K bits sequentially into uint16 words, flushing to the array every 16 bits. This creates a layout where the K-bit values are bit-packed but aligned to 16-word boundaries for efficient extraction.

**Unpacking (`exl3_dq.cuh`):**

The function `fshift(b, a, shift)` extracts a 16-bit window from a 64-bit (b<<32 | a) pair:

```c
uint32_t fshift(uint32_t b, uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}
```

For a tile at offset `t_offset` (0..255), the bit position of word 0 is:

```
b0 = t_offset * K + K - 16 + 256 * K
```

This formula:
- `t_offset * K` — start bit of the desired value
- `+ K - 16` — shift so the 16-bit window aligns the K-bit value at the top bits
- `+ 256 * K` — ensures positive index before modulo to avoid negative indices in C++

Two uint32 words `a` and `b` are loaded from the packed array, forming a 64-bit window. The `fshift` extracts the right 16-bit aligned slice.

The 16-bit word, after masking, becomes a uint16 index that's passed to the codebook decoder:

```c
uint32_t w0 = fshift(b, a, s0) & 0xffff;
return decode_3inst<cb>(w0);
```

**Dispatch by bitrate (from `dq_dispatch`):**

| K | Function | Outputs |
|---|---|---|
| 1 | `dq8_aligned_1bit` | 8 fp16 values |
| 2 | `dq8_aligned_2bits` | 8 fp16 values |
| 3 | `dq8<bits, 4>` | 8 fp16 values (align=4 packed) |
| 4 | `dq8_aligned_4bits` | 8 fp16 values |
| 5 | `dq4 × 2` | 8 fp16 values |
| 6 | `dq4 × 2` | 8 fp16 values |
| 7 | `dq2x2 × 2` | 8 fp16 values |
| 8 | `dq4 × 2` | 8 fp16 values |

Each call returns 8 decoded fp16 values per warp, arranged as 2× `FragB` (each FragB = 2 × `half2`).

### 2.2 suh — Input-side Scale/Hadamard

```
Shape:  [in_features]
Dtype:  float16
```

Scale factors applied to the input activations **before** the Hadamard transform.
Each element `suh[i]` scales input channel `i`.

The suh tensor encodes:
- Channel-wise scaling factors (from quantization regularization)
- Random sign flips (from incoherence processing)

In older model versions, su was stored as packed int16 (16 random sign bits per group of 16 channels). The `unpack_bf` method in `exl3.py` converts packed int16 → float16:

```python
bits = (bitfield.unsqueeze(-1) & masks) > 0
result = where(bits, -1.0, 1.0)  # sign flip only
```

Modern models store suh directly as float16 (pre-combined scale + sign).

### 2.3 svh — Output-side Scale/Hadamard

```
Shape:  [out_features]
Dtype:  float16
```

Scale factors applied to the output activations after the Hadamard transform.
Each element `svh[i]` scales output channel `i`.

Same encoding as suh, but for the output dimension.

### 2.4 mcg / mul1 — Codebook Selector (optional)

```
Shape:  scalar (0-d)
Dtype:  int32 (stored as uint32 cast to int in safetensors)
```

Determines which procedural codebook variant is used for **all tiles** in this layer.
If neither tensor exists, cb=0 is used.

| Tensor | Codebook | LCG Multiplier |
|---|---|---|
| (none) | cb=0 | 89226354 |
| `mcg` | cb=1 | 0xCBAC1FED |
| `mul1` | cb=2 | 0x83DCD12D |

Key: `mcg` and `mul1` are encoded as their **multiplier constants cast to uint32** in the safetensors file. During loading, these are checked for existence, not value.

---

## 3. Procedural Codebook Decode

Each 16-bit word from the packed trellis is decoded to a float16 value via one of three deterministic procedures.

### 3.1 Codebook 0 (default, no mcg/mul1)

```python
def decode_cb0(x: uint16) -> float16:
    x = uint32(x)                     # zero-extend to 32-bit
    x = x * 89226354                  # LCG multiply (wraps at 2^32)
    x = x + 64248484                  # LCG add
    # LOP3 with truth table 0x6a:
    # ((x & 0x8fff8fff) | (~x & 0x3b603b60))
    x = (x & 0x8fff8fff) | (~x & 0x3b603b60)
    # Reinterpret low 16 bits as float16
    lo = float16(x & 0xFFFF)
    # Reinterpret high 16 bits as float16
    hi = float16((x >> 16) & 0xFFFF)
    return lo + hi
```

### 3.2 Codebook 1 (mcg, MCG codebook)

```python
def decode_cb1(x: uint16) -> float16:
    x = uint32(x)
    x = x * 0xCBAC1FED                # LCG multiply (wraps at 2^32)
    # Same LOP3 as cb0
    x = (x & 0x8fff8fff) | (~x & 0x3b603b60)
    lo = float16(x & 0xFFFF)
    hi = float16((x >> 16) & 0xFFFF)
    return lo + hi
```

### 3.3 Codebook 2 (mul1, MUL1 codebook)

```python
def decode_cb2(x: uint16) -> float16:
    x = uint32(x)
    x = x * 0x83DCD12D                # LCG multiply (wraps at 2^32)
    # vabsdiff4: sum of byte-level absolute differences from 0
    # Equivalent to: sum of 4 bytes of x
    byte0 = (x >> 0) & 0xFF
    byte1 = (x >> 8) & 0xFF
    byte2 = (x >> 16) & 0xFF
    byte3 = (x >> 24) & 0xFF
    sum = byte0 + byte1 + byte2 + byte3 + 0x6400
    # Affine transform
    k_inv = 0.00677    # 1/147.7  (half: 0x1eee)
    k_bias = -10.39    # (half: 0xc931)
    return float16(sum) * k_inv + k_bias
```

### 3.4 LOP3 Truth Table Details

The LOP3 instruction on NVIDIA GPUs evaluates a 3-input Boolean function. With truth table 0x6a = 0b0110_1010:

| a | b | c | out |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 (= c) |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 (= c) |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 (= c) |
| 1 | 1 | 0 | 1 (= a) |
| 1 | 1 | 1 | 0 |

Expression: `out = (a & b) | (~a & c)` = `c ^ (a & (b ^ c))`

With a=x, b=0x8fff8fff, c=0x3b603b60:
```
result = (x & 0x8fff8fff) | (~x & 0x3b603b60)
```

---

## 4. Reconstruct (Full Weight Decode)

The `reconstruct` kernel produces a full FP16 weight matrix from the packed trellis.

```
Output shape:  [in_features, out_features]  (row-major)
Dtype:         float16
```

**Algorithm (CPU version):**

```python
def reconstruct(trellis, in_features, out_features, K, cb):
    """
    trellis:  [tiles_k, tiles_n, tile_packed_size]  int16
    K:        bitrate
    cb:       codebook variant (0, 1, or 2)

    Returns:  [in_features, out_features]  float16
    """
    tiles_k = in_features // 16
    tiles_n = out_features // 16
    output = zeros(in_features, out_features, dtype=float16)

    for tk in range(tiles_k):
        for tn in range(tiles_n):
            tile_packed = trellis[tk, tn, :]  # packed tile data

            # Decode 256 values per tile
            for i in range(256):
                # Extract K-bit word at position i
                w = extract_word(tile_packed, K, i)  # returns uint16
                # Decode to float16
                val = decode(w, cb)  # cb0/cb1/cb2

                r = tk * 16 + (i // 16)   # row in weight matrix
                c = tn * 16 + (i % 16)    # col in weight matrix
                output[r, c] = val

    return output
```

Note: The **Tensor Core permutation** used in `reconstruct.cu` (`__shfl_down_sync` + shared memory scatter) is specific to the NVIDIA Tensor Core layout and is **not needed for the CPU decoder**. The CUDA kernel permutes tile values to match Tensor Core's m16n8k16 MMA format (8 warps × 8 rows × 4 cols). For CPU, values should be stored directly in row-major order.

---

## 5. Inference Pipeline

### 5.1 Weight Layout Convention

The weight matrix is stored as `[in_features, out_features]` (transposed from PyTorch's `[out_features, in_features]` convention).

The matmul is: `y = x @ W` where:
- `x`: `[batch, in_features]`
- `W`: `[in_features, out_features]`
- `y`: `[batch, out_features]`

This is the same as `y = x @ W` in linear algebra notation, NOT PyTorch's `F.linear(x, weight)` which does `x @ weight.T`.

### 5.2 Two Forward Strategies

EXL3 supports two mathematically equivalent forward strategies. Our implementation uses **Strategy A** (reconstruct path, matching exllamav3's batch > 32 path). Strategy B is documented for reference.

#### Strategy A: Reconstruct path (what we use)

Decode raw weights from trellis at load time, then apply Hadamard transforms to activations during forward:

```
input x:  [batch, in_features]

# Apply input-side incoherence
x_had = fwht_128(x * suh)            # pre_scale before FWHT
x_had *= 0.088388347648              # 1/sqrt(128)

# Reconstructed raw weights (no Hadamard pre-applied)
w = reconstruct(trellis)             # [in_features, out_features] fp16

# GEMM
y = x_had @ w                        # fp16 matmul

# Apply output-side incoherence
y = fwht_128(y)                      # 128-block FWHT on output
y *= 0.088388347648                  # 1/sqrt(128) normalization
y *= svh                             # Element-wise scale

if bias is not None:
    y += bias

return y
```

Note: `suh` is applied **before** the FWHT (as a pre-scale), `svh` is applied **after** the FWHT + norm (as a post-scale). The FWHT itself is always unnormalized; normalization is applied as a separate multiply after the transform.

#### Strategy B: Weight-fused path (exllamav3 `get_weight_tensor`)

Pre-apply the Hadamard transforms into the weight tensor itself, then do a plain matmul. Used by exllamav3 for model inspection/saving:

```python
def get_weight_tensor(self):
    suh = self.suh.unsqueeze(1)   # [in_features, 1]
    svh = self.svh.unsqueeze(0)   # [1, out_features]
    w = self.get_inner_weight_tensor()  # raw trellis decode

    # Pre-apply input Hadamard + suh
    w = preapply_had_l(w, had_k)  # (1/sqrt(128)) * H @ w  (per 128-row block)
    w *= suh                      # element-wise input scale

    # Pre-apply output Hadamard + svh
    w = preapply_had_r(w, had_n)  # w @ H * (1/sqrt(128))  (per 128-col block)
    w *= svh                      # element-wise output scale
    return w
```

Forward then becomes a plain matmul on already-Hadamarded weights:
```python
y = x @ get_weight_tensor()
```

**The two strategies produce identical results.** Strategy A applies `1/sqrt(128)` to the activations during forward; Strategy B bakes the same factor into the weights at load time.

### 5.3 Hadamard Transform (128×128)

The Hadamard transform operates on the 128-element leading dimension for both input and output.

**Our implementation** (`hadamard_rotate_128` / `hadamard_rotate_128_cuda`):

```
output = FWHT(input × pre_scale) × norm × post_scale
```

Where:
- `pre_scale`: optional [dim] element-wise scale, applied before FWHT (in fp16)
- `FWHT`: unnormalized fast Walsh-Hadamard transform (in fp32)
- `norm`: post-transform normalization factor
- `post_scale`: optional [dim] element-wise scale, applied after norm (in fp16)

The FWHT-128 butterfly in 7 stages (log2(128) = 7 steps):

```python
def fwht_128(x):
    """In-place fast Walsh-Hadamard transform, length 128."""
    n = 128
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                a = x[..., j]
                b = x[..., j + step]
                x[..., j] = a + b
                x[..., j + step] = a - b
        step *= 2
    return x
```

### 5.4 ABI Note: Normalization Convention vs exllamav3

The `norm` parameter in our API has **different semantics** from the `scale` parameter in exllamav3's `ext.had_r_128`. This is a common source of confusion:

| | exllamav3 `ext.had_r_128` | Our `hadamard_rotate_128[_cuda]` |
|---|---|---|
| Kernel computes | `r_scale = scale × **1/√128**` | `r_scale = **norm**` (direct) |
| Call for input side | `had_r_128(..., suh, None, scale=1.0)` | `hadamard_rotate_128(..., pre_scale=suh, norm=INV_SQRT_128)` |
| Effective r_scale | `1.0 × 1/√128 = 1/√128` | `1/√128` |
| Call for output side | `had_r_128(..., None, svh, scale=1.0)` | `hadamard_rotate_128(..., post_scale=svh, norm=INV_SQRT_128)` |
| Effective r_scale | same: `1/√128` | same: `1/√128` |

**Both sides use the same `1/√128` normalization** in both implementations. The difference is only in how the parameter is labeled and whether the kernel multiplies by `1/√128` internally.

The default `norm = INV_SQRT_128` in our Nim API is correct for **both** input and output sides when using the reconstruct path (Strategy A). The exllamav3 reference calls `ext.had_r_128(..., scale=1.0)` for both sides — the `1.0` does **not** mean "no normalization"; it means "use the kernel's built-in `1/√128`".

> ℹ️ The older spec note `norm_factor: 1/sqrt(128) for suh, 1.0 for svh` described the exllamav3 kernel API where `scale=1.0` maps to `1/√128`. It does NOT apply to our kernel, which interprets `norm` directly.

---

## 6. EXL3 Weight Tensors in Safetensors

### 6.1 Naming Convention

For each linear layer `{layer_path}` (e.g., `model.layers.0.self_attn.q_proj`):

| Tensor Name | Shape | Dtype | Required |
|---|---|---|---|
| `{layer_path}.trellis` | [tiles_k, tiles_n, 256*K//16] | int16 | Yes |
| `{layer_path}.suh` | [in_features] | float16 | Yes |
| `{layer_path}.svh` | [out_features] | float16 | Yes |
| `{layer_path}.mcg` | scalar | uint32 (→int) | No (if present → cb=1) |
| `{layer_path}.mul1` | scalar | uint32 (→int) | No (if present → cb=2) |
| `{layer_path}.bias` | [out_features] | float16 | No |

### 6.2 Layer Types That Have EXL3 Weights

Only **linear projection layers** are quantized. In Qwen3-0.6B-EXL3-5bpw:

| Layer | in_features | out_features | trellis shape |
|---|---|---|---|
| `q_proj` | 1024 | 2048 | [64, 128, 80] |
| `k_proj` | 1024 | 1024 | [64, 64, 80] |
| `v_proj` | 1024 | 1024 | [64, 64, 80] |
| `o_proj` | 2048 | 1024 | [128, 64, 80] |
| `gate_proj` | 1024 | 3072 | [64, 192, 80] |
| `up_proj` | 1024 | 3072 | [64, 192, 80] |
| `down_proj` | 3072 | 1024 | [192, 64, 80] |
| `lm_head` | (lm head dims) | — | — |

### 6.3 Non-quantized Tensors

- Embedding weights: stored as float16/bfloat16, unchanged
- Layer norm weights: stored as float16, unchanged
- Rotary embeddings: stored as float16, unchanged

---

## 7. Deriving Parameters from Model

### 7.1 From config.json

```json
{
    "quantization_config": {
        "quant_method": "exl3",
        "version": "0.0.1",
        "bits": 5.0,
        "head_bits": 6,
        "calibration": { "rows": 100, "cols": 2048 },
        "out_scales": "auto"
    }
}
```

- `quantization_config.bits` → K (bitrate), always a whole number for EXL3
- `quantization_config.head_bits` → K for lm_head (if different)

If `quantization_config` is absent, the model is not EXL3-quantized.

### 7.2 Deriving K from trellis shape

```python
tile_packed_size = trellis.shape[2]  # e.g., 80
K = tile_packed_size * 16 // 256     # e.g., 80 * 16 / 256 = 5
```

### 7.3 Deriving Codebook from mcg/mul1

```python
has_mcg = "mcg" in safetensors_keys   # → cb=1
has_mul1 = "mul1" in safetensors_keys # → cb=2
else cb=0
```

---

## 8. Our Implementation Details

Our Nim implementation (`workspace/transformers/src/` and `workspace/positron/`) follows Strategy A (reconstruct path):

### 8.1 Forward Path

```nim
# CUDA path
let xf16 = x.to(kFloat16)
let xh = hadamard_rotate_128_cuda(xf16,
              pre_scale = some(self.suh),
              post_scale = none(Tensor))
              # norm=INV_SQRT_128 (default): FWHT(x·suh) × 1/√128
result = F.matmul(xh, self.weight)
result = hadamard_rotate_128_cuda(result,
              pre_scale = none(Tensor),
              post_scale = some(self.svh))
              # norm=INV_SQRT_128 (default): FWHT(y) × 1/√128 × svh

# CPU fallback (same logic, portable tensor operations)
let xf16 = x.to(kFloat16)
let xh = hadamard_rotate_128(xf16,
              pre_scale = some(self.suh),
              post_scale = none(Tensor))
result = F.matmul(xh, self.weight)
result = hadamard_rotate_128(result,
              pre_scale = none(Tensor),
              post_scale = some(self.svh))
```

### 8.2 Weight Loading

```nim
let w = exl3_reconstruct(trellis, K, cb, in_f, out_f).contiguous()
# w is [in_features, out_features], raw fp16 from trellis decode
# No Hadamard pre-applied — Hadamard is applied to activations during forward

Linear.init(
  weight = w,
  bias,
  suh,
  svh
)
```

### 8.3 Load-Time Decode for Autoregressive Decoding

For batch-1 autoregressive decoding:

1. **Decode weights once** at load time — store the full FP16 weight matrix
   - Cost: ~0.3 seconds for Qwen3-0.6B on CPU, ~3-5 seconds for 7B
   - Decoded memory: in_features × out_features × 2 bytes per linear layer

2. **Forward pass** then becomes: Hadamard → matmul → Hadamard:
   - `xh = FWHT(x·suh) × 1/√128`
   - `y = xh @ W`
   - `y = FWHT(y) × 1/√128 × svh`

3. **Hadamard on activations** is lightweight:
   - 7 stages of butterfly per activation vector
   - For Qwen3-0.6B (dim=1024, n_layers=28): 28 × 6 × 1024 × 2 = ~344K fp16 ops per token
   - Negligible compared to matmul

---

## 9. Codebook C++ Port Reference

### 9.1 Funnel Shift

```cpp
// Portable CPU version of __funnelshift_r
inline uint32_t funnel_shift_r(uint32_t b, uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}
```

### 9.2 Codebook Decode (cb=0, CPU)

```cpp
float decode_cb0(uint16_t x) {
    uint32_t ux = x;
    ux = ux * 89226354u + 64248484u;
    ux = (ux & 0x8fff8fffu) | (~ux & 0x3b603b60u);
    // Reinterpret as fp16 and add
    uint16_t lo = ux & 0xFFFF;
    uint16_t hi = (ux >> 16) & 0xFFFF;
    return half_to_float(lo) + half_to_float(hi);
}
```

### 9.3 Codebook Decode (cb=1, MCG, CPU)

```cpp
float decode_cb1(uint16_t x) {
    uint32_t ux = x;
    ux = ux * 0xCBAC1FEDu;
    ux = (ux & 0x8fff8fffu) | (~ux & 0x3b603b60u);
    uint16_t lo = ux & 0xFFFF;
    uint16_t hi = (ux >> 16) & 0xFFFF;
    return half_to_float(lo) + half_to_float(hi);
}
```

### 9.4 Codebook Decode (cb=2, MUL1, CPU)

```cpp
float decode_cb2(uint16_t x) {
    uint32_t ux = x;
    ux = ux * 0x83DCD12Du;
    // vabsdiff4(x, 0, 0x6400) equivalent
    uint32_t sum = 0x6400u;
    sum += (ux >> 0) & 0xFF;
    sum += (ux >> 8) & 0xFF;
    sum += (ux >> 16) & 0xFF;
    sum += (ux >> 24) & 0xFF;
    // Affine: (sum - 510) * 0.00677
    return ((float)(sum) - 510.0f) * 0.00677f;
}
```

### 9.5 Half Conversion

On x86, use `_mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(v)))` for F16C.
On ARM, use `vcvt_f32_f16`.
In Nim with libtorch, use `torch::from_blob` and let PyTorch handle the format.
