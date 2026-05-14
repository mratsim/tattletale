# Tattletale Development Conventions

## Purpose

This document establishes coding conventions for the Tattletale transformers library. These conventions ensure:

1. **Consistency** across the codebase
2. **Interoperability** with C libraries (libtorch, CUDA)
3. **Performance** through explicit memory management
4. **Correctness** via clear ownership and mutation semantics
5. **Discoverability** through comprehensive documentation

---

## 1. Parameter Ordering

### 1.1 Core Principle

> **Context → Out → InOut → In**

### 1.2 Parameter Categories

| Order | Category | Description | Examples |
|-------|----------|-------------|----------|
| **1** | **Context/Self** | Self/context pointers, state | `self:`, `ctx:`, `engine:` |
| **2** | **Out** | Output-only arguments (written to) | `result: var Tensor`, `output: ptr Tensor` |
| **3** | **InOut** | Arguments read AND modified | `cache: var KVCache`, `state: var InferenceContext` |
| **4** | **In** | Input-only arguments (read-only) | `input: Tensor`, `positions: Tensor`, `config: Config` |

### 1.3 Special Rules

#### Rule 1: Complex to Simple (Within Categories)

Within each category (Out/InOut/In), order from complex to simple:

```nim
# ✅ Correct: Tensor → Tensor → bool
proc forward*(
    self: GatedMLP,           # Context (1)
    x: Tensor                 # In (4)
): Tensor                     # Return

# ✅ Correct: Tensor → int → bool
proc softmax*(
    input: Tensor,            # In (4) — complex
    dim: int                  # In (4) — simple
): Tensor
```

**Rationale:** Complex types (tensors) carry more semantic weight and should be established first. Scalar parameters (int, bool) are modifiers that come last.

#### Rule 2: Default Parameters Go Last

Parameters with default values must be placed at the end, allowing callers to omit them:

```nim
# ✅ Correct: default param last
proc applyRope*(
    self: RotaryPositionEmbedding,
    q, k: Tensor,
    offset: int = 0           # Default, can be omitted
): (Tensor, Tensor)

# Usage:
let (qRot, kRot) = rotary.applyRope(q, k)           # ✅ offset defaults to 0
let (qRot, kRot) = rotary.applyRope(q, k, 128)      # ✅ explicit offset

# ❌ Wrong: default param in middle
proc applyRope*(
    self: RotaryPositionEmbedding,
    offset: int = 0,          # ❌ Can't omit this and specify q, k
    q, k: Tensor
): (Tensor, Tensor)
```

**Interaction with main ordering:** Default parameters still respect the Context → Out → InOut → In ordering within their category. They're simply placed at the end of that category.

#### Rule 3: Avoid Redundant Parameters

Don't pass parameters that can be derived from other arguments:

```nim
# ✅ Correct: seqLen extracted from x
proc forward*(
    self: TransformerBlock,
    x: Tensor                 # Shape: (batch, seq_len, hidden_size)
): Tensor

# ❌ Wrong: redundant seqLen parameter
proc forward*(
    self: TransformerBlock,
    x: Tensor,
    seqLen: int               # ❌ Redundant: x.size(1)
)
```

**Rationale:** Redundant parameters create opportunities for inconsistency and bugs. Extract dimensions from tensors as needed.

#### Rule 4: Exception for Bindings/Wrappers

When creating bindings to upstream libraries (libtorch, CUDA kernels, C APIs), **match the upstream parameter ordering** even if it violates our conventions:

```nim
# ✅ Correct: Matches libtorch's parameter order
func linear*(input, weight: Tensor): Tensor
func linear*(input, weight, bias: Tensor): Tensor

# ✅ Correct: Matches CUDA kernel signature
proc launchKernel(
    gridDim: Dim3,
    blockDim: Dim3,
    sharedMem: csize_t,
    stream: cudaStream_t,
    kernel: pointer,
    args: pointer
): void
```

**Rationale:**
1. **Documentation alignment** — Easier to refer to upstream docs
2. **Portability** — Easier to compare ports across languages
3. **Maintenance** — Upstream API changes are easier to track
4. **Interoperability** — Reduces cognitive load when switching between layers

**When this applies:**
- Direct libtorch bindings (`workspace/libtorch/`)
- CUDA kernel wrappers
- C library FFI bindings
- Ported algorithms where reference impl is in another language

**When this does NOT apply:**
- High-level Tattletale APIs (`workspace/transformers/`)
- User-facing interfaces
- Internal abstractions

---

## 2. Documentation Standards

### 2.1 Docstring Structure

All public procs/types must follow this structure:

```nim
proc forward*(self: GatedMLP, x: Tensor): Tensor =
  ## Brief one-line description.
  ##
  ## Extended description if needed (optional).
  ##
  ## Args:
  ##   x: Input tensor of shape (..., hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (..., hidden_size)
  ##
  ## Computes:
  ##   gate_up_proj = self.gate_up_proj.forward(x)
  ##   activation = silu_and_mul(gate_up_proj)
  ##   return self.down_proj.forward(activation)
```

### 2.2 Required Sections

| Section | When Required | Format |
|---------|---------------|--------|
| **Brief** | Always | Single line, imperative mood |
| **Args** | When proc has parameters | `name: Description with shape` |
| **Returns** | When proc returns non-void | `Type with shape` or description |
| **Note** | For important caveats | Free-form text |
| **Computes** | For complex operations | Step-by-step pseudocode |

### 2.3 Shape Documentation

#### Rule 1: Always Document Tensor Shapes

```nim
# ✅ Correct
proc forward*(
    self: TransformerBlock,
    x: Tensor,                # (batch, seq_len, hidden_size)
    residual: Option[Tensor]  # (batch, seq_len, hidden_size) or None
): (Tensor, Tensor)           # (output, residual): both (batch, seq_len, hidden_size)

# ❌ Wrong: missing shapes
proc forward*(
    self: TransformerBlock,
    x: Tensor,                # Input tensor
    residual: Option[Tensor]  # Optional residual
): (Tensor, Tensor)
```

#### Rule 2: Use `...` for Broadcast Dimensions

```nim
# ✅ Correct: supports any leading dimensions
proc forward*(
    self: GatedMLP,
    x: Tensor                 # (..., hidden_size)
): Tensor                     # (..., hidden_size)

# ✅ Correct: explicit batch dimension
proc forward*(
    self: Linear,
    x: Tensor                 # (batch_size, in_features)
): Tensor                     # (batch_size, out_features)
```

#### Rule 3: Document Multiple Accepted Shapes

```nim
# ✅ Correct: documents both formats
func scaledDotProductAttention*(
    query, key, value: Tensor # (B, H, L, d_k) or (B, L, H * d_k)
): Tensor                     # (B, H, L, d_v) or (B, L, H * d_v)
```

#### Rule 4: Document Broadcast Behavior

```nim
# ✅ Correct: documents broadcast
func scaledDotProductAttention*(
    query: Tensor,            # (B, H_q, L, d_k)
    key: Tensor,              # (B, H_kv, L, d_k) — H_kv broadcast to H_q if H_kv < H_q
    value: Tensor             # (B, H_kv, L, d_v) — H_kv broadcast to H_q if H_kv < H_q
): Tensor                     # (B, H_q, L, d_v)
```

### 2.4 Examples from Codebase

#### Layer Forward (transformer.nim)

```nim
proc forward*(self: var TransformerBlock, x: Tensor, residual: Option[Tensor]): (Tensor, Tensor) =
  ## Forward pass for a transformer block with long residual stream.
  ##
  ## This pattern defers residual additions to the norm layers, enabling:
  ## - Pipeline parallelism support (residual can cross stage boundaries)
  ## - Fused norm+residual kernels (vLLM optimization)
  ## - Single addition per layer (instead of two)
  ##
  ## Args:
  ##   x: Input tensor of shape (batch, seq_len, hidden_size)
  ##   residual: Optional residual from previous layer. If None, uses x.
  ##
  ## Returns:
  ##   (output, residual) where:
  ##     - output: Tensor of shape (batch, seq_len, hidden_size)
  ##     - residual: The accumulated residual, passed through unchanged
  ##
  ## Note:
  ##   RoPE positions and KV cache are handled internally by the attention layer.
  ##
  ## Computation:
  ##   residual = residual.get(x)  # Use x if residual is None
  ##   (h, residual) = self.attn_norm.forward_with_residual(x, residual)
  ##   attn_out = self.attn.forward(h)
  ##   (h2, residual) = self.mlp_norm.forward_with_residual(attn_out, residual)
  ##   mlp_out = self.mlp.forward(h2)
  ##   (mlp_out, residual)
```

#### MLP Forward (mlp.nim)

```nim
proc forward*(self: GatedMLP, x: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (..., hidden_size)
  ##
  ## Computes:
  ##   gate_up_proj = self.gate_up_proj.forward(x)  # (..., 2 * intermediate_size)
  ##   activation = silu_and_mul(gate_up_proj)        # (..., intermediate_size)
  ##   return self.down_proj.forward(activation)      # (..., hidden_size)
```

#### SDPA (libtorch/tensors_nn.nim)

```nim
func scaledDotProductAttention*(
    query, key, value: Tensor,
    attn_mask: Option[Tensor] = none(Tensor),
    dropout_p: float64 = 0.0,
    is_causal: bool = false,
    scale: Option[float64] = none(float64)
): Tensor =
  ## Computes softmax(Q @ K^T / scale) @ V with efficient memory attention.
  ##
  ## Uses Paged KV cache for efficient generation and supports:
  ## - Attention masking (padding masks, etc.)
  ## - Causal/prefix attention (autoregressive decoding)
  ## - Grouped-Query Attention (GQA) for efficient LLaMA-style models
  ##
  ## Args:
  ##   query: Query tensor of shape (B, H_q, L, d_k) or (B, L, H_q * d_k)
  ##   key: Key tensor of shape (B, H_kv, L, d_k) or (B, L, H_kv * d_k)
  ##   value: Value tensor of shape (B, H_kv, L, d_v) or (B, L, H_kv * d_v)
  ##   attn_mask: Optional attention mask. If provided, applied before softmax.
  ##   dropout_p: Dropout probability (default: 0.0, no dropout)
  ##   is_causal: If true, applies causal mask (default: false)
  ##   scale: Scaling factor for Q @ K^T. If None, uses 1/sqrt(d_k)
  ##
  ## Returns:
  ##   Attention output of shape (B, H_q, L, d_v) or (B, L, H_q * d_v)
  ##
  ## Note:
  ##   - Supports GQA: if H_kv < H_q, keys/values are broadcast to match query heads
  ##   - For causal attention, use is_causal=true instead of manual mask
```

---

## 3. Testing Conventions

### 3.1 Test File Naming

- `tests/test_<model>_<scope>.nim` or `tests/t_*.nim` -> live test
- `tests/bug_test_<scope>.nim` -> test that doesn't pass (for example raw TorchTensor destructors)
- `tests/manual_test_<scope>.nim` -> test that needs to run manually (for example requires a GPU)
- Fixtures: `tests/fixtures/`
- Fixtures generator: `tests/testgen/`
- Group by semantic folder if tests become too unwieldly

Examples:
- `test_qwen3_rope.nim` → RoPE unit tests
- `test_qwen3_full_inference.nim` → Full model integration test

### 3.2 Test Structure

```nim
import workspace/libtorch_testutils

proc main() =
  runTest "Test name — what it verifies":
    proc(): bool =
      # Setup
      let fixture = loadFixture("...")

      # Exercise
      let result = functionUnderTest(fixture.input)

      # Verify
      assertAllClose(result, fixture.expected, rtol=1e-3)
      true

when isMainModule:
  main()
```

### 3.3 Fixture Management

Fixtures should be loaded from `fixtures/` directory:

```nim
const FixtureDir = currentSourcePath().parentDir() / "fixtures" / "Qwen3-0.6B"

proc loadFixture(name: string): JsonNode =
  (FixtureDir / name).parseFile()
```
We use jsony for parsing directly into an object with fixed fields
or packedjson when we need dynamic fields

---

## 4. Documenting Invariants

For complex types with non-trivial state, document the invariants that must hold:

```nim
type KVCache* = object
  ## KV Cache for attention layers.
  ##
  ## INVARIANT:
  ##   - keys and values have same shape: (max_batch, kv_heads, max_seq, head_dim)
  ##   - offset <= max_seq
  ##   - keys and values are preallocated (no cat/append in hot path)
  keys: Tensor
  values: Tensor
  offset: int

# INVARIANT: Config fields are immutable after loadQwen3Config()
type Qwen3Config* = ref object
  hiddenSize*: int
  numLayers*: int
```

---

## 5. Summary Checklist

Before committing code, verify:

- [ ] **Parameter order**: Context → Out → InOut → In
- [ ] **No redundant params**: Extract from tensors, don't pass separately
- [ ] **Default params**: At the end (can be omitted)
- [ ] **Bindings exception**: Upstream order for wrappers/FFI
- [ ] **Docstring structure**: Brief, Args, Returns, (Note/Computes if needed)
- [ ] **Shape docs**: All tensors have shapes documented
- [ ] **Broadcast docs**: Document if dimensions are broadcast
- [ ] **Invariants**: Documented for complex types
- [ ] **Tests**: Follow naming and structure conventions

---
