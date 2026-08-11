## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Epilogue fusion operations for the tiled GEMM.
##
## An Epilogue computes the output tile
##   D = f(AB)
## where
##   D   is the destination (var, the output tile in registers or gmem)
##   AB  is the accumulated GEMM result (the accumulator fragment)
##
## The op object is the epilogue configuration.
## It carries its operands as fields (EpiAXPBY's C, EpiAddBias's bias).
## `apply` reads them through the op-held smem pointers, filled by `preflight`.
##
## `apply` iterates `size(D)`, the destination's logical size.
## For each flat index i it computes D(i) from AB(i) and the staged operands.
## Each tensor is indexed through its own layout, so operand ranks and
## strides are unconstrained and need not match D's.
##
## alpha and beta are runtime values.
## The `apply` branches on them once, before the element loop, so the
## element loop has no branches on op state.
##
## ── Contract ──
## Each op must provide:
##   `preflight(op: var Self)`  a TEMPLATE that injects the staging buffer
##                            into the caller scope ({.inject.}, shared memory via {.shared.} on GPU)
##                            and stages the op's gmem operands into it
##                            (cp.async / TMA pending). EpiAXPBY's is a
##                            no-op stub: C is read per-thread from gmem in
##                            `apply` (direct register→gmem is the default;
##                            smem staging is future LoadKind work).
##
##   `apply(op, D, AB)`         The actual epilogue function
##
## D and AB share the shape type Sh: the compiler enforces equal shapes,
## a mismatched shape is a type error. They do not have to have the same types
## hence epilogue can do type conversions.
##
## gemm_tiled calls, in order:
##   `op.preflight()`           stages gmem operands (EpiAXPBY: no-op stub today)
##   `op.apply(D, AB)`          the epilogue math (EpiAXPBY reads C via op.C_gmem)
##
## Target contract:
##   `shard(op, D, tD, tDv)`    partition the captured operands by thread, projecting
##                              onto the output fragment the kernel already partitioned
##   `preflight(op)`            initiate async data movement (no-op stub today)
##   `apply(op, AB, ...)`       the epilogue math, pure — reads the accumulator and the
##                              shard's operand fragments, produces the output fragment.
##                              `apply` STAYS a concept member
##   `finalStore(op, out, D)`   the STORE — writes the output fragment to the gmem
##                              destination. A separate responsibility from the math:
##                              it is where ragged-tile predication and
##                              flash-attention-style delayed stores live
##   Today's fused `apply(D, AB)` (compute + store in one, D = the gmem destination
##   written directly) is the v1 form, the target splits it into apply + finalStore.
##
## The concept is Nim concepts V2 (bare `concept`, `Self` = the matched type).
## V2 concepts only accept proc requirements, so `preflight` cannot be a concept member.
## It still is a compile-time error not to have `preflight`, just that the message won't be as nice.
##
## Note: currently we need to use the verbose `TensorView[T, Sh, St] or Tensor[T, Sh, St]`
##       instead of AnyTensor[T, Sh, St] or the concept don't match.

import ./int_tuples
import ./layouts
import ./tensors

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Epilogue concept
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  ## The concept: an Epilogue computes the output tile with the 2-arg
  ## `apply(D, AB)` (the target name is `finalStore`, see the module
  ## header). D and AB share the shape type Sh.
  ## This is the only compiler-enforced member. Any user-defined type
  ## with a matching apply is an epilogue; the shipped ops are examples,
  ## not a closed zoo.
  ##
  ## Contract (structural, see the module header):
  ##   `template preflight(op: var Self): untyped`
  ##     A template that injects the staging buffer into the caller scope
  ##     ({.inject.}, shared on GPU) and stages the op's gmem operands.
  ##   gemm_tiled calls `op.preflight()` before `op.apply(D, AB)`
  ##   a missing `preflight` is a compile error at that call site.
  ##
  ## Target contract (design doc, CUTLASS correspondence in the scratchpad):
  ##   `shard(op, D, tD, tDv)`   partition the captured operands by thread,
  ##                             projecting onto the output fragment the
  ##                             kernel already partitioned
  ##   `preflight(op)`           initiate async data movement (no-op today)
  ##   `apply(op, AB, ...)`      the math — STAYS a concept member
  ##   `finalStore(op, out, D)`  the store, separate from the math
  proc apply(op: Self, D: var (TensorView or Tensor), AB: TensorView or Tensor)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAXPBY: D = α·AB + β·C
# ═════════════════════════════════════════════════════════════════════════

type EpiAXPBY*[T, Sh, StC] = object
  ## Linear combination of the accumulator and the C operand.
  ## D = α·AB + β·C
  alpha, beta: T
  C_gmem: TensorView[T, Sh, StC]
  # C_smem: ptr UncheckedArray[T]
  #   Future: cp.async / TMA smem staging of C (LoadKind ikTMAStaged,
  #   GEMM-ARCHITECTURE.md §4). For now C is read per-thread from gmem in
  #   `apply`, the CUTLASS LinearCombination default: no staging, no race.

func initEpiAXPBY*[T, Sh, StC](
    alpha: T;
    beta: T;
    C: TensorView[T, Sh, StC]): EpiAXPBY[T, Sh, StC] =
  ## Build the op with its C operand. The C view is the same shape as
  ## the output tile (shared Sh, enforced by the compiler).
  result = EpiAXPBY[T, Sh, StC](alpha: alpha, beta: beta, C_gmem: C)

func cView*[T, Sh, StC](op: var EpiAXPBY[T, Sh, StC]): var TensorView[T, Sh, StC] =
  ## The C operand view in gmem (the destination D), writable by apply.
  op.C_gmem

template preflight*[T, Sh, StC](op: var EpiAXPBY[T, Sh, StC]): untyped =
  ## No-op: C is read per-thread from gmem in `apply` (CUTLASS
  ## LinearCombination default: direct register→gmem, see
  ## GEMM-ARCHITECTURE.md §4). cp.async / TMA smem staging is pending;
  ## when it lands, this template injects the {.shared.} staging buffer
  ## into the caller scope and fills it here.
  discard

func apply*[T, Sh, StD, StAB, StC](
    op: EpiAXPBY[T, Sh, StC];
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = α·AB + β·C, element-wise over the D tile's (M, N) shape.
  ## Dispatch hoisted out of the loops (four flat paths):
  ##   β == 0 → C is never read, this saves memory bandwidth
  ##   α == 1 → the α multiply is skipped
  ## On GPU the branches are uniform
  ##   α/β are identical for every thread
  ##   no warp divergence, the cost is on instruction cache / code size
  let S = size(D.layout)
  if op.beta == T(0):
    if op.alpha == T(1):
      for i in 0 ..< S:
        D(i) = AB(i)
    else:
      for i in 0 ..< S:
        D(i) = op.alpha * AB(i)
  elif op.alpha == T(1):
    for i in 0 ..< S:
      # This is a single FMA instruction,
      # specializing for β == doesn't gain performance
      D(i) = AB(i) + op.beta * op.C_gmem(i)
  else:
    for i in 0 ..< S:
      D(i) = op.alpha * AB(i) + op.beta * op.C_gmem(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity: D = AB (α=1, β=0, compile-time constants)
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity epilogue: D = AB, C ignored. Used when the GEMM has no C
  ## operand. C stays in the signature (the concept is uniform) but is
  ## never read.

template preflight*(op: var EpiIdentity): untyped =
  ## EpiIdentity `preflight` is a no-op
  discard

func apply*[T, Sh, StD, StAB](
    op: EpiIdentity;
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  for i in 0 ..< size(D.layout):
    D(i) = AB(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias: D = AB + bias (column broadcast)
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias*[T, Sh, St] = object
  ## D = AB + bias.
  ## Bias is a column vector broadcasted onto AB
  bias_gmem: TensorView[T, Sh, St]
  # bias_smem: ptr UncheckedArray[T]
  #   Future: cp.async / TMA smem staging of the bias (LoadKind ikTMAStaged,
  #   GEMM-ARCHITECTURE.md §4). For now bias is read per-thread from gmem in
  #   `apply`, the CUTLASS EpilogueWithBroadcast default: no staging, no race.

func initEpiAddBias*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiAddBias[T, Sh, St] =
  result.bias_gmem = bias

template preflight*[T, Sh, St](op: var EpiAddBias[T, Sh, St]): untyped =
  ## No-op: bias is read per-thread from gmem in `apply` (CUTLASS
  ## EpilogueWithBroadcast default: direct register→gmem, see
  ## GEMM-ARCHITECTURE.md §4). cp.async / TMA smem staging is pending;
  ## when it lands, this template injects the {.shared.} staging buffer
  ## into the caller scope and fills it here.
  discard

func apply*[T, Sh, StD, StAB, StB](
    op: EpiAddBias[T, Sh, StB];
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  for i in 0 ..< size(D.layout):
    D(i) = AB(i) + op.bias_gmem(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU: D = max(0, AB)
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB)

template preflight*(op: var EpiReLU): untyped =
  ## Nothing to stage: ReLU has no inputs.
  discard

func apply*[T, Sh, StD, StAB](
    op: EpiReLU;
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  for i in 0 ..< size(D.layout):
    D(i) = max(AB(i), T(0))
