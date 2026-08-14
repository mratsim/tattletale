## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Epilogue fusion operations for the tiled GEMM.
##
## An epilogue computes the output tile
##   D = f(AB)
## where
##   D   is the destination (the output tile in gmem, written per thread)
##   AB  is the accumulated GEMM result (the accumulator fragment)
##
## Each op is split into two concept members:
##   `apply(op, res, AB)`      the math, pure: writes the result fragment
##                            `res` (registers) from the accumulator and
##                            the op's operands. No store, no predication.
##   `finalStore(op, res, D)`  the store: writes the result fragment to
##                            the gmem destination. This is where
##                            ragged-tile predication and delayed stores
##                            (flash-attention-style) live. One shared
##                            implementation for every op (see below the
##                            concept).
##
## The op object is the epilogue configuration. It carries its operands
## as fields (EpiAXPBY's C, EpiAddBias's bias) and the store mask
## (storeMask, set by gemm_cta on ragged boundary tiles). `apply` reads
## the operands through the op-held gmem views.
##
## `apply` iterates the fragment size. For each flat index i it writes
## res(i) from AB(i) and the op's operands. Each tensor is indexed
## through its own layout, so operand ranks and strides are unconstrained
## and need not match res's.
##
## alpha and beta are runtime values. `apply` branches on them once,
## before the element loop, so the element loop has no branches on op
## state. On GPU the branches are uniform (alpha and beta are identical
## for every thread), so there is no warp divergence, the cost is on
## instruction cache / code size.
##
## Ragged boundary tiles (the problem M or N is not a multiple of the
## tile dims): gemm_cta computes the tile's valid extent, derives a
## store mask from it (cStoreMask), and stores it on the op. `finalStore`
## skips the masked-off elements, so the padding outside the problem is
## never written. The math is never predicated: the elements outside the
## valid extent are computed like any other (their A/B operands were
## zero-filled at the gmem → smem load) but not stored.
##
## ── Contract ──
## Each op must provide:
##   `preflight(op: var Self)`  a TEMPLATE that injects the staging buffer
##                            into the caller scope ({.inject.}, shared memory via {.shared.} on GPU)
##                            and stages the op's gmem operands into it
##                            (cp.async / TMA pending). All shipped ops
##                            have no-op stubs: operands are read
##                            per-thread from gmem in `apply` (direct
##                            register→gmem is the default. smem staging
##                            is future LoadKind work).
##
##   `apply(op, res, AB)`     The epilogue math, pure: writes the result
##                            fragment `res` (registers) from the
##                            accumulator `AB` and the op's operands.
##
##   `finalStore(op, res, D)` The store: writes the result fragment to
##                            the gmem destination, skipping the elements
##                            outside the tile's valid (M, N) extent
##                            (op.storeMask, see below). One shared
##                            implementation for every op.
##
## Store predication:
##   `storeMask*: int`, bit i set = fragment element i is inside the
##   valid (M, N) range of the tile and may be stored. Defaults to all
##   bits set (no predication). gemm_cta overwrites it on ragged
##   boundary tiles. `finalStore` skips the masked-off stores.
##
## res and AB share the shape type Sh: the compiler enforces equal shapes,
## a mismatched shape is a type error. They do not have to have the same types
## hence the epilogue can do type conversions.
##
## gemm_cta calls, in order:
##   `op.preflight()`           stages gmem operands (no-op stub today)
##   `op.apply(res, AB)`        the epilogue math (EpiAXPBY reads C via op.C_gmem)
##   `op.storeMask = ...`       gemm_cta derives the mask from the tile's valid extent
##   `op.finalStore(res, D)`    the store, predicated for ragged tiles
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
import ./atoms
import ./atoms_mma_partitioning

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Epilogue concept
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  ## The concept: an Epilogue computes the output tile with the two
  ## members `apply(res, AB)` (the math) and `finalStore(res, D)` (the
  ## store). res and AB share the shape type Sh.
  ## Any user-defined type with both members is an epilogue. The shipped
  ## ops are examples, not a closed zoo. `finalStore` is satisfied by a
  ## single generic implementation shared by every op (see below).
  ##
  ## Contract (structural, see the module header):
  ##   `template preflight(op: var Self): untyped`
  ##     A template that injects the staging buffer into the caller scope
  ##     ({.inject.}, shared on GPU) and stages the op's gmem operands.
  ##   gemm_cta calls `op.preflight()` before `op.apply(res, AB)`
  ##   a missing `preflight` is a compile error at that call site.
  ##
  ## Store predication (ragged tiles):
  ##   `storeMask*: int`, bit i set = fragment element i is inside the
  ##   valid (M, N) range of the tile and may be stored. Defaults to all
  ##   bits set (no predication). gemm_cta overwrites it on ragged
  ##   boundary tiles. `finalStore` skips the masked-off stores.
  proc apply(op: Self, res: var (TensorView or Tensor), AB: TensorView or Tensor)
  proc finalStore(op: Self, res: (TensorView or Tensor), D: var (TensorView or Tensor))

# ═════════════════════════════════════════════════════════════════════════
#  finalStore — the store, shared by every op
# ═════════════════════════════════════════════════════════════════════════

func finalStore*[T, Sh, StR, StD](
    op: Epilogue;
    res: (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD])) {.inline.} =
  ## Write the result fragment to the gmem destination, skipping the
  ## elements outside the tile's valid (M, N) extent (ragged boundary
  ## tiles): op.storeMask bit i set = element i may be stored.
  ##
  ## This is the store side of the epilogue, separate from the math
  ## (apply), and one implementation shared by every op: the op type is
  ## never used here, only its storeMask field.
  ## The mask is uniform across threads (gemm_cta computes it once per
  ## CTA), so the all-ones check below is a uniform branch and a full
  ## tile takes the straight copy path.
  let S = toIntVal(size(D.layout))
  if op.storeMask == (1 shl S) - 1:
    for i in 0 ..< S:
      D(i) = res(i)
  else:
    for i in 0 ..< S:
      if ((op.storeMask shr i) and 1) != 0:
        D(i) = res(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAXPBY: D = α·AB + β·C
# ═════════════════════════════════════════════════════════════════════════

type EpiAXPBY*[T, Sh, StC] = object
  ## Linear combination of the accumulator and the C operand.
  ## D = α·AB + β·C
  alpha, beta: T
  C_gmem: TensorView[T, Sh, StC]
  storeMask*: int = -1
    ## Store predication: bit i set = fragment element i is inside the
    ## valid (M, N) range of the tile and may be stored. All bits set by
    ## default (no predication). gemm_cta computes it for ragged boundary
    ## tiles. `finalStore` skips the masked-off stores.
  # C_smem: ptr UncheckedArray[T]
  #   Future: cp.async / TMA smem staging of C. For now C is read
  #   per-thread from gmem in `apply`: no staging, no race.

func initEpiAXPBY*[T, Sh, StC](
    alpha: T;
    beta: T;
    C: TensorView[T, Sh, StC]): EpiAXPBY[T, Sh, StC] =
  ## Build the op with its C operand. The C view is the same shape as
  ## the output tile (shared Sh, enforced by the compiler).
  ## The store mask starts all-ones (no predication): gemm_cta overwrites
  ## it for ragged boundary tiles.
  result = EpiAXPBY[T, Sh, StC](alpha: alpha, beta: beta, C_gmem: C, storeMask: -1)

template preflight*[T, Sh, StC](op: var EpiAXPBY[T, Sh, StC]): untyped =
  ## No-op: C is read per-thread from gmem in `apply` (direct
  ## register→gmem). cp.async / TMA smem staging is pending.
  ## when it lands, this template injects the {.shared.} staging buffer
  ## into the caller scope and fills it here.
  discard

func apply*[T, Sh, StAB, StC, StR](
    op: EpiAXPBY[T, Sh, StC];
    res: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = α·AB + β·C, element-wise over the tile's (M, N) shape.
  ## Writes the result fragment `res`, no gmem store (see finalStore).
  ## Dispatch hoisted out of the loop (four flat paths):
  ##   β == 0 → C is never read, this saves memory bandwidth
  ##   α == 1 → the α multiply is skipped
  ## On GPU the branches are uniform
  ##   α/β are identical for every thread
  ##   no warp divergence, the cost is on instruction cache / code size
  let S = size(res.layout)
  if op.beta == T(0):
    if op.alpha == T(1):
      for i in 0 ..< S:
        res(i) = AB(i)
    else:
      for i in 0 ..< S:
        res(i) = op.alpha * AB(i)
  elif op.alpha == T(1):
    for i in 0 ..< S:
      # A single FMA per element
      res(i) = AB(i) + op.beta * op.C_gmem(i)
  else:
    for i in 0 ..< S:
      res(i) = op.alpha * AB(i) + op.beta * op.C_gmem(i)


# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity: D = AB (α=1, β=0, compile-time constants)
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity epilogue: D = AB. Used when the GEMM has no C operand.
  storeMask*: int = -1
    ## Store predication bitmask (see EpiAXPBY.storeMask)

template preflight*(op: var EpiIdentity): untyped =
  ## EpiIdentity `preflight` is a no-op
  discard

func apply*[T, Sh, StAB, StR](
    op: EpiIdentity;
    res: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB, element-wise over the tile's (M, N) shape.
  for i in 0 ..< size(res.layout):
    res(i) = AB(i)


# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias: D = AB + bias (column broadcast)
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias*[T, Sh, St] = object
  ## D = AB + bias.
  ## Bias is a column vector broadcasted onto AB
  bias_gmem: TensorView[T, Sh, St]
  storeMask*: int = -1
    ## Store predication bitmask (see EpiAXPBY.storeMask)
  # bias_smem: ptr UncheckedArray[T]
  #   Future: cp.async / TMA smem staging of the bias. For now bias is
  #   read per-thread from gmem in `apply`: no staging, no race.

func initEpiAddBias*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiAddBias[T, Sh, St] =
  ## Build the op with its bias operand. The store mask starts all-ones
  ## (no predication): gemm_cta overwrites it for ragged boundary tiles.
  result.bias_gmem = bias
  result.storeMask = -1

template preflight*[T, Sh, St](op: var EpiAddBias[T, Sh, St]): untyped =
  ## No-op: bias is read per-thread from gmem in `apply` (direct
  ## register→gmem). cp.async / TMA smem staging is pending.
  ## when it lands, this template injects the {.shared.} staging buffer
  ## into the caller scope and fills it here.
  discard

func apply*[T, Sh, StAB, StB, StR](
    op: EpiAddBias[T, Sh, StB];
    res: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB + bias, element-wise over the tile's (M, N) shape.
  ## The bias is a column broadcast: its view has stride-0 rows.
  for i in 0 ..< size(res.layout):
    res(i) = AB(i) + op.bias_gmem(i)


# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU: D = max(0, AB)
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB)
  storeMask*: int = -1
    ## Store predication bitmask (see EpiAXPBY.storeMask)

template preflight*(op: var EpiReLU): untyped =
  ## Nothing to stage: ReLU has no inputs.
  discard

func apply*[T, Sh, StAB, StR](
    op: EpiReLU;
    res: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB), element-wise over the tile's (M, N) shape.
  for i in 0 ..< size(res.layout):
    res(i) = max(AB(i), T(0))

# ═════════════════════════════════════════════════════════════════════════
#  shard: the per-thread epilogue hook (gemm_gpu's epi.shard)
# ═════════════════════════════════════════════════════════════════════════
#
#  gemm_gpu receives the op with problem-level operand views.
#  It calls `epi.shard(tma, thr, mCTA, nCTA)` to project them onto the thread's fragment of the CTA's tile.
#  The shard is a template per op, resolved at the call site like preflight:
#  a missing shard is a compile error at gemm_gpu's call.

template shard*[T, ShC, StC](
    op: EpiAXPBY[T, ShC, StC];
    tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## The per-thread C fragment: partition the op's problem-level C view
  ## at the CTA's tile. The sharded op is the fragment-typed epilogue
  ## gemm_cta consumes.
  const tileM = tma.thrM * tma.atom.mnk.m
  const tileN = tma.thrN * tma.atom.mnk.n
  initEpiAXPBY(op.alpha, op.beta,
    tma.partition_C(thr, local_tile(op.C_gmem, (tileM, tileN), (mCTA, nCTA))))

template shard*[T, Sh, St](
    op: EpiAddBias[T, Sh, St];
    tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## The per-thread bias fragment: partition the op's problem-level bias view
  ## (stride-0 rows) at the CTA's tile.
  const tileM = tma.thrM * tma.atom.mnk.m
  const tileN = tma.thrN * tma.atom.mnk.n
  initEpiAddBias(
    tma.partition_C(thr, local_tile(op.bias_gmem, (tileM, tileN), (mCTA, nCTA))))

template shard*(op: EpiIdentity; tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## EpiIdentity has no operands: the op passes through.
  op

template shard*(op: EpiReLU; tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## EpiReLU has no operands: the op passes through.
  op

