# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## rmsnorm proof-kernel pair: backward signature and backward structure
##
## Test that `bkMetal.bwd_diff` expands the plain-form rmsnorm proof
## kernel into a const pair carrying the mechanically derived backward
## signature plus a self-contained backward kernel:
##
## - the backward signature is `d_y, d_x, d_w, d_rrms` plus the forward signature verbatim, `rrms` at its forward position
## - `rrms` records `pkResidual`, its gradient is an incoming cotangent
## - the backward recomputes the statistic from the primals, so the printer prunes `y` and `rrms` from the printed parameters
## - every cotangent re-derives the per-row statistic, `d_w` accumulates per column across the row loop
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/autodiff --nimcache:nimcache/tests/autodiff \
##     workspace/crucible/tests/codegen/autodiff/t_bwd_diff_entry_rmsnorm_pair.nim

import std/strutils
import workspace/crucible

# ── Proof kernel ──────────────────────────────────────────────────────────

proc rmsnormPlain(y, x, w, rrms: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## One thread normalizes two rows of eight columns:
  ## `y[r][i] = x[r][i] * rrms[r] * w[i]` with `rrms[r] = rsqrt(mean(x²) + eps)`.
  ## The statistic exports into `rrms`, the backward re-derives its value
  ## from the primals.
  for r in 0 ..< 2:
    let b = r * 8
    let ssq = x[b + 0] * x[b + 0] + x[b + 1] * x[b + 1] +
              x[b + 2] * x[b + 2] + x[b + 3] * x[b + 3] +
              x[b + 4] * x[b + 4] + x[b + 5] * x[b + 5] +
              x[b + 6] * x[b + 6] + x[b + 7] * x[b + 7]
    let inv = rsqrt(ssq / 8.0'f32 + 1.0e-6'f32)
    rrms[r] = inv
    y[b + 0] = x[b + 0] * inv * w[0]
    y[b + 1] = x[b + 1] * inv * w[1]
    y[b + 2] = x[b + 2] * inv * w[2]
    y[b + 3] = x[b + 3] * inv * w[3]
    y[b + 4] = x[b + 4] * inv * w[4]
    y[b + 5] = x[b + 5] * inv * w[5]
    y[b + 6] = x[b + 6] * inv * w[6]
    y[b + 7] = x[b + 7] * inv * w[7]

const rmsnormPair = bkMetal.bwd_diff(rmsnormPlain)

# ── Pair metadata: mirror ABI and residual slot ───────────────────────────

func paramNamed(ad: KernelPair, name: string): BwdParamMeta =
  for m in ad.params:
    if m.name == name:
      return m
  doAssert false, "no parameter named " & name

static:
  doAssert rmsnormPair.kernelName == "rmsnormPlain"
  doAssert rmsnormPair.backend == bkMetal

  # Backward signature: the d_-block lists the gradient of every forward
  # parameter in forward order, block 2 is the forward signature verbatim.
  # `rrms` keeps its forward position (4th parameter).
  doAssert rmsnormPair.bwdSignature == @["d_y", "d_x", "d_w", "d_rrms",
                                         "y", "x", "w", "rrms"]

  let yMeta = rmsnormPair.paramNamed("y")
  doAssert yMeta.kind == pkResidual
  doAssert yMeta.gradKind == gkBuffer and yMeta.gradName == "d_y"

  # The exported statistic: a residual at its forward position whose
  # gradient is an incoming cotangent slot. A buffer residual carries no scalar workspace, so no reduce kernel exists
  let rrmsMeta = rmsnormPair.paramNamed("rrms")
  doAssert rrmsMeta.kind == pkResidual
  doAssert rrmsMeta.gradKind == gkBuffer and rrmsMeta.gradName == "d_rrms"

  for name in ["x", "w"]:
    let m = rmsnormPair.paramNamed(name)
    doAssert m.kind == pkBuffer, name
    doAssert m.gradKind == gkBuffer and m.gradName == "d_" & name
    doAssert m.extent == xeWrapperField, name

  doAssert rmsnormPair.scalarWorkspaces.len == 0
  doAssert rmsnormPair.reduceSources.len == 0

# ── Backward kernel structure ─────────────────────────────────────────────

static:
  doAssert "kernel void rmsnormPlain_bwd(" in rmsnormPair.bwdSource

  # The backward recomputes the statistic from the primals.
  # The printed parameter list keeps the cotangents plus the primals the kernel reads.
  # No cotangent reads `y` or `rrms`, so the printer prunes them
  doAssert rmsnormPair.bwdKernelParams == @["d_y", "d_x", "d_w", "d_rrms", "x", "w"]

  # The statistic value re-derives inside every cotangent: the primal
  # `rsqrt` call and the `sqrt` of the rsqrt cotangent rule both re-emit
  doAssert "rsqrt(" in rmsnormPair.bwdSource
  doAssert "sqrt(" in rmsnormPair.bwdSource

  # d_w accumulates per column across the row loop
  doAssert "for(int r = 0; r < 2; r += 1)" in rmsnormPair.bwdSource
  doAssert "d_w[" in rmsnormPair.bwdSource
  doAssert "d_x[" in rmsnormPair.bwdSource

  # The forward source keeps the exported statistic store
  doAssert "kernel void rmsnormPlain(" in rmsnormPair.fwdSource
  doAssert "rrms[r]" in rmsnormPair.fwdSource

echo "  OK — rmsnorm pair: mirror ABI, rrms residual at its forward position, backward recomputes the statistic"
