# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## bwd_diff compile-time entry: const kernel pairs
##
## Test that `bkGlsl.bwd_diff` / `bkMetal.bwd_diff` expand a `{.global.}` kernel to a const `KernelPair` with mechanically derived artifacts:
##
## - the backward signature of axpy is `(d_out, d_a, d_b, d_alpha, out, a, b, alpha)` exactly
## - `d_alpha` records a 1-element gradient buffer, matched by name
## - buffer lengths record their source, forward outputs keep their block-2 position
## - both backend kinds yield independent artifacts, derived kernels named `<kernel>_bwd`
##
## Run:
##   nim c --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/autodiff --nimcache:nimcache/tests/autodiff \
##     workspace/crucible/tests/codegen/autodiff/t_bwd_diff_entry_pair.nim

import std/strutils
import workspace/crucible

# ── Fixtures ──────────────────────────────────────────────────────────────

proc axpy(`out`, a, b: ptr UncheckedArray[float32], alpha: float32) {.global.} =
  for i in 0 ..< 8:
    `out`[i] = alpha * a[i] + b[i]

proc axpyN(`out`, a, b: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
  for i in 0 ..< n:
    `out`[i] = alpha * a[i] + b[i]

proc axpyS(y, a: ptr array[8, float32]) {.global.} =
  for i in 0 ..< 8:
    y[i] = a[i] * 2.0'f32

proc stat2(`out`, aux, a: ptr UncheckedArray[float32]) {.global.} =
  # Stub exporting two forward outputs: the residual-slot fixture.
  for i in 0 ..< 8:
    `out`[i] = a[i] * 2.0'f32
    aux[i] = a[i] + 1.0'f32

# ── Expansion ─────────────────────────────────────────────────────────────

const axpyGlsl = bkGlsl.bwd_diff(axpy)
const axpyMetal = bkMetal.bwd_diff(axpy)
const axpyNPair = bkMetal.bwd_diff(axpyN)
const axpySPair = bkMetal.bwd_diff(axpyS)
const stat2Pair = bkMetal.bwd_diff(stat2)

# ── Mirror-rule signature, verbatim ───────────────────────────────────────

static:
  doAssert axpyMetal.kernelName == "axpy"
  doAssert axpyMetal.bwdSignature == @["d_out", "d_a", "d_b", "d_alpha",
                                       "out", "a", "b", "alpha"]
  doAssert axpyGlsl.bwdSignature == axpyMetal.bwdSignature

# ── Parameter metadata ────────────────────────────────────────────────────

func paramNamed(pair: KernelPair, name: string): BwdParamMeta =
  for m in pair.params:
    if m.name == name:
      return m
  doAssert false, "no parameter named " & name

static:
  # d_alpha: scalar cotangent, gradKind gkBuffer1, matched by name not type
  let alpha = axpyMetal.paramNamed("alpha")
  doAssert alpha.kind == pkScalar
  doAssert alpha.gradKind == gkBuffer1
  doAssert alpha.gradName == "d_alpha"
  doAssert "alpha" in axpyMetal.reduceSources[0].name
  doAssert axpyMetal.reduceSources[0].name.endsWith("axpy_bwd_d_alpha_reduce")
  doAssert "axpy_bwd_d_alpha_reduce" in axpyMetal.reduceSources[0].source

  # buffers: same-shape gradients, length source per parameter
  for name in ["out", "a", "b"]:
    let m = axpyMetal.paramNamed(name)
    doAssert m.gradKind == gkBuffer and m.gradName == "d_" & name
    doAssert m.extent == xeWrapperField, name & ": the kernel states no length"

  for name in ["out", "a", "b"]:
    let m = axpyNPair.paramNamed(name)
    doAssert m.extent == xeLenParam and m.extentParam == "n", name

  for name in ["y", "a"]:
    let m = axpySPair.paramNamed(name)
    doAssert m.extent == xeStaticShape and m.staticLen == 8, name

  # forward outputs: residual kind, forward positions in block 2
  doAssert stat2Pair.paramNamed("out").kind == pkResidual
  doAssert stat2Pair.paramNamed("aux").kind == pkResidual
  doAssert stat2Pair.paramNamed("a").kind == pkBuffer
  doAssert stat2Pair.bwdSignature == @["d_out", "d_aux", "d_a", "out", "aux", "a"]
  # block 2 starts after the d_-block, the residuals keep their forward order
  let block2 = stat2Pair.bwdSignature[3 ..< stat2Pair.bwdSignature.len]
  doAssert block2 == @["out", "aux", "a"]

# ── Pair artifacts ────────────────────────────────────────────────────────

static:
  doAssert axpyMetal.backend == bkMetal
  doAssert axpyGlsl.backend == bkGlsl
  doAssert axpyMetal.fwdSource != axpyGlsl.fwdSource,
    "the two backend kinds yield independent artifacts"

  # Metal sources: one standalone kernel each, derived name, no collision
  doAssert "kernel void axpy(" in axpyMetal.fwdSource
  doAssert "_bwd" notin axpyMetal.fwdSource
  doAssert "kernel void axpy_bwd(" in axpyMetal.bwdSource
  doAssert "d_out" in axpyMetal.bwdSource and "d_a" in axpyMetal.bwdSource
  doAssert "d_out" in axpyGlsl.bwdSource
  doAssert "axpy_bwd" in axpyGlsl.bwdSource

  # the forward source is unchanged by differentiation
  doAssert "alpha * a[i]" in axpyMetal.fwdSource
  doAssert "out[i]" in axpyMetal.fwdSource
