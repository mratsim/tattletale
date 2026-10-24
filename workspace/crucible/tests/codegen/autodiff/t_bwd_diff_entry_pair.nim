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

proc rsqrt1(y, a: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    y[i] = rsqrt(a[i])

proc stat2(`out`, aux, a: ptr UncheckedArray[float32]) {.global.} =
  # Stub exporting two forward outputs: the residual-slot fixture.
  for i in 0 ..< 8:
    `out`[i] = a[i] * 2.0'f32
    aux[i] = a[i] + 1.0'f32

proc axpy_bwd(d_out, d_a, d_b, d_alpha: ptr UncheckedArray[float32], `out`, a, b: ptr UncheckedArray[float32], alpha: float32) {.global.} =
  # Hand-written mirror of the transposed axpy backward, see the transposed
  # pair's assertions below: same signature, d_alpha accumulated in-kernel.
  for i in 0 ..< 8:
    d_a[i] += d_out[i] * alpha
    d_b[i] += d_out[i]
    d_alpha[0] += d_out[i] * a[i]

proc sdpa_fwd(q, k, v, `out`, lse: ptr UncheckedArray[float32]) {.global.} =
  # FA-shaped stub: five forward parameters, out and lse are forward outputs
  # (the save fixture). Trivial math, no SDPA.
  for i in 0 ..< 8:
    `out`[i] = q[i] + k[i] + v[i]
  lse[0] = `out`[0] + 1.0'f32

proc sdpa_fwd_bwd(d_q, d_k, d_v, d_out, d_lse: ptr UncheckedArray[float32], q, k, v, `out`, lse: ptr UncheckedArray[float32]) {.global.} =
  # Declared backward in mirror order, gradients first:
  # (d_q, d_k, d_v, d_out, d_lse) in forward-parameter order,
  # block 2 the forward signature verbatim.
  # Reads out and lse, the saved outputs, so the save wiring is observable.
  for i in 0 ..< 8:
    d_q[i] += d_out[i] * `out`[i] * 0.5'f32
    d_k[i] += d_out[i] * lse[0]
    d_v[i] += d_out[i] + d_lse[0]
  d_lse[0] += d_out[0]

# ── Expansion ─────────────────────────────────────────────────────────────

const axpyGlsl = bkGlsl.bwd_diff(axpy)
const axpyMetal = bkMetal.bwd_diff(axpy)
const axpyNPair = bkMetal.bwd_diff(axpyN)
const axpySPair = bkMetal.bwd_diff(axpyS)
const stat2Pair = bkMetal.bwd_diff(stat2)
const rsqrtPair = bkMetal.bwd_diff(rsqrt1)
const axpyDeclared = bkMetal.bwd_diff(axpy, axpy_bwd)
const sdpaDeclared = bkMetal.bwd_diff(sdpa_fwd, sdpa_fwd_bwd,
                                      save = [out, lse])
const axpyDeclaredGlsl = bkGlsl.bwd_diff(axpy, axpy_bwd)

# ── Mirror-rule signature, verbatim ───────────────────────────────────────

static:
  doAssert axpyMetal.kernelName == "axpy"
  doAssert axpyMetal.bwdSignature == @["d_out", "d_a", "d_b", "d_alpha",
                                       "out", "a", "b", "alpha"]
  doAssert axpyGlsl.bwdSignature == axpyMetal.bwdSignature

# ── Parameter metadata ────────────────────────────────────────────────────

func paramNamed(ad: KernelPair, name: string): BwdParamMeta =
  for m in ad.params:
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

  # the math whitelist transposes at the entry: the rsqrt cotangent
  # re-emits sqrt, the primal callee, over the argument
  doAssert "rsqrt(" in rsqrtPair.fwdSource
  doAssert "sqrt(" in rsqrtPair.bwdSource
  doAssert "2.0f" in rsqrtPair.bwdSource

  # the forward source is unchanged by differentiation
  doAssert "alpha * a[i]" in axpyMetal.fwdSource
  doAssert "out[i]" in axpyMetal.fwdSource


# ── Declared backward: the escape hatch ───────────────────────────────────

static:
  # the transposer is skipped, the hand-written kernel is the backward,
  # and the mirror signature still derives from the forward symbol
  doAssert axpyDeclared.bwdSignature == axpyMetal.bwdSignature
  doAssert axpyDeclaredGlsl.bwdSignature == axpyMetal.bwdSignature
  doAssert axpyDeclared.kernelName == "axpy"
  doAssert "kernel void axpy_bwd(" in axpyDeclared.bwdSource
  doAssert "d_a[i] += (d_out[i] * alpha)" in axpyDeclared.bwdSource
  doAssert "d_alpha[0] += (d_out[i] * a[i])" in axpyDeclared.bwdSource
  doAssert axpyDeclared.fwdSource == axpyMetal.fwdSource,
    "the forward source is the same kernel the transposed pair prints"

  # no reduce machinery exists on the declared path.
  # The hand-written kernel accumulates the scalar cotangent itself.
  doAssert axpyDeclared.reduceSources.len == 0
  doAssert axpyDeclared.scalarWorkspaces.len == 0
  doAssert axpyDeclared.bwdSource notin axpyMetal.bwdSource

  # parameter metadata derives from the forward symbol as on the transposed path
  doAssert axpyDeclared.paramNamed("out").kind == pkResidual
  doAssert axpyDeclared.paramNamed("a").gradKind == gkBuffer
  doAssert axpyDeclared.paramNamed("alpha").gradKind == gkBuffer1
  doAssert axpyDeclared.paramNamed("alpha").gradName == "d_alpha"

  # FA-shaped stub: five forward parameters, the mirror order is
  # (d_q, d_k, d_v, d_out, d_lse) then the forward signature verbatim
  doAssert sdpaDeclared.bwdSignature == @["d_q", "d_k", "d_v", "d_out", "d_lse",
                                          "q", "k", "v", "out", "lse"]
  doAssert sdpaDeclared.paramNamed("out").kind == pkResidual
  doAssert sdpaDeclared.paramNamed("lse").kind == pkResidual
  doAssert sdpaDeclared.paramNamed("q").kind == pkBuffer
  doAssert sdpaDeclared.saveNames == @["out", "lse"]
  doAssert axpyDeclared.saveNames.len == 0
  doAssert stat2Pair.saveNames.len == 0
