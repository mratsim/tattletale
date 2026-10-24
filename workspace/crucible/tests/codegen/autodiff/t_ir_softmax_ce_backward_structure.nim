# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## softmax cross-entropy backward structure at IR level
##
## The loop-free softmax-CE kernel `loss[0] = mean(log(sum(exp(l-m)) + m - t*l))`
## over two rows of eight logits expands through `bkMetal.bwd_diff` into the pair checked below:
##
## - zero scalar workspaces and zero reduce kernels, the 1-element loss broadcast into every read buffer's cotangent
## - the backward signature carries the d_-block in forward order plus the forward signature verbatim
## - the printed backward carries the (d_loss/2) broadcast, the argmax-gated max scatter, the sumexp accumulator, and the target terms
## - the printed forward calls the backend `log` builtin for Nim's `ln`, mapped at print time
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/autodiff --nimcache:nimcache/tests/autodiff \
##     workspace/crucible/tests/codegen/autodiff/t_ir_softmax_ce_backward_structure.nim

import std/[math, strutils]
import workspace/crucible

# ── Proof kernel ──────────────────────────────────────────────────────────

proc softmaxCEPlain(loss, logits, target, rowMax, rowSumexp: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## One thread scores two rows of eight logits, `m` the per-row max.
  ## `loss[0] = mean(ln(sum(exp(l - m)) + m - t*l))`, statistics export into `rowMax`/`rowSumexp`.
  # Row 0: the max as a balanced tree of single-assignment ternaries
  # (max() is a refused call, loop-carried accumulation refuses too).
  let m01 = if logits[0] > logits[1]: logits[0] else: logits[1]
  let m23 = if logits[2] > logits[3]: logits[2] else: logits[3]
  let m45 = if logits[4] > logits[5]: logits[4] else: logits[5]
  let m67 = if logits[6] > logits[7]: logits[6] else: logits[7]
  let m0123 = if m01 > m23: m01 else: m23
  let m4567 = if m45 > m67: m45 else: m67
  let m0 = if m0123 > m4567: m0123 else: m4567
  # Row 0: sumexp as an unrolled sum of eight shifted exponentials.
  let e0 = exp(logits[0] - m0)
  let e1 = exp(logits[1] - m0)
  let e2 = exp(logits[2] - m0)
  let e3 = exp(logits[3] - m0)
  let e4 = exp(logits[4] - m0)
  let e5 = exp(logits[5] - m0)
  let e6 = exp(logits[6] - m0)
  let e7 = exp(logits[7] - m0)
  let s0 = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7
  rowMax[0] = m0
  rowSumexp[0] = s0
  # Row 0: the loss term, an unrolled dot against the one-hot target.
  let lt0 = target[0]*logits[0] + target[1]*logits[1] + target[2]*logits[2] +
            target[3]*logits[3] + target[4]*logits[4] + target[5]*logits[5] +
            target[6]*logits[6] + target[7]*logits[7]
  let ce0 = ln(s0) + m0 - lt0
  # Row 1, same shape over columns 8..15.
  let n01 = if logits[8] > logits[9]: logits[8] else: logits[9]
  let n23 = if logits[10] > logits[11]: logits[10] else: logits[11]
  let n45 = if logits[12] > logits[13]: logits[12] else: logits[13]
  let n67 = if logits[14] > logits[15]: logits[14] else: logits[15]
  let n0123 = if n01 > n23: n01 else: n23
  let n4567 = if n45 > n67: n45 else: n67
  let n0 = if n0123 > n4567: n0123 else: n4567
  let f0 = exp(logits[8] - n0)
  let f1 = exp(logits[9] - n0)
  let f2 = exp(logits[10] - n0)
  let f3 = exp(logits[11] - n0)
  let f4 = exp(logits[12] - n0)
  let f5 = exp(logits[13] - n0)
  let f6 = exp(logits[14] - n0)
  let f7 = exp(logits[15] - n0)
  let s1 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7
  rowMax[1] = n0
  rowSumexp[1] = s1
  let lt1 = target[8]*logits[8] + target[9]*logits[9] + target[10]*logits[10] +
            target[11]*logits[11] + target[12]*logits[12] +
            target[13]*logits[13] + target[14]*logits[14] +
            target[15]*logits[15]
  let ce1 = ln(s1) + n0 - lt1
  loss[0] = (ce0 + ce1) / 2.0'f32

const softmaxCEPair = bkMetal.bwd_diff(softmaxCEPlain)

# ── Pair metadata ─────────────────────────────────────────────────────────

static:
  doAssert softmaxCEPair.kernelName == "softmaxCEPlain"
  doAssert softmaxCEPair.backend == bkMetal

  # Backward signature: the d_-block in forward order, block 2 verbatim.
  doAssert softmaxCEPair.bwdSignature == @[
    "d_loss", "d_logits", "d_target", "d_rowMax", "d_rowSumexp",
    "loss", "logits", "target", "rowMax", "rowSumexp"]

  # No float scalar parameter means no scalar cotangent, hence no reduce-partials workspace and no reduce kernel.
  # The d_loss broadcast needs neither.
  doAssert softmaxCEPair.scalarWorkspaces.len == 0
  doAssert softmaxCEPair.reduceSources.len == 0

  # The backward reads the gradient shadows and the logits/target primals only.
  # The printer prunes the unread primals, the statistics' cotangents stay bound.
  doAssert softmaxCEPair.bwdKernelParams == @[
    "d_loss", "d_logits", "d_target", "d_rowMax", "d_rowSumexp", "logits", "target"]

  # The forward writes loss[0] plus the per-row statistics exports.
  doAssert "loss[0] = ((ce0 + ce1) / 2.0f);" in softmaxCEPair.fwdSource
  doAssert "rowMax[0] = m0;" in softmaxCEPair.fwdSource
  doAssert "rowMax[1] = n0;" in softmaxCEPair.fwdSource
  doAssert "rowSumexp[0] = s0;" in softmaxCEPair.fwdSource
  doAssert "rowSumexp[1] = s1;" in softmaxCEPair.fwdSource

# ── Backward structure: the (d_loss/2) broadcast and the one-hot terms ────

static:
  let bwd = softmaxCEPair.bwdSource

  # The natural-log cotangent is 1/sum, folded into the per-logit softmax terms `(d_loss/2)/sum * exp(l_k - m)`. No log call survives.
  doAssert "(d_loss[0] / 2.0f) / (" in bwd
  doAssert "ln(" notin bwd
  doAssert "log(" notin bwd

  # The max's direct cotangent scatters through the argmax ternary chain.
  doAssert "? d_rowMax[0] : 0.0f" in bwd
  doAssert "? d_rowMax[1] : 0.0f" in bwd

  # The sumexp accumulator's cotangent transposes into per-logit terms.
  doAssert "d_rowSumexp[0] * exp((logits[0]" in bwd
  doAssert "d_rowSumexp[1] * exp((logits[8]" in bwd

  # One-hot target, all 16 columns, minus sign:
  # `d_logits[k] -= (d_loss/2)*target[k]`, `d_target[j] -= (d_loss/2)*logits[j]`.
  for k in 0 ..< 16:
    doAssert "d_logits[" & $k & "] = (d_logits[" & $k &
      "] + (-(d_loss[0] / 2.0f) * target[" & $k & "]));" in bwd
    doAssert "d_target[" & $k & "] = (d_target[" & $k &
      "] + (-(d_loss[0] / 2.0f) * logits[" & $k & "]));" in bwd

  # The broadcast factor reaches both rows: 16 softmax terms,
  # the argmax-gated scatters, and the 32 target terms.
  doAssert bwd.count("d_loss[0] / 2.0f") >= 64

# ── Forward structure: the backend's natural-log name ─────────────────────

static:
  let fwd = softmaxCEPair.fwdSource
  # MSL has no `ln` builtin. The printer maps the Nim callee `ln`
  # onto the backend's natural-log builtin at print time.
  doAssert "log(s0)" in fwd
  doAssert "log(s1)" in fwd
  doAssert "ln(" notin fwd

echo "  OK — softmax-CE pair: zero reduce kernels, (d_loss/2) broadcast, one-hot target terms"
