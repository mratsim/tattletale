# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Autograd: the context record is the tape.
##
## Lifecycle of one training step, one record throughout:
##
##   forward ops --> backward(loss) appends grad entries --> eval() drains pending --> grad(t) --> resetTape()
##
## backward() only enqueues, it never computes. Before the first eval, forward and grad entries drain together.
## After one eval the walk reads the drained forward entries and the grad entries drain next.
##
## The vjp of each op builds on the no-grad erased primitives, dispatched on the entry's kernel id:
##
## - add: identity, output cotangent = input cotangent (aliased, no op)
## - mul: cotangent of each operand is the output cotangent times the sibling operand
## - mulScalar: cotangent of the operand is the output cotangent times the recorded scalar
##
## - reshape: the cotangent buffer reinterpreted over the input's layout, no op
## - permute: a metadata view over the cotangent buffer with inverse-permuted strides
## - slice, gather: scatter kernels into a zeroed gradient (k_slice_bwd, k_gather_bwd)
##
## - softmax, gelu, layernorm, cross entropy: fused backward kernels, one output per entry
## - where: composed from the erased primitives, branch indicators mask the operand cotangents
##   and the float mask takes the straight-through subgradient (a - b) * g
##
## - sgd, adam: terminal, in-place parameter updates have no vjp
## - seed: a fill of ones over the loss shape
##
## Cotangent accumulation per tensor id:
##
## - the first consumer's contribution writes the entry
## - a later consumer's contribution adds through addNoGrad
## - a tensor without the grad flag takes no contribution, so an op whose inputs all lack the flag produces no grad entries
## - grad-chain outputs carry the grad flag off, so the walk never differentiates a grad

import std/tables

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/tape,
  workspace/tensorcery/src/kernels/k_mul,
  workspace/tensorcery/src/kernels/k_shape,
  workspace/tensorcery/src/kernels/k_layernorm,
  workspace/tensorcery/src/tensors

proc accumulate(ctx: Ctx, cotan: var Table[int, AnyTensorImpl], tid: int, contribution: AnyTensorImpl) =
  ## Adds one vjp contribution into the cotangent of tensor id `tid`.
  ## The first contribution writes the entry, later ones add through addNoGrad.
  if tid in cotan:
    cotan[tid] = cotan[tid].addNoGrad(contribution)
  else:
    cotan[tid] = contribution

proc backward*(loss: Tensor) =
  ## Appends the grad segment of `loss` to the context record.
  ##
  ## The cotangent of the loss tensor is all ones, so backward differentiates the sum
  ## of the loss tensor's elements, a scalar loss gives the usual scalar semantics.
  ## Gradients materialize on the next eval, or on reading them through grad().
  ##
  ## Raises AssertionDefect when `loss` does not require grad.
  doAssert loss.getRequiresGrad(),
      "backward: the loss tensor does not require grad, nothing to differentiate"

  let ctx = loss.getCtx()
  var cotan: Table[int, AnyTensorImpl]

  # seed: d(sum(loss))/d(loss) is all ones over the loss shape
  cotan[loss.getImpl().getTid()] = loss.getImpl().onesNoGrad()

  for i in countdown(ctx.record.len - 1, 0):
    let entry = ctx.record[i]
    if entry.output notin cotan:
      continue
    let g = cotan[entry.output]

    case entry.kernel
    of opAdd:
      # vjp is the identity on each input slot. A tensor that several slots consume
      # accumulates once per slot through the loop.
      for tid in entry.inputs:
        if tid != 0 and tapeGet(ctx, tid).getRequiresGrad():
          accumulate(ctx, cotan, tid, g)
    of opMul:
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        let args = decodeMulArgs(entry.blob)
        accumulate(ctx, cotan, tid, g.mulScalarNoGrad(float64(args.scalars.scalar)))
    of opMul2:
      let tidA = entry.inputs[0]
      let tidB = entry.inputs[1]
      let a = tapeGet(ctx, tidA)
      let b = tapeGet(ctx, tidB)
      if a.getRequiresGrad():
        accumulate(ctx, cotan, tidA, g.mulNoGrad(b))
      if b.getRequiresGrad():
        accumulate(ctx, cotan, tidB, g.mulNoGrad(a))
    of opFill:
      # a fill output feeds only the cotangent map, no vjp runs on it
      discard
    of opReshape:
      # vjp is the identity as a metadata view over the input's layout.
      # A reshape keeps the logical element order and its input is contiguous, so the view
      # reinterprets the cotangent buffer over the input's shape and strides.
      # A strided cotangent (a shape op consumed the reshape output) has no stride
      # image for the regrouped axes and materializes first. A zero add copies it
      # into a fresh contiguous buffer.
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        var contrib = g
        if not isContiguous(g.getShape(), g.getStrides()):
          contrib = g.addNoGrad(g.onesNoGrad().mulScalarNoGrad(0))
        accumulate(ctx, cotan, tid, contrib.aliasAs(tapeGet(ctx, tid)))
    of opPermute:
      # vjp is a metadata view over the cotangent buffer with the input's shape
      # and the cotangent strides permuted by the inverse permutation. Output axis i
      # reads source axis perm[i], so input axis j reads cotangent axis
      # inv[j] with inv[perm[i]] = i.
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        let img = decodeShapeBlob(entry.blob)
        let rank = int(img.rank)
        var inv: array[4, int32]
        for i in 0 ..< rank:
          inv[int(img.a[i])] = int32(i)
        let gs = g.getStrides()
        var strides: array[4, int32]
        for j in 0 ..< rank:
          strides[j] = gs[inv[j]]
        let gStrides = SmallSeq[4, int32].initFrom(strides.toOpenArray(0, rank - 1))
        accumulate(ctx, cotan, tid,
            g.aliasStrided(tapeGet(ctx, tid).getShape(), gStrides))
    of opSlice:
      # vjp scatters the view's cotangent into a fresh zeroed gradient over the parent shape,
      # one entry per cotangent element, positions from the recorded slice starts.
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        let img = decodeShapeBlob(entry.blob)
        var starts: array[4, int32]
        for i in 0 ..< int(img.rank):
          starts[i] = img.a[i]
        let startSeq = SmallSeq[4, int32].initFrom(starts.toOpenArray(0, int(img.rank) - 1))
        accumulate(ctx, cotan, tid, tapeGet(ctx, tid).sliceBwdNoGrad(
            g, tapeGet(ctx, entry.output), startSeq))
    of opGather:
      # vjp scatter-adds the output cotangent into the table gradient. Each index entry
      # adds to one table entry. The int32 indices never require grad.
      let tableTid = entry.inputs[0]
      let idxTid = entry.inputs[1]
      if tapeGet(ctx, tableTid).getRequiresGrad():
        accumulate(ctx, cotan, tableTid, tapeGet(ctx, tableTid).gatherBwdNoGrad(
            g, tapeGet(ctx, idxTid)))
    of opLayerNorm:
      let xTid = entry.inputs[0]
      let wTid = entry.inputs[1]
      let bTid = entry.inputs[2]
      let x = tapeGet(ctx, xTid)
      let w = tapeGet(ctx, wTid)
      let b = tapeGet(ctx, bTid)
      let eps = float64(decodeLayerNormArgs(entry.blob).scalars.eps)
      if x.getRequiresGrad():
        accumulate(ctx, cotan, xTid, x.layernormBwdDxNoGrad(g, w, eps))
      if w.getRequiresGrad():
        accumulate(ctx, cotan, wTid, w.layernormBwdWNoGrad(g, x, eps))
      if b.getRequiresGrad():
        accumulate(ctx, cotan, bTid, b.layernormBwdBNoGrad(g))
    of opGelu:
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        accumulate(ctx, cotan, tid, tapeGet(ctx, tid).geluBwdNoGrad(g))
    of opSoftmax:
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        accumulate(ctx, cotan, tid,
            tapeGet(ctx, tid).softmaxBwdNoGrad(g, tapeGet(ctx, entry.output)))
    of opCrossEntropy:
      let logitsTid = entry.inputs[0]
      let tgtTid = entry.inputs[1]
      if tapeGet(ctx, logitsTid).getRequiresGrad():
        accumulate(ctx, cotan, logitsTid,
            tapeGet(ctx, logitsTid).crossEntropyBwdNoGrad(g, tapeGet(ctx, tgtTid)))
    of opWhere:
      # vjp composed from the erased primitives. The select branches on cond != 0,
      # so each operand cotangent is masked by its branch indicator: da = g * step(cond),
      # db = g * (1 - step(cond)) with step(cond) built through a where over a ones fill
      # and its zero. The cotangent of the float mask takes the straight-through value (a - b) * g.
      let condTid = entry.inputs[0]
      let aTid = entry.inputs[1]
      let bTid = entry.inputs[2]
      let cond = tapeGet(ctx, condTid)
      let a = tapeGet(ctx, aTid)
      let b = tapeGet(ctx, bTid)
      if a.getRequiresGrad() or b.getRequiresGrad():
        let ones = cond.onesNoGrad()
        let zeros = ones.mulScalarNoGrad(0)
        let step = ones.where(cond, zeros)
        if a.getRequiresGrad():
          accumulate(ctx, cotan, aTid, g.mulNoGrad(step))
        if b.getRequiresGrad():
          accumulate(ctx, cotan, bTid,
              g.mulNoGrad(ones.addNoGrad(step.mulScalarNoGrad(-1))))
      if cond.getRequiresGrad():
        let diff = a.addNoGrad(b.mulScalarNoGrad(-1))
        accumulate(ctx, cotan, condTid, g.mulNoGrad(diff))
    of opSgd, opAdam:
      # terminal ops: in-place parameter updates have no vjp, parameters are leaves
      discard
    of opSoftmaxBwd, opGeluBwd, opLayerNormBwdDx, opLayerNormBwdW, opLayerNormBwdB,
        opCrossEntropyBwd, opGatherBwd, opSliceBwd:
      # grad-segment entries: the walk never differentiates a grad, and their outputs
      # are consumed by nothing
      discard

  for tid, g in cotan:
    ctx.setGrad(tid, g)

proc grad*(t: Tensor): Tensor =
  ## Returns the gradient backward() recorded for `t`, over the same shape and device.
  ## The gradient tensor is lazy like any other: it materializes on the next eval.
  ##
  ## Raises AssertionDefect when backward() recorded no gradient for `t`, naming the tensor id.
  let ctx = t.getCtx()
  doAssert tape.hasGrad(ctx, t.getImpl().getTid()),
      "grad: no gradient recorded for tensor " & $t.getImpl().getTid() &
      ", backward did not run or the tape was reset"
  toTensor(ctx, tape.getGrad(ctx, t.getImpl().getTid()), t.getDtype(), t.getDevice())

proc resetTape*(ctx: Ctx) =
  ## Clears the record and the gradient state: the training-step boundary.
  ## Op ids and tensor ids keep counting across resets.
  ctx.reset()
  tape.clear(ctx)
