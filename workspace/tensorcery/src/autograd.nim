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
## The vjp of each op builds on the no-grad erased primitives, dispatched on the entry's
## kernel id:
##
## - add: identity, output cotangent = input cotangent (aliased, no op)
## - mul: cotangent of each operand is the output cotangent times the sibling operand
## - mulScalar: cotangent of the operand is the output cotangent times the recorded scalar
## - seed: a fill of ones over the loss shape
##
## Cotangent accumulation per tensor id:
##
## - the first consumer's contribution writes the entry
## - a later consumer's contribution adds through addNoGrad
## - a tensor without the grad flag takes no contribution, so an op whose inputs all
##   lack the flag produces no grad entries
## - grad-chain outputs carry the grad flag off, so the walk never differentiates a grad

import std/tables

import
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/tape,
  workspace/tensorcery/src/kernels/k_mul,
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
    of opReshape, opPermute:
      # vjp is the identity as a metadata view: the cotangent storage reinterpreted
      # over the input's layout. No op records, no enqueue.
      let tid = entry.inputs[0]
      if tapeGet(ctx, tid).getRequiresGrad():
        accumulate(ctx, cotan, tid, g.aliasAs(tapeGet(ctx, tid)))
    of opSlice:
      doAssert false, "slice gradient arrives with the fused backward kernels"
    of opSgd, opAdam:
      # terminal ops: in-place parameter updates have no vjp, parameters are leaves
      discard
    of opGather, opLayerNorm, opGelu, opSoftmax, opCrossEntropy, opWhere:
      # microgpt op set kernels ship their vjps with their forward ops
      doAssert false, "op " & $entry.kernel & " reached the backward walk without a vjp"

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
