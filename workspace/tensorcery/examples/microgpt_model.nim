# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Microgpt on tensorcery: a GPT assembled from the supported op set only.
##
## Every linear maps through the gather-expand matmul ("EXP matmul"):
##
##   y[b,t,o] = sum_i x[b,t,i] * w[o,i]  # sum over the I axis
##   xexp = gather(reshape(x, [B*T, I]), idxRow)   # idxRow[b,t,o] = b*T+t
##   wexp = gather(w, idxCol)                      # idxCol[b,t,o] = o
##   y = pairwiseSumAxis(xexp * wexp, 3)  # reduce the last axis
##
## Notes:
## - gathers over constant index tensors broadcast both operands into one
##   rank-4 shape
## - strided views materialize through `x + (x * 0.0)` ahead of any reshape
## - attention runs over a batch-head axis (BH = B*nHead), one EXP matmul
##   computes the scores and one computes attention @ V
## - the causal mask is a constant 0/1 `where` over the scaled scores

import std/[math, random, sequtils]

import workspace/data_structures/src/small_seqs
import workspace/tensorcery
import workspace/tensorcery/src/tape

export tensorcery

type
  GptConfig* = object
    ## Model geometry. headDim * nHead must equal dModel.
    vocabSize*: int
    dModel*: int
    nHead*: int
    nLayer*: int
    blockSize*: int  ## context length T
    batchSize*: int

  GptParams* = object
    ## Trainable parameters and persistent adam moments, one Tensor per matrix.
    wte*, wpe*, lmHead*: Tensor
    layerWq*, layerWk*, layerWv*, layerWo*: seq[Tensor]
    layerFc1*, layerFc2*: seq[Tensor]
    layerLn1W*, layerLn1B*, layerLn2W*, layerLn2B*: seq[Tensor]
    lnfW*, lnfB*: Tensor
    mWte*, vWte*, mWpe*, vWpe*, mLmHead*, vLmHead*: Tensor
    mLnfW*, vLnfW*, mLnfB*, vLnfB*: Tensor
    layerM*: seq[array[10, Tensor]]  ## adam first moments, per layer
    layerV*: seq[array[10, Tensor]]  ## adam second moments, per layer

  GptConsts* = object
    ## Constant gather indices and mask tensors for one (batch, context) shape.
    ## Built once per backend and reused by every training step and generation pass.
    posIdx*: Tensor          ## int32 [B, T], value t
    linRowD*, linColD*: Tensor    ## int32 [B, T, D] for linears with D outputs
    linRow4D*, linCol4D*: Tensor  ## int32 [B, T, 4D] for the fc1 linear
    linRowV*, linColV*: Tensor    ## int32 [B, T, V] for the LM head
    bhIdx*: Tensor           ## int32 [BH, T], value bh
    attnIdx*: Tensor         ## int32 [BH, T, headDim], value bh*T + t
    mask*: Tensor            ## float32 [BH, T, T], 1.0 where key position j <= query i
    negBig*: Tensor          ## float32 [BH, T, T], -1e9 filler for masked scores

proc shapeDims(t: Tensor): array[4, int] =
  let sh = t.getShape()
  for i in 0 ..< int(sh.len):
    result[i] = int(sh[i])

proc fullRange(n: int): Slice[int] = 0 .. (n - 1)

proc pairwiseSumAxis(t: Tensor, axis: int): Tensor =
  # Reduces one rank-4 axis by adding contiguous halves, one add per halving:
  # [A, B, C, N] shrinks to axis size 1 in ceil(log2 N) adds, then the size-1
  # axis drops and the result is contiguous rank 3.
  doAssert t.getShape().len == 4, "pairwiseSumAxis: rank 4 required, got rank " & $t.getShape().len
  var cur = t
  var dims = cur.shapeDims()
  var n = dims[axis]
  while n > 1:
    let lo = n div 2
    var loSel, hiSel: array[4, Slice[int]]
    for a in 0 ..< 4:
      loSel[a] = fullRange(dims[a])
      hiSel[a] = fullRange(dims[a])
    loSel[axis] = fullRange(lo)
    hiSel[axis] = lo .. (n - 1)
    let a = cur.slice(loSel[0], loSel[1], loSel[2], loSel[3])
    let b = cur.slice(hiSel[0], hiSel[1], hiSel[2], hiSel[3])
    cur = a + b
    dims = cur.shapeDims()
    n = dims[axis]
  var outDims: array[3, int]
  var k = 0
  for a in 0 ..< 4:
    if a != axis:
      outDims[k] = dims[a]
      k.inc
  cur.reshape(outDims[0], outDims[1], outDims[2])

proc materialize(x: Tensor): Tensor =
  ## Contiguous copy of a strided view: reshape requires contiguous storage,
  ## and every recorded op output is fresh contiguous.
  x + (x * 0.0)

proc buildLinIdx(ctx: Ctx, Hw: static BackendKind, B, T, O: int): tuple[row, col: Tensor] =
  ## Constant gather indices that broadcast one linear's operands to [B, T, O, I]:
  ## - `row`: value b*T+t across the output axis, selecting the x sample of each
  ##   batch position from the flattened [B*T, I] input
  ## - `col`: 0 ..< O across the output axis, selecting the weight entries
  var row, col: seq[int32]
  for b in 0 ..< B:
    for t in 0 ..< T:
      for o in 0 ..< O:
        row.add int32(b * T + t)
        col.add int32(o)
  (toTensor(ctx, row, Hw, B, T, O), toTensor(ctx, col, Hw, B, T, O))

proc buildConsts*(ctx: Ctx, Hw: static BackendKind, cfg: GptConfig): GptConsts =
  ## Builds every constant the model forward reads, for one batch size.
  ## All tensors require no grad and stay identical across training steps.
  let B = cfg.batchSize
  let T = cfg.blockSize
  let D = cfg.dModel
  let H = cfg.nHead
  let HD = D div H
  let V = cfg.vocabSize
  let BH = B * H

  var pos: seq[int32]
  for b in 0 ..< B:
    for t in 0 ..< T:
      pos.add int32(t)
  result.posIdx = toTensor(ctx, pos, Hw, B, T)

  (result.linRowD, result.linColD) = buildLinIdx(ctx, Hw, B, T, D)
  (result.linRow4D, result.linCol4D) = buildLinIdx(ctx, Hw, B, T, 4 * D)
  (result.linRowV, result.linColV) = buildLinIdx(ctx, Hw, B, T, V)

  var bh: seq[int32]
  for b in 0 ..< BH:
    for t in 0 ..< T:
      bh.add int32(b)
  result.bhIdx = toTensor(ctx, bh, Hw, BH, T)

  var attn: seq[int32]
  for b in 0 ..< BH:
    for t in 0 ..< T:
      for d in 0 ..< HD:
        attn.add int32(b * T + t)
  result.attnIdx = toTensor(ctx, attn, Hw, BH, T, HD)

  var maskData, negData: seq[float32]
  for b in 0 ..< BH:
    for i in 0 ..< T:
      for j in 0 ..< T:
        maskData.add if j <= i: 1.0'f32 else: 0.0'f32
        negData.add -1e9'f32
  result.mask = toTensor(ctx, maskData, Hw, BH, T, T)
  result.negBig = toTensor(ctx, negData, Hw, BH, T, T)

proc expMatmul*(x, w: Tensor, idxRow, idxCol: Tensor): Tensor =
  ## y[b,t,o] = sum_i x[b,t,i] * w[o,i] from gather + elementwise mul + pairwise add.
  ##
  ## Expected input:
  ## - `x`: [B, T, I], contiguous
  ## - `w`: [O, I], contiguous, entry [o, i] maps output feature o to input i
  ## - `idxRow`, `idxCol`: int32 [B, T, O] constants from buildLinIdx
  ##
  ## Output: [B, T, O], contiguous.
  let dims = x.shapeDims()
  let xflat = x.reshape(dims[0] * dims[1], dims[2])
  let xexp = gather(xflat, idxRow)  # [B, T, O, I]
  let wexp = gather(w, idxCol)      # [B, T, O, I]
  pairwiseSumAxis(xexp * wexp, 3)

proc initGptParams*(ctx: Ctx, Hw: static BackendKind, cfg: GptConfig, std: float64): GptParams =
  ## Creates the model parameters and the adam moment buffers:
  ## - weights: gaussian with standard deviation `std`
  ## - layernorm scale/shift: ones / zeros
  ## - adam moments m and v: zeros, one pair per parameter, same shape
  ##
  ## Every learned tensor requires grad and the moments never do.
  let D = cfg.dModel
  let V = cfg.vocabSize
  let T = cfg.blockSize
  proc numel(dims: openArray[int]): int =
    result = 1
    for d in dims:
      result *= d

  proc gaussTensor(dims: varargs[int]): Tensor =
    # Learned parameter: gaussian weights, requires grad.
    var data = newSeq[float32](numel(dims))
    for i in 0 ..< data.len:
      data[i] = float32(gauss(0.0, std))
    toGradTensor(ctx, data, Hw, dims)

  proc onesParam(dims: varargs[int]): Tensor =
    # Learned layernorm scale, requires grad.
    toGradTensor(ctx, newSeqWith(numel(dims), 1'f32), Hw, dims)

  proc zerosParam(dims: varargs[int]): Tensor =
    # Learned layernorm shift, requires grad.
    toGradTensor(ctx, newSeqWith(numel(dims), 0'f32), Hw, dims)

  proc zerosMoment(dims: varargs[int]): Tensor =
    # Adam moment buffer: never requires grad.
    toTensor(ctx, newSeqWith(numel(dims), 0'f32), Hw, dims)

  result.wte = gaussTensor(V, D)
  result.wpe = gaussTensor(T, D)
  result.lmHead = gaussTensor(V, D)
  result.lnfW = onesParam(D)
  result.lnfB = zerosParam(D)
  result.mWte = zerosMoment(V, D)
  result.vWte = zerosMoment(V, D)
  result.mWpe = zerosMoment(T, D)
  result.vWpe = zerosMoment(T, D)
  result.mLmHead = zerosMoment(V, D)
  result.vLmHead = zerosMoment(V, D)
  result.mLnfW = zerosMoment(D)
  result.vLnfW = zerosMoment(D)
  result.mLnfB = zerosMoment(D)
  result.vLnfB = zerosMoment(D)
  for layer in 0 ..< cfg.nLayer:
    result.layerWq.add gaussTensor(D, D)
    result.layerWk.add gaussTensor(D, D)
    result.layerWv.add gaussTensor(D, D)
    result.layerWo.add gaussTensor(D, D)
    result.layerFc1.add gaussTensor(4 * D, D)
    result.layerFc2.add gaussTensor(D, 4 * D)
    result.layerLn1W.add onesParam(D)
    result.layerLn1B.add zerosParam(D)
    result.layerLn2W.add onesParam(D)
    result.layerLn2B.add zerosParam(D)
    var m, v: array[10, Tensor]
    for slot in 0 ..< 4:
      m[slot] = zerosMoment(D, D)
      v[slot] = zerosMoment(D, D)
    m[4] = zerosMoment(4 * D, D)
    v[4] = zerosMoment(4 * D, D)
    m[5] = zerosMoment(D, 4 * D)
    v[5] = zerosMoment(D, 4 * D)
    for slot in 6 ..< 10:
      m[slot] = zerosMoment(D)
      v[slot] = zerosMoment(D)
    result.layerM.add m
    result.layerV.add v

proc reRegister*(ctx: Ctx, params: GptParams, consts: GptConsts, cfg: GptConfig) =
  ## Re-registers the persistent tensors after resetTape:
  ## - the tape registry is cleared at the step boundary
  ## - backward() resolves every recorded tensor id through the registry,
  ##   constants included
  template reg(t: Tensor) =
    tape.register(ctx, t.getImpl().getTid(), t.getImpl())
  reg(params.wte)
  reg(params.wpe)
  reg(params.lmHead)
  reg(params.lnfW)
  reg(params.lnfB)
  reg(params.mWte)
  reg(params.vWte)
  reg(params.mWpe)
  reg(params.vWpe)
  reg(params.mLmHead)
  reg(params.vLmHead)
  reg(params.mLnfW)
  reg(params.vLnfW)
  reg(params.mLnfB)
  reg(params.vLnfB)
  for layer in 0 ..< cfg.nLayer:
    reg(params.layerWq[layer])
    reg(params.layerWk[layer])
    reg(params.layerWv[layer])
    reg(params.layerWo[layer])
    reg(params.layerFc1[layer])
    reg(params.layerFc2[layer])
    reg(params.layerLn1W[layer])
    reg(params.layerLn1B[layer])
    reg(params.layerLn2W[layer])
    reg(params.layerLn2B[layer])
    for slot in 0 ..< 10:
      reg(params.layerM[layer][slot])
      reg(params.layerV[layer][slot])
  reg(consts.posIdx)
  reg(consts.linRowD)
  reg(consts.linColD)
  reg(consts.linRow4D)
  reg(consts.linCol4D)
  reg(consts.linRowV)
  reg(consts.linColV)
  reg(consts.bhIdx)
  reg(consts.attnIdx)
  reg(consts.mask)
  reg(consts.negBig)

proc attention*(h: Tensor, wq, wk, wv, wo: Tensor, consts: GptConsts, cfg: GptConfig): Tensor =
  ## One causal multi-head attention block over the pre-normalized input `h`.
  ##
  ## Expected input:
  ## - `h`: [B, T, D], contiguous
  ## - `wq`, `wk`, `wv`, `wo`: [D, D] weight matrices, contiguous
  ##
  ## Output: [B, T, D] with the attention projection. The residual add stays with the caller.
  let B = cfg.batchSize
  let T = cfg.blockSize
  let D = cfg.dModel
  let H = cfg.nHead
  let HD = D div H
  let BH = B * H

  let q = expMatmul(h, wq, consts.linRowD, consts.linColD)
  let k = expMatmul(h, wk, consts.linRowD, consts.linColD)
  let v = expMatmul(h, wv, consts.linRowD, consts.linColD)

  # head split: [B, T, H, HD] -> permute [B, H, T, HD] -> materialize -> [BH, T, HD]
  let qh = materialize(permute(q.reshape(B, T, H, HD), 0, 2, 1, 3)).reshape(BH, T, HD)
  let kh = materialize(permute(k.reshape(B, T, H, HD), 0, 2, 1, 3)).reshape(BH, T, HD)
  let vh = materialize(permute(v.reshape(B, T, H, HD), 0, 2, 1, 3)).reshape(BH, T, HD)

  # scores: qExp[bh,j,i,d] = q[bh,i,d] and kExpSwapped[bh,j,i,d] = k[bh,j,d]
  let qhFlat = qh.reshape(BH, T * HD)
  let khFlat = kh.reshape(BH, T * HD)
  let qExp = gather(qhFlat, consts.bhIdx).reshape(BH, T, T, HD)
  let kExp = permute(gather(khFlat, consts.bhIdx).reshape(BH, T, T, HD), 0, 2, 1, 3)
  let scores = pairwiseSumAxis(qExp * kExp, 3)  # [BH, T, T]: scores[bh,j,i]
  # transpose to [bh,i,j] so the softmax runs over the key axis j
  let scaled = permute(scores, 0, 2, 1) * (1.0 / sqrt(float64(HD)))
  let masked = where(consts.mask, scaled, consts.negBig)
  let attnWeights = softmax(masked)  # [BH, T, T]: attnWeights[bh,i,j]

  # attention @ V: headOut[bh,i,d] = sum_j attnWeights[bh,i,j] * v[bh,j,d]
  let attnFlat = attnWeights.reshape(BH * T, T)  # line (bh, i) of attnFlat lists attnWeights[bh, i, 0 ..< T]
  let aExp = permute(gather(attnFlat, consts.attnIdx), 0, 1, 3, 2)
  # aExp[bh,i,j,d] = attnWeights[bh,i,j]
  let vExp = gather(vh.reshape(BH, T * HD), consts.bhIdx).reshape(BH, T, T, HD)
  # vExp[bh,i,j,d] = v[bh,j,d]
  let headOut = pairwiseSumAxis(aExp * vExp, 2).reshape(BH, T, HD)

  # [BH, T, HD] -> [B, H, T, HD] -> permute [B, T, H, HD] -> materialize -> [B, T, D]
  let headOutBt = materialize(permute(headOut.reshape(B, H, T, HD), 0, 2, 1, 3)).reshape(B, T, D)
  expMatmul(headOutBt, wo, consts.linRowD, consts.linColD)

proc gptLogits*(params: GptParams, consts: GptConsts, cfg: GptConfig, tokens: Tensor): Tensor =
  ## Forward up to the LM head: returns the logits [B, T, V] for `tokens` [B, T].
  ## Generation reads the last position's logits and samples on the host.
  let B = cfg.batchSize
  let T = cfg.blockSize
  let D = cfg.dModel
  let eps = 1e-5

  let tokEmb = gather(params.wte, tokens)
  let posEmb = gather(params.wpe, consts.posIdx)
  var h = tokEmb + posEmb

  for layer in 0 ..< cfg.nLayer:
    let ln1 = layernorm(h, params.layerLn1W[layer], params.layerLn1B[layer], eps)
    let attnOut = attention(ln1, params.layerWq[layer], params.layerWk[layer],
        params.layerWv[layer], params.layerWo[layer], consts, cfg)
    h = h + attnOut
    let ln2 = layernorm(h, params.layerLn2W[layer], params.layerLn2B[layer], eps)
    let fc1 = expMatmul(ln2, params.layerFc1[layer], consts.linRow4D, consts.linCol4D)
    let act = gelu(fc1)
    let fc2 = expMatmul(act, params.layerFc2[layer], consts.linRowD, consts.linColD)
    h = h + fc2

  let lnf = layernorm(h, params.lnfW, params.lnfB, eps)
  expMatmul(lnf, params.lmHead, consts.linRowV, consts.linColV)

proc gptForward*(params: GptParams, consts: GptConsts, cfg: GptConfig, tokens, targets: Tensor): Tensor =
  ## Full forward: embeddings, nLayer attention + MLP blocks, final layernorm,
  ## LM head, mean cross entropy over every position.
  ##
  ## Expected input:
  ## - `tokens`: int32 [B, T]
  ## - `targets`: int32 [B, T], shifted one position ahead of `tokens`
  ##
  ## Output: the mean loss tensor of shape [B, T].
  ## backward() differentiates the sum of its elements (the mean-loss gradient).
  ## The element sum is the loss value the training loop reports.
  let B = cfg.batchSize
  let T = cfg.blockSize
  let logits = gptLogits(params, consts, cfg, tokens)
  let loss = crossEntropy(logits, targets)
  loss * (1.0 / float64(B * T))

proc adamStepAll*(params: GptParams, cfg: GptConfig, step: int, lr, beta1, beta2, eps: float64) =
  ## Enqueues the in-place adam update of every parameter from its fresh gradient:
  ## - all updates join the caller's record and drain with the next eval
  ## - `step` counts from 1 and drives the bias corrections
  template update(p, m, v: Tensor) =
    adamUpdate(p, grad(p), m, v, lr, beta1, beta2, eps, step)
  update(params.wte, params.mWte, params.vWte)
  update(params.wpe, params.mWpe, params.vWpe)
  update(params.lmHead, params.mLmHead, params.vLmHead)
  update(params.lnfW, params.mLnfW, params.vLnfW)
  update(params.lnfB, params.mLnfB, params.vLnfB)
  for layer in 0 ..< cfg.nLayer:
    update(params.layerWq[layer], params.layerM[layer][0], params.layerV[layer][0])
    update(params.layerWk[layer], params.layerM[layer][1], params.layerV[layer][1])
    update(params.layerWv[layer], params.layerM[layer][2], params.layerV[layer][2])
    update(params.layerWo[layer], params.layerM[layer][3], params.layerV[layer][3])
    update(params.layerFc1[layer], params.layerM[layer][4], params.layerV[layer][4])
    update(params.layerFc2[layer], params.layerM[layer][5], params.layerV[layer][5])
    update(params.layerLn1W[layer], params.layerM[layer][6], params.layerV[layer][6])
    update(params.layerLn1B[layer], params.layerM[layer][7], params.layerV[layer][7])
    update(params.layerLn2W[layer], params.layerM[layer][8], params.layerV[layer][8])
    update(params.layerLn2B[layer], params.layerM[layer][9], params.layerV[layer][9])
