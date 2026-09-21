# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Gated delta-rule references for the positron naive test tier, GDN family
## (arXiv:2412.06464, one scalar log-decay per value head):
##
##   S ← S·exp(g) + k ⊗ (β·(v − (S·exp(g))·k))
##   y ← S'·(q/√Dk)
##
## Two independently spelled forms, sharing no code, never calling each other:
## - per token, gdnDecodeStep and gdnPrefillPerToken walk the recurrence
## - chunked, gdnPrefillChunked runs the WY/UT transform per block, with a per-block
##   solve and ratio-form assembly
##
##   per token: read → decay → kv read → update → y, one step per token
##   chunked: cumg → A → u solve → y and carry, one block per iteration
##
## The intra-family comparison of the two forms is a diagnostic self-check,
## its tolerance from the rounding model.

import std/math
import naive_tensors

func mAt[F](m: NaiveMat[F]; r, c: int): F =
  ## Read-only element (r, c) of a row-major matrix.
  m.data[r * m.cols + c]

func cAt[F](t: NaiveCube[F]; p, r, c: int): F =
  ## Read-only element (p, r, c) of a plane-major cube.
  t.data[(p * t.rows + r) * t.cols + c]

# ─── Per-token spelling ──────────────────────────────────────────────

proc gdnDecodeStep*[F: float32|float64](
    state: var NaiveCube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var NaiveMat[F],        # (B·Hv, Dv) f32-width outputs, one row per head
    q, k: NaiveMat[F],         # (B·Hk, Dk) f32-width queries and keys
    v: NaiveMat[F],            # (B·Hv, Dv) f32-width values
    beta, g: openArray[F],     # (B·Hv,) beta and log-decay, one per value head
    Hv, Hk, hkRatio: int) =
  ## One gated delta-rule step (T = 1) over every value head:
  ##
  ## Contract:
  ## - decay the state by exp(g) per head BEFORE the kv read
  ## - kv[r] = Σ_dk decayed[r, dk]·k[dk], delta[r] = β·(v[r] − kv[r])
  ## - state[r, dk] += k[dk]·delta[r], y[r] = Σ_dk state[r, dk]·(q[dk]/√Dk)
  ##
  ## The state stays at the F width through the step, mirroring the kernel contract.
  ## Head mapping is the kernel's GQA form, value head bh reading key head
  ## (bh mod Hv) div hkRatio + (bh div Hv)·Hk.
  let bhMax = state.planes
  let dv = state.rows
  let dk = state.cols
  doAssert q.rows == Hk and q.cols == dk, "q shape mismatch"
  doAssert k.rows == Hk and k.cols == dk, "k shape mismatch"
  doAssert v.rows == bhMax and v.cols == dv, "v shape mismatch"
  doAssert y.rows == bhMax and y.cols == dv, "y shape mismatch"
  doAssert beta.len == bhMax and g.len == bhMax, "beta/g length mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  let qScale = sqrt(F(dk))
  var decayed = newSeq[F](dv * dk)
  var kvRow = newSeq[F](dv)
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    let gamma = exp(g[bh])
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        decayed[row * dk + dkc] = gamma * state.at(bh, row, dkc)
      var acc: F = 0
      for dkc in 0 ..< dk:
        acc += decayed[row * dk + dkc] * mAt(k, hk, dkc)
      kvRow[row] = acc
    for row in 0 ..< dv:
      let delta = beta[bh] * (mAt(v, bh, row) - kvRow[row])
      for dkc in 0 ..< dk:
        state.at(bh, row, dkc) = decayed[row * dk + dkc] +
          mAt(k, hk, dkc) * delta
      var acc: F = 0
      for dkc in 0 ..< dk:
        acc += state.at(bh, row, dkc) * (mAt(q, hk, dkc) / qScale)
      y.at(bh, row) = acc

proc gdnPrefillPerToken*[F: float32|float64](
    state: var NaiveCube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var NaiveCube[F],       # (B·Hv, T, Dv) f32-width outputs
    q, k: NaiveCube[F],        # (B·Hk, T, Dk) f32-width queries and keys
    v: NaiveCube[F],           # (B·Hv, T, Dv) f32-width values
    beta, g: NaiveMat[F],      # (B·Hv, T) beta and log-decay per value head
    Hv, Hk, hkRatio: int) =
  ## Walks the gated recurrence over T tokens, the state carrying across tokens,
  ## one step per token:
  ##
  ## Contract:
  ## - the step arithmetic restates gdnDecodeStep with the token axis outermost
  ## - the chunked spelling shares nothing with this walk
  let bhMax = state.planes
  let dv = state.rows
  let dk = state.cols
  let T = beta.cols
  doAssert q.planes == Hk and q.rows == T and q.cols == dk, "q shape mismatch"
  doAssert k.planes == Hk and k.rows == T and k.cols == dk, "k shape mismatch"
  doAssert v.planes == bhMax and v.rows == T and v.cols == dv, "v shape mismatch"
  doAssert y.planes == bhMax and y.rows == T and y.cols == dv, "y shape mismatch"
  doAssert beta.rows == bhMax and g.rows == bhMax, "beta/g head count mismatch"
  doAssert g.cols == T, "g token count mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  let qScale = sqrt(F(dk))
  var decayed = newSeq[F](dv * dk)
  var kvRow = newSeq[F](dv)
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for t in 0 ..< T:
      let gamma = exp(mAt(g, bh, t))
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          decayed[row * dk + dkc] = gamma * state.at(bh, row, dkc)
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += decayed[row * dk + dkc] * cAt(k, hk, t, dkc)
        kvRow[row] = acc
      for row in 0 ..< dv:
        let delta = mAt(beta, bh, t) * (cAt(v, bh, t, row) - kvRow[row])
        for dkc in 0 ..< dk:
          state.at(bh, row, dkc) = decayed[row * dk + dkc] +
            cAt(k, hk, t, dkc) * delta
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += state.at(bh, row, dkc) * (cAt(q, hk, t, dkc) / qScale)
        y.at(bh, t, row) = acc

# ─── Chunked spelling, the WY/UT transform ───────────────────────────

type
  GdnChunkedOut* = object
    ## Chunked prefill result.
    y*: NaiveCube[float64]      ## (B·Hv, T, Dv) f64 outputs
    state*: NaiveCube[float64]  ## (B·Hv, Dv, Dk) f64 state after the last token

proc gdnPrefillChunked*(
    state0: NaiveCube[float64],  # (B·Hv, Dv, Dk) f64 initial state, read-only
    q, k: NaiveCube[float64],    # (B·Hk, T, Dk) f64 queries and keys
    v: NaiveCube[float64],       # (B·Hv, T, Dv) f64 values
    beta, g: NaiveMat[float64],  # (B·Hv, T) f64 beta and log-decay per head
    Hv, Hk, hkRatio, chunkLen: int): GdnChunkedOut =
  ## GDN prefill through the WY/UT transform, block by block, fp64, per
  ## block of chunkLen tokens over the in-block cumulative log decay:
  ##
  ## Contract:
  ##
  ##   │ pairdecay  exp(cumg[t] − cumg[s]) computed from the log sum, an
  ##   │            exact-zero decay channel staying defined there
  ##   │ solve      A[t, s] = pairdecay(t, s)·(k_t·k_s), s < t only
  ##   │            u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
  ##   │            G_t = exp(cumg[t])·(S_carry·k_t), token-order solve
  ##   │ outputs    y_t = exp(cumg[t])·(S_carry·q̃_t) +
  ##   │            Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s
  ##   │ carry      S = exp(cumg[end])·S_carry +
  ##   │            Σ_s pairdecay(end, s)·k_s⊗u_s
  ##
  ## The result equals the per-token walk up to fp64 reassociation.
  let bhMax = state0.planes
  let dv = state0.rows
  let dk = state0.cols
  let T = beta.cols
  doAssert q.planes == Hk and q.rows == T and q.cols == dk, "q shape mismatch"
  doAssert k.planes == Hk and k.rows == T and k.cols == dk, "k shape mismatch"
  doAssert v.planes == bhMax and v.rows == T and v.cols == dv, "v shape mismatch"
  doAssert beta.rows == bhMax and g.rows == bhMax, "beta/g head count mismatch"
  doAssert g.cols == T, "g token count mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  doAssert chunkLen >= 1, "chunkLen must be at least 1"
  result.y = NaiveCube[float64](planes: bhMax, rows: T, cols: dv)
  result.y.data = newSeq[float64](bhMax * T * dv)
  result.state = NaiveCube[float64](planes: bhMax, rows: dv, cols: dk)
  result.state.data = newSeq[float64](bhMax * dv * dk)
  let qScale = sqrt(float64(dk))

  var carry = newSeq[float64](dv * dk)
  var cumg = newSeq[float64](chunkLen)
  var amat = newSeq[float64](chunkLen * chunkLen)
  var bmat = newSeq[float64](chunkLen * chunkLen)
  var uvec = newSeq[float64](dv * chunkLen)
  var readCarry = newSeq[float64](dv)   # S_carry read against one k or q row
  var gread = newSeq[float64](dv)       # G_t, the decayed carry read
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        carry[row * dk + dkc] = cAt(state0, bh, row, dkc)
    var c0 = 0
    while c0 < T:
      let cLen = min(chunkLen, T - c0)
      # In-block cumulative log decay.
      cumg[0] = mAt(g, bh, c0)
      for i in 1 ..< cLen:
        cumg[i] = cumg[i - 1] + mAt(g, bh, c0 + i)
      # Update vectors, solved in token order.
      for t in 0 ..< cLen:
        let gt = c0 + t
        # G_t = exp(cumg[t])·(S_carry·k_t)
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += carry[row * dk + dkc] * cAt(k, hk, gt, dkc)
          readCarry[row] = acc
        let decayT = exp(cumg[t])
        for row in 0 ..< dv:
          gread[row] = decayT * readCarry[row]
        # A[t, s] for s < t.
        for s in 0 ..< t:
          var kdot: float64 = 0
          for dkc in 0 ..< dk:
            kdot += cAt(k, hk, gt, dkc) * cAt(k, hk, c0 + s, dkc)
          amat[t * chunkLen + s] = exp(cumg[t] - cumg[s]) * kdot
        # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
        for row in 0 ..< dv:
          let ubase = mAt(beta, bh, gt) * (cAt(v, bh, gt, row) - gread[row])
          var acc: float64 = 0
          for s in 0 ..< t:
            acc += amat[t * chunkLen + s] * uvec[row * chunkLen + s]
          uvec[row * chunkLen + t] = ubase - mAt(beta, bh, gt) * acc
      # Block outputs, y_t via the decayed carry read plus the ratio form,
      # the s = t ratio term included
      for t in 0 ..< cLen:
        let gt = c0 + t
        # B[t, s] = pairdecay(t, s)·(q̃_t·k_s) for s ≤ t.
        for s in 0 .. t:
          var qkdot: float64 = 0
          for dkc in 0 ..< dk:
            qkdot += (cAt(q, hk, gt, dkc) / qScale) * cAt(k, hk, c0 + s, dkc)
          bmat[t * chunkLen + s] = exp(cumg[t] - cumg[s]) * qkdot
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += carry[row * dk + dkc] * (cAt(q, hk, gt, dkc) / qScale)
          var oacc = exp(cumg[t]) * acc
          for s in 0 .. t:
            oacc += bmat[t * chunkLen + s] * uvec[row * chunkLen + s]
          result.y.at(bh, gt, row) = oacc
      # Carry out of the block.
      let decayEnd = exp(cumg[cLen - 1])
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          var acc = decayEnd * carry[row * dk + dkc]
          for s in 0 ..< cLen:
            acc += exp(cumg[cLen - 1] - cumg[s]) * cAt(k, hk, c0 + s, dkc) *
              uvec[row * chunkLen + s]
          carry[row * dk + dkc] = acc
      c0 += cLen
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        result.state.at(bh, row, dkc) = carry[row * dk + dkc]
