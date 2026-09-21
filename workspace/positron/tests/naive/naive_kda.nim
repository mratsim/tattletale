# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kimi Delta Attention references for the positron naive test tier (arXiv:2510.26692):
##
##   S ← S·Diag(exp(g)) + k ⊗ (β·(v − (S·Diag(exp(g)))·k))
##   y ← S'·(q/√Dk)
##
## Decay per key channel, one log-decay per KEY channel, applied BEFORE the kv read, so
##   the kv read contracts the decayed state kᵀ·Diag(exp(g))·S
## - g is a (B·Hk[, T], Dk) matrix, one row per key head
## - a scalar decay applied after the contraction is a different op, kᵀ·S·exp(g), the GDN shape
##
## Two independently spelled forms, sharing no code, never calling each other:
## - per token, kdaDecodeStep and kdaPrefillPerToken walk the recurrence
## - chunked, kdaPrefillChunked solves each block through the WY/UT transform with the per-channel pair decay
##
##   per token: read → decay → kv read → update → y, one step per token
##   chunked: cumg → A → u solve → y and carry, one block per iteration
##
## The intra-family comparison is a diagnostic self-check, its tolerance
## from the rounding model, the q scale an elementwise division by √Dk.

import std/math
import naive_tensors

func mAt[F](m: NaiveMat[F]; r, c: int): F =
  ## Read-only element (r, c) of a row-major matrix.
  m.data[r * m.cols + c]

func cAt[F](t: NaiveCube[F]; p, r, c: int): F =
  ## Read-only element (p, r, c) of a plane-major cube.
  t.data[(p * t.rows + r) * t.cols + c]

# ─── Per-token spelling ──────────────────────────────────────────────

proc kdaDecodeStep*[F: float32|float64](
    state: var NaiveCube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var NaiveMat[F],        # (B·Hv, Dv) f32-width outputs, one row per head
    q, k, g: NaiveMat[F],      # (B·Hk, Dk) f32-width queries, keys, log-decay
    v: NaiveMat[F],            # (B·Hv, Dv) f32-width values
    beta: openArray[F],        # (B·Hv,) beta, one per value head
    Hv, Hk, hkRatio: int) =
  ## One Kimi Delta Attention step (T = 1) over every value head:
  ##
  ## Contract:
  ## - decay the state by exp(g) per key channel BEFORE the kv read
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
  doAssert g.rows == Hk and g.cols == dk, "g shape mismatch"
  doAssert v.rows == bhMax and v.cols == dv, "v shape mismatch"
  doAssert y.rows == bhMax and y.cols == dv, "y shape mismatch"
  doAssert beta.len == bhMax, "beta length mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  let qScale = sqrt(F(dk))
  var decayed = newSeq[F](dv * dk)
  var kvRow = newSeq[F](dv)
  var gRow = newSeq[F](dk)   # exp of the per-channel log-decay, one key head
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for dkc in 0 ..< dk:
      gRow[dkc] = exp(mAt(g, hk, dkc))
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        decayed[row * dk + dkc] = gRow[dkc] * state.at(bh, row, dkc)
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

proc kdaPrefillPerToken*[F: float32|float64](
    state: var NaiveCube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var NaiveCube[F],       # (B·Hv, T, Dv) f32-width outputs
    q, k, g: NaiveCube[F],     # (B·Hk, T, Dk) f32-width queries, keys, log-decay
    v: NaiveCube[F],           # (B·Hv, T, Dv) f32-width values
    beta: NaiveMat[F],         # (B·Hv, T) beta per value head
    Hv, Hk, hkRatio: int) =
  ## Walks the Kimi Delta Attention recurrence over T tokens, the state
  ## carrying across tokens, one step per token:
  ##
  ## Contract:
  ## - the step arithmetic restates kdaDecodeStep with the token axis outermost
  ## - the chunked spelling shares nothing with this walk
  let bhMax = state.planes
  let dv = state.rows
  let dk = state.cols
  let T = beta.cols
  doAssert q.planes == Hk and q.rows == T and q.cols == dk, "q shape mismatch"
  doAssert k.planes == Hk and k.rows == T and k.cols == dk, "k shape mismatch"
  doAssert g.planes == Hk and g.rows == T and g.cols == dk, "g shape mismatch"
  doAssert v.planes == bhMax and v.rows == T and v.cols == dv, "v shape mismatch"
  doAssert y.planes == bhMax and y.rows == T and y.cols == dv, "y shape mismatch"
  doAssert beta.rows == bhMax, "beta head count mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  let qScale = sqrt(F(dk))
  var decayed = newSeq[F](dv * dk)
  var kvRow = newSeq[F](dv)
  var gRow = newSeq[F](dk)   # exp of the per-channel log-decay, one token
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for t in 0 ..< T:
      for dkc in 0 ..< dk:
        gRow[dkc] = exp(g.data[(hk * T + t) * dk + dkc])
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          decayed[row * dk + dkc] = gRow[dkc] * state.at(bh, row, dkc)
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += decayed[row * dk + dkc] * k.data[(hk * T + t) * dk + dkc]
        kvRow[row] = acc
      for row in 0 ..< dv:
        let delta = beta.data[bh * T + t] * (v.data[(bh * T + t) * dv + row] - kvRow[row])
        for dkc in 0 ..< dk:
          state.at(bh, row, dkc) = decayed[row * dk + dkc] +
            k.data[(hk * T + t) * dk + dkc] * delta
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += state.at(bh, row, dkc) *
            (q.data[(hk * T + t) * dk + dkc] / qScale)
        y.at(bh, t, row) = acc

# ─── Chunked spelling, the WY/UT transform, per-channel pair decay ───

type
  KdaChunkedOut* = object
    ## Chunked prefill result.
    y*: NaiveCube[float64]      ## (B·Hv, T, Dv) f64 outputs
    state*: NaiveCube[float64]  ## (B·Hv, Dv, Dk) f64 state after the last token

proc kdaPrefillChunked*(
    state0: NaiveCube[float64],  # (B·Hv, Dv, Dk) f64 initial state, read-only
    q, k, g: NaiveCube[float64], # (B·Hk, T, Dk) f64 queries, keys, log-decay
    v: NaiveCube[float64],       # (B·Hv, T, Dv) f64 values
    beta: NaiveMat[float64],     # (B·Hv, T) f64 beta per value head
    Hv, Hk, hkRatio, chunkLen: int): KdaChunkedOut =
  ## KDA prefill through the WY/UT transform, block by block, fp64, per
  ## block of chunkLen tokens over the cumulative log decay:
  ##
  ## Contract:
  ##
  ##   │ pairdecay  exp(cumg[t, dk] − cumg[s, dk]) computed from the log
  ##   │            sum, exact-zero decay channels staying defined
  ##   │ solve      A[t, s] = Σ_dk pairdecay(t, s)[dk]·k_t[dk]·k_s[dk]
  ##   │            u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
  ##   │            G_t[r] = Σ_dk exp(cumg[t, dk])·k_t[dk]·S_carry[r, dk]
  ##   │ outputs    y_t[r] = H_t[r] + Σ_{s≤t} B[t, s]·u_s[r]
  ##   │            B[t, s] = Σ_dk pairdecay(t, s)[dk]·q̃_t[dk]·k_s[dk]
  ##   │            H_t[r] = Σ_dk exp(cumg[t, dk])·q̃_t[dk]·S_carry[r, dk]
  ##   │ carry      S[r, dk] = exp(cumg[end, dk])·S_carry[r, dk] +
  ##   │            Σ_s pairdecay(end, s)[dk]·k_s[dk]·u_s[r]
  ##
  ## The result equals the per-token walk up to fp64 reassociation.
  let bhMax = state0.planes
  let dv = state0.rows
  let dk = state0.cols
  let T = beta.cols
  doAssert q.planes == Hk and q.rows == T and q.cols == dk, "q shape mismatch"
  doAssert k.planes == Hk and k.rows == T and k.cols == dk, "k shape mismatch"
  doAssert g.planes == Hk and g.rows == T and g.cols == dk, "g shape mismatch"
  doAssert v.planes == bhMax and v.rows == T and v.cols == dv, "v shape mismatch"
  doAssert beta.rows == bhMax, "beta head count mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  doAssert chunkLen >= 1, "chunkLen must be at least 1"
  result.y = NaiveCube[float64](planes: bhMax, rows: T, cols: dv)
  result.y.data = newSeq[float64](bhMax * T * dv)
  result.state = NaiveCube[float64](planes: bhMax, rows: dv, cols: dk)
  result.state.data = newSeq[float64](bhMax * dv * dk)
  let qScale = sqrt(float64(dk))

  var carry = newSeq[float64](dv * dk)
  var cumg = newSeq[float64](chunkLen * dk)
  var amat = newSeq[float64](chunkLen * chunkLen)
  var bmat = newSeq[float64](chunkLen * chunkLen)
  var uvec = newSeq[float64](dv * chunkLen)
  var gread = newSeq[float64](dv)       # G_t, the decayed carry read
  var hread = newSeq[float64](dv)       # H_t, the decayed carry read against q
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        carry[row * dk + dkc] = cAt(state0, bh, row, dkc)
    var c0 = 0
    while c0 < T:
      let cLen = min(chunkLen, T - c0)
      # In-block cumulative log decay, one row per token.
      for dkc in 0 ..< dk:
        cumg[dkc] = g.data[(hk * T + c0) * dk + dkc]
      for i in 1 ..< cLen:
        for dkc in 0 ..< dk:
          cumg[i * dk + dkc] = cumg[(i - 1) * dk + dkc] +
            g.data[(hk * T + c0 + i) * dk + dkc]
      # Update vectors, solved in token order.
      for t in 0 ..< cLen:
        let gt = c0 + t
        # G_t[r] = Σ_dk exp(cumg[t, dk])·k_t[dk]·S_carry[r, dk]
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumg[t * dk + dkc]) * k.data[(hk * T + gt) * dk + dkc] *
              carry[row * dk + dkc]
          gread[row] = acc
        # A[t, s] for s < t.
        for s in 0 ..< t:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumg[t * dk + dkc] - cumg[s * dk + dkc]) *
              k.data[(hk * T + gt) * dk + dkc] * k.data[(hk * T + c0 + s) * dk + dkc]
          amat[t * chunkLen + s] = acc
        # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
        for row in 0 ..< dv:
          let ubase = beta.data[bh * T + gt] *
            (v.data[(bh * T + gt) * dv + row] - gread[row])
          var acc: float64 = 0
          for s in 0 ..< t:
            acc += amat[t * chunkLen + s] * uvec[row * chunkLen + s]
          uvec[row * chunkLen + t] = ubase - beta.data[bh * T + gt] * acc
      # Block outputs, y_t[r] = H_t[r] plus the B[t, s] ratio form
      for t in 0 ..< cLen:
        let gt = c0 + t
        # B[t, s] for s ≤ t, the s = t pair decay exp(0) = 1 per channel.
        for s in 0 .. t:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumg[t * dk + dkc] - cumg[s * dk + dkc]) *
              (q.data[(hk * T + gt) * dk + dkc] / qScale) *
              k.data[(hk * T + c0 + s) * dk + dkc]
          bmat[t * chunkLen + s] = acc
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumg[t * dk + dkc]) *
              (q.data[(hk * T + gt) * dk + dkc] / qScale) * carry[row * dk + dkc]
          hread[row] = acc
          var oacc = hread[row]
          for s in 0 .. t:
            oacc += bmat[t * chunkLen + s] * uvec[row * chunkLen + s]
          result.y.at(bh, gt, row) = oacc
      # Carry out of the block, per key channel.
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          var acc = exp(cumg[(cLen - 1) * dk + dkc]) * carry[row * dk + dkc]
          for s in 0 ..< cLen:
            acc += exp(cumg[(cLen - 1) * dk + dkc] - cumg[s * dk + dkc]) *
              k.data[(hk * T + c0 + s) * dk + dkc] * uvec[row * chunkLen + s]
          carry[row * dk + dkc] = acc
      c0 += cLen
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        result.state.at(bh, row, dkc) = carry[row * dk + dkc]
