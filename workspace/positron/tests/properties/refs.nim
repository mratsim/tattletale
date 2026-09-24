# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ─── Ceramic test reference tier ──────────────────────────────────────

## Reference walks of the ceramic suites, the retired naive tier's exact
## arithmetic carried on plain seq storage:
##
## | section   | contents                                                            |
## | --------- | ------------------------------------------------------------------- |
## | storage   | `Mat`/`Cube` over owned seqs, the at accessors and the copy helpers |
## | GDN/KDA   | the per-token decode steps, the per-token prefills and the chunked  |
## |           | WY/UT prefills, the per-token and chunked spellings sharing no code |
## | layer ops | the fused decoder layer's stage ops (norms, conv, gates, router)    |
## | grouped   | the grouped GEMM reference and the one-token layer composition      |
##
## Arithmetic contract, carried verbatim from the retired tier:
##
## - every operand order, accumulator form and rounding site is unchanged
## - a suite's recorded numbers carry over when the reference import repoints here
## - the bit surgery comes from the properties support surface

import std/math
import ../properties/properties

# ─── Storage types and accessors ─────────────────────────────────────

type
  Mat*[T] = object
    ## Row-major M×N matrix over owned storage.
    ## - element (r, c) lives at data[r * cols + c]
    rows*, cols*: int
    data*: seq[T]

  Cube*[T] = object
    ## Row-major P×R×C tensor over owned storage.
    ## - element (p, r, c) lives at data[(p * rows + r) * cols + c]
    planes*, rows*, cols*: int
    data*: seq[T]

func at*[T](m: var Mat[T]; r, c: int): var T =
  ## Returns element (r, c) as a mutable reference. Both indices must
  ## sit inside the shape, out-of-shape access is a caller bug.
  m.data[r * m.cols + c]

func at*[T](m: var Cube[T]; p, r, c: int): var T =
  ## Returns element (p, r, c) as a mutable reference, with all three
  ## indices inside the shape.
  m.data[(p * m.rows + r) * m.cols + c]

func copyTensor*[T](m: Mat[T]): Mat[T] =
  ## Returns an independent copy of the matrix.
  result.rows = m.rows
  result.cols = m.cols
  result.data = newSeq[T](m.data.len)
  for i in 0 ..< m.data.len:
    result.data[i] = m.data[i]

func copyTensor*[T](t: Cube[T]): Cube[T] =
  ## Returns an independent copy of the cube.
  result.planes = t.planes
  result.rows = t.rows
  result.cols = t.cols
  result.data = newSeq[T](t.data.len)
  for i in 0 ..< t.data.len:
    result.data[i] = t.data[i]

func mAt*[F](m: Mat[F]; r, c: int): F =
  ## Read-only element (r, c) of a row-major matrix.
  m.data[r * m.cols + c]

func cAt*[F](t: Cube[F]; p, r, c: int): F =
  ## Read-only element (p, r, c) of a plane-major cube.
  t.data[(p * t.rows + r) * t.cols + c]

func widenF64*[T: float32|float64](m: Mat[T]): Mat[float64] =
  ## Returns the fp64 widening copy of a float-width matrix.
  result.rows = m.rows
  result.cols = m.cols
  result.data = newSeq[float64](m.data.len)
  for i in 0 ..< m.data.len:
    result.data[i] = float64(m.data[i])

func widenF64*[T: float32|float64](t: Cube[T]): Cube[float64] =
  ## Returns the fp64 widening copy of a float-width cube.
  result.planes = t.planes
  result.rows = t.rows
  result.cols = t.cols
  result.data = newSeq[float64](t.data.len)
  for i in 0 ..< t.data.len:
    result.data[i] = float64(t.data[i])

func castCube*[F: float32|float64](t: Cube[float32]): Cube[F] =
  ## Returns the cube at the run's float width, an independent copy when F is fp32 and the exact widening
  ## when F is fp64, so both dtype runs
  ## of one comparison see identical values.
  when F is float32:
    result = t.copyTensor()
  else:
    result = t.widenF64()

func castMat*[F: float32|float64](t: Mat[float32]): Mat[F] =
  ## Returns the matrix at the run's float width, same contract as castCube.
  when F is float32:
    result = t.copyTensor()
  else:
    result = t.widenF64()

func zerosCube*[F: float32|float64](p, r, c: int): Cube[F] =
  ## Returns a zero cube of the given shape at the run's float width.
  Cube[F](planes: p, rows: r, cols: c, data: newSeq[F](p * r * c))

func fillMat*[T](m: var Mat[T]; value: T) =
  ## Overwrites every matrix element with `value`.
  for i in 0 ..< m.data.len:
    m.data[i] = value

func fillCube*[T](t: var Cube[T]; value: T) =
  ## Overwrites every cube element with `value`.
  for i in 0 ..< t.data.len:
    t.data[i] = value

# ─── GDN references (arXiv:2412.06464, one scalar log-decay per value head) ──

## Gated delta-rule recurrence, GDN family:
##
##   S ← S·exp(g) + k ⊗ (β·(v − (S·exp(g))·k))
##   y ← S'·(q/√Dk)
##
## Two independently spelled forms, sharing no code, never calling each other:
## - per token, gdnDecodeStep and gdnPrefillPerToken walk the recurrence
## - chunked, gdnPrefillChunked runs the WY/UT transform per block,
##   one solve plus one ratio-form assembly per block

func mAtGdn[F](m: Mat[F]; r, c: int): F =
  ## Read-only element (r, c) of a row-major matrix.
  m.data[r * m.cols + c]

func cAtGdn[F](t: Cube[F]; p, r, c: int): F =
  ## Read-only element (p, r, c) of a plane-major cube.
  t.data[(p * t.rows + r) * t.cols + c]

proc gdnDecodeStep*[F: float32|float64](
    state: var Cube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var Mat[F],        # (B·Hv, Dv) f32-width outputs, one row per head
    q, k: Mat[F],         # (B·Hk, Dk) f32-width queries and keys
    v: Mat[F],            # (B·Hv, Dv) f32-width values
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
        acc += decayed[row * dk + dkc] * mAtGdn(k, hk, dkc)
      kvRow[row] = acc
    for row in 0 ..< dv:
      let delta = beta[bh] * (mAtGdn(v, bh, row) - kvRow[row])
      for dkc in 0 ..< dk:
        state.at(bh, row, dkc) = decayed[row * dk + dkc] +
          mAtGdn(k, hk, dkc) * delta
      var acc: F = 0
      for dkc in 0 ..< dk:
        acc += state.at(bh, row, dkc) * (mAtGdn(q, hk, dkc) / qScale)
      y.at(bh, row) = acc

proc gdnPrefillPerToken*[F: float32|float64](
    state: var Cube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var Cube[F],       # (B·Hv, T, Dv) f32-width outputs
    q, k: Cube[F],        # (B·Hk, T, Dk) f32-width queries and keys
    v: Cube[F],           # (B·Hv, T, Dv) f32-width values
    beta, g: Mat[F],      # (B·Hv, T) beta and log-decay per value head
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
      let gamma = exp(mAtGdn(g, bh, t))
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          decayed[row * dk + dkc] = gamma * state.at(bh, row, dkc)
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += decayed[row * dk + dkc] * cAtGdn(k, hk, t, dkc)
        kvRow[row] = acc
      for row in 0 ..< dv:
        let delta = mAtGdn(beta, bh, t) * (cAtGdn(v, bh, t, row) - kvRow[row])
        for dkc in 0 ..< dk:
          state.at(bh, row, dkc) = decayed[row * dk + dkc] +
            cAtGdn(k, hk, t, dkc) * delta
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += state.at(bh, row, dkc) * (cAtGdn(q, hk, t, dkc) / qScale)
        y.at(bh, t, row) = acc

type
  GdnChunkedOut* = object
    ## Chunked prefill result.
    y*: Cube[float64]      ## (B·Hv, T, Dv) f64 outputs
    state*: Cube[float64]  ## (B·Hv, Dv, Dk) f64 state after the last token

proc gdnPrefillChunked*(
    state0: Cube[float64],  # (B·Hv, Dv, Dk) f64 initial state, read-only
    q, k: Cube[float64],    # (B·Hk, T, Dk) f64 queries and keys
    v: Cube[float64],       # (B·Hv, T, Dv) f64 values
    beta, g: Mat[float64],  # (B·Hv, T) f64 beta and log-decay per head
    Hv, Hk, hkRatio, chunkLen: int): GdnChunkedOut =
  ## GDN prefill through the WY/UT transform, block by block, fp64, per
  ## block of chunkLen tokens over the in-block cumulative log decay:
  ##
  ## Contract:
  ##
  ##   │ pairdecay  exp(cumulogdecay[t] − cumulogdecay[s]) computed from the log sum, an
  ##   │            exact-zero decay channel staying defined there
  ##   │ solve      A[t, s] = pairdecay(t, s)·(k_t·k_s), s < t only
  ##   │            u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
  ##   │            G_t = exp(cumulogdecay[t])·(S_carry·k_t), token-order solve
  ##   │ outputs    y_t = exp(cumulogdecay[t])·(S_carry·q̃_t) +
  ##   │            Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s
  ##   │ carry      S = exp(cumulogdecay[end])·S_carry +
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
  result.y = Cube[float64](planes: bhMax, rows: T, cols: dv)
  result.y.data = newSeq[float64](bhMax * T * dv)
  result.state = Cube[float64](planes: bhMax, rows: dv, cols: dk)
  result.state.data = newSeq[float64](bhMax * dv * dk)
  let qScale = sqrt(float64(dk))

  var carry = newSeq[float64](dv * dk)
  var cumulogdecay = newSeq[float64](chunkLen)
  var amat = newSeq[float64](chunkLen * chunkLen)
  var bmat = newSeq[float64](chunkLen * chunkLen)
  var uvec = newSeq[float64](dv * chunkLen)
  var readCarry = newSeq[float64](dv)   # S_carry read against one k or q row
  var gread = newSeq[float64](dv)       # G_t, the decayed carry read
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        carry[row * dk + dkc] = cAtGdn(state0, bh, row, dkc)
    var c0 = 0
    while c0 < T:
      let cLen = min(chunkLen, T - c0)
      # In-block cumulative log decay.
      cumulogdecay[0] = mAtGdn(g, bh, c0)
      for i in 1 ..< cLen:
        cumulogdecay[i] = cumulogdecay[i - 1] + mAtGdn(g, bh, c0 + i)
      # Update vectors, solved in token order.
      for t in 0 ..< cLen:
        let gt = c0 + t
        # G_t = exp(cumulogdecay[t])·(S_carry·k_t)
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += carry[row * dk + dkc] * cAtGdn(k, hk, gt, dkc)
          readCarry[row] = acc
        let decayT = exp(cumulogdecay[t])
        for row in 0 ..< dv:
          gread[row] = decayT * readCarry[row]
        # A[t, s] for s < t.
        for s in 0 ..< t:
          var kdot: float64 = 0
          for dkc in 0 ..< dk:
            kdot += cAtGdn(k, hk, gt, dkc) * cAtGdn(k, hk, c0 + s, dkc)
          amat[t * chunkLen + s] = exp(cumulogdecay[t] - cumulogdecay[s]) * kdot
        # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
        for row in 0 ..< dv:
          let ubase = mAtGdn(beta, bh, gt) * (cAtGdn(v, bh, gt, row) - gread[row])
          var acc: float64 = 0
          for s in 0 ..< t:
            acc += amat[t * chunkLen + s] * uvec[row * chunkLen + s]
          uvec[row * chunkLen + t] = ubase - mAtGdn(beta, bh, gt) * acc
      # Block outputs, y_t via the decayed carry read plus the ratio form,
      # the s = t ratio term included
      for t in 0 ..< cLen:
        let gt = c0 + t
        # B[t, s] = pairdecay(t, s)·(q̃_t·k_s) for s ≤ t.
        for s in 0 .. t:
          var qkdot: float64 = 0
          for dkc in 0 ..< dk:
            qkdot += (cAtGdn(q, hk, gt, dkc) / qScale) * cAtGdn(k, hk, c0 + s, dkc)
          bmat[t * chunkLen + s] = exp(cumulogdecay[t] - cumulogdecay[s]) * qkdot
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += carry[row * dk + dkc] * (cAtGdn(q, hk, gt, dkc) / qScale)
          var oacc = exp(cumulogdecay[t]) * acc
          for s in 0 .. t:
            oacc += bmat[t * chunkLen + s] * uvec[row * chunkLen + s]
          result.y.at(bh, gt, row) = oacc
      # Carry out of the block.
      let decayEnd = exp(cumulogdecay[cLen - 1])
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          var acc = decayEnd * carry[row * dk + dkc]
          for s in 0 ..< cLen:
            acc += exp(cumulogdecay[cLen - 1] - cumulogdecay[s]) * cAtGdn(k, hk, c0 + s, dkc) *
              uvec[row * chunkLen + s]
          carry[row * dk + dkc] = acc
      c0 += cLen
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        result.state.at(bh, row, dkc) = carry[row * dk + dkc]

# ─── KDA references (arXiv:2510.26692, decay per key channel) ─────────

## Kimi Delta Attention recurrence:
##
##   S ← S·Diag(exp(g)) + k ⊗ (β·(v − (S·Diag(exp(g)))·k))
##   y ← S'·(q/√Dk)
##
## Decay per key channel, one log-decay per KEY channel, applied BEFORE the kv read, so the kv read
## contracts the decayed state kᵀ·Diag(exp(g))·S:
## - g is a (B·Hk[, T], Dk) matrix, one row per key head
## - a scalar decay applied after the contraction is a different op,
##   kᵀ·S·exp(g), the GDN shape
##
## Two independently spelled forms, sharing no code, never calling each other:
## - per token, kdaDecodeStep and kdaPrefillPerToken walk the recurrence
## - chunked, kdaPrefillChunked solves each block through the WY/UT transform
##   with the per-channel pair decay

func mAtKda[F](m: Mat[F]; r, c: int): F =
  ## Read-only element (r, c) of a row-major matrix.
  m.data[r * m.cols + c]

func cAtKda[F](t: Cube[F]; p, r, c: int): F =
  ## Read-only element (p, r, c) of a plane-major cube.
  t.data[(p * t.rows + r) * t.cols + c]

proc kdaDecodeStep*[F: float32|float64](
    state: var Cube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var Mat[F],        # (B·Hv, Dv) f32-width outputs, one row per head
    q, k, g: Mat[F],      # (B·Hk, Dk) f32-width queries, keys, log-decay
    v: Mat[F],            # (B·Hv, Dv) f32-width values
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
      gRow[dkc] = exp(mAtKda(g, hk, dkc))
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        decayed[row * dk + dkc] = gRow[dkc] * state.at(bh, row, dkc)
      var acc: F = 0
      for dkc in 0 ..< dk:
        acc += decayed[row * dk + dkc] * mAtKda(k, hk, dkc)
      kvRow[row] = acc
    for row in 0 ..< dv:
      let delta = beta[bh] * (mAtKda(v, bh, row) - kvRow[row])
      for dkc in 0 ..< dk:
        state.at(bh, row, dkc) = decayed[row * dk + dkc] +
          mAtKda(k, hk, dkc) * delta
      var acc: F = 0
      for dkc in 0 ..< dk:
        acc += state.at(bh, row, dkc) * (mAtKda(q, hk, dkc) / qScale)
      y.at(bh, row) = acc

proc kdaPrefillPerToken*[F: float32|float64](
    state: var Cube[F],   # (B·Hv, Dv, Dk) f32-width state, in place
    y: var Cube[F],       # (B·Hv, T, Dv) f32-width outputs
    q, k, g: Cube[F],     # (B·Hk, T, Dk) f32-width queries, keys, log-decay
    v: Cube[F],           # (B·Hv, T, Dv) f32-width values
    beta: Mat[F],         # (B·Hv, T) beta per value head
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

type
  KdaChunkedOut* = object
    ## Chunked prefill result.
    y*: Cube[float64]      ## (B·Hv, T, Dv) f64 outputs
    state*: Cube[float64]  ## (B·Hv, Dv, Dk) f64 state after the last token

proc kdaPrefillChunked*(
    state0: Cube[float64],  # (B·Hv, Dv, Dk) f64 initial state, read-only
    q, k, g: Cube[float64], # (B·Hk, T, Dk) f64 queries, keys, log-decay
    v: Cube[float64],       # (B·Hv, T, Dv) f64 values
    beta: Mat[float64],     # (B·Hv, T) f64 beta per value head
    Hv, Hk, hkRatio, chunkLen: int): KdaChunkedOut =
  ## KDA prefill through the WY/UT transform, block by block, fp64, per
  ## block of chunkLen tokens over the cumulative log decay:
  ##
  ## Contract:
  ##
  ##   │ pairdecay  exp(cumulogdecay[t, dk] − cumulogdecay[s, dk]) computed from the log
  ##   │            sum, exact-zero decay channels staying defined
  ##   │ solve      A[t, s] = Σ_dk pairdecay(t, s)[dk]·k_t[dk]·k_s[dk]
  ##   │            u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s
  ##   │            G_t[r] = Σ_dk exp(cumulogdecay[t, dk])·k_t[dk]·S_carry[r, dk]
  ##   │ outputs    y_t[r] = H_t[r] + Σ_{s≤t} B[t, s]·u_s[r]
  ##   │            B[t, s] = Σ_dk pairdecay(t, s)[dk]·q̃_t[dk]·k_s[dk]
  ##   │            H_t[r] = Σ_dk exp(cumulogdecay[t, dk])·q̃_t[dk]·S_carry[r, dk]
  ##   │ carry      S[r, dk] = exp(cumulogdecay[end, dk])·S_carry[r, dk] +
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
  result.y = Cube[float64](planes: bhMax, rows: T, cols: dv)
  result.y.data = newSeq[float64](bhMax * T * dv)
  result.state = Cube[float64](planes: bhMax, rows: dv, cols: dk)
  result.state.data = newSeq[float64](bhMax * dv * dk)
  let qScale = sqrt(float64(dk))

  var carry = newSeq[float64](dv * dk)
  var cumulogdecay = newSeq[float64](chunkLen * dk)
  var amat = newSeq[float64](chunkLen * chunkLen)
  var bmat = newSeq[float64](chunkLen * chunkLen)
  var uvec = newSeq[float64](dv * chunkLen)
  var gread = newSeq[float64](dv)       # G_t, the decayed carry read
  var hread = newSeq[float64](dv)       # H_t, the decayed carry read against q
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        carry[row * dk + dkc] = cAtKda(state0, bh, row, dkc)
    var c0 = 0
    while c0 < T:
      let cLen = min(chunkLen, T - c0)
      # In-block cumulative log decay, one row per token.
      for dkc in 0 ..< dk:
        cumulogdecay[dkc] = g.data[(hk * T + c0) * dk + dkc]
      for i in 1 ..< cLen:
        for dkc in 0 ..< dk:
          cumulogdecay[i * dk + dkc] = cumulogdecay[(i - 1) * dk + dkc] +
            g.data[(hk * T + c0 + i) * dk + dkc]
      # Update vectors, solved in token order.
      for t in 0 ..< cLen:
        let gt = c0 + t
        # G_t[r] = Σ_dk exp(cumulogdecay[t, dk])·k_t[dk]·S_carry[r, dk]
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumulogdecay[t * dk + dkc]) * k.data[(hk * T + gt) * dk + dkc] *
              carry[row * dk + dkc]
          gread[row] = acc
        # A[t, s] for s < t.
        for s in 0 ..< t:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumulogdecay[t * dk + dkc] - cumulogdecay[s * dk + dkc]) *
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
            acc += exp(cumulogdecay[t * dk + dkc] - cumulogdecay[s * dk + dkc]) *
              (q.data[(hk * T + gt) * dk + dkc] / qScale) *
              k.data[(hk * T + c0 + s) * dk + dkc]
          bmat[t * chunkLen + s] = acc
        for row in 0 ..< dv:
          var acc: float64 = 0
          for dkc in 0 ..< dk:
            acc += exp(cumulogdecay[t * dk + dkc]) *
              (q.data[(hk * T + gt) * dk + dkc] / qScale) * carry[row * dk + dkc]
          hread[row] = acc
          var oacc = hread[row]
          for s in 0 .. t:
            oacc += bmat[t * chunkLen + s] * uvec[row * chunkLen + s]
          result.y.at(bh, gt, row) = oacc
      # Carry out of the block, per key channel.
      for row in 0 ..< dv:
        for dkc in 0 ..< dk:
          var acc = exp(cumulogdecay[(cLen - 1) * dk + dkc]) * carry[row * dk + dkc]
          for s in 0 ..< cLen:
            acc += exp(cumulogdecay[(cLen - 1) * dk + dkc] - cumulogdecay[s * dk + dkc]) *
              k.data[(hk * T + c0 + s) * dk + dkc] * uvec[row * chunkLen + s]
          carry[row * dk + dkc] = acc
      c0 += cLen
    for row in 0 ..< dv:
      for dkc in 0 ..< dk:
        result.state.at(bh, row, dkc) = carry[row * dk + dkc]

# ─── grouped_mm reference ─────────────────────────────────────────────

## Reference spelling of the `at::_grouped_mm` op:
##
##   out[r, i] = El(sum_h a[r, h] · mat2ᵀ[group(r), h, i])   fp32 accumulate
##   El = one round-to-nearest-even round in the family dtype
##
## | term     | contract                                                                                                       |
## | -------- | -------------------------------------------------------------------------------------------------------------- |
## | call     | (a, mat2ᵀ, offs), rows of `a` pair-major grouped by expert id, group g covering rows offs[g-1] ..< offs[g]     |
## | offs     | inclusive per-expert end offsets, non-decreasing, the last entry closes the row range, a repeat an empty group |
## | w        | the pre-transpose cube [E, I, H], the transpose in the call is convention, mat2ᵀ (e, h, i) reads w[e, i, h]    |
## | rounding | one fp32 sequential accumulation per output element, one El round at the store                                 |

type GmmFamily* = enum
  ## 16-bit storage families the grouped GEMM rounds into at the store.
  gmmBf16, gmmF16

proc gmmWiden*(fam: GmmFamily, h: uint16): float32 =
  ## Returns the exact fp32 widening of a family-dtype bit pattern.
  if fam == gmmBf16: bf16ToF32(h) else: fp16ToFp32(h)

proc gmmRoundEl*(fam: GmmFamily, x: float32): uint16 =
  ## Returns the family-dtype round-to-nearest-even bit pattern of an fp32 value,
  ## the one El round at the grouped GEMM's store.
  if fam == gmmBf16: f32ToBf16(x) else: fp32ToFp16(x)

proc gmmName*(fam: GmmFamily): string =
  ## Returns the family dtype's display name.
  if fam == gmmBf16: "bf16" else: "fp16"

proc groupOfRow*(offs: seq[int32], row: int): int =
  ## Returns the expert group owning `row` under the inclusive end-offset contract
  ## -1 for a row past the last offset, a caller bug
  doAssert offs.len > 0 and row >= 0 and row < offs[^1].int,
    "row outside the offsets coverage"
  for g in 0 ..< offs.len:
    if row < offs[g].int:
      return g
  doAssert false, "unreachable"

proc groupedMm*(fam: GmmFamily; a: Mat[uint16]; w: Cube[uint16];
    offs: seq[int32]): Mat[uint16] =
  ## Grouped GEMM over expert groups, one fp32 accumulation and one El
  ## round per output element.
  ##
  ## Expected input:
  ## - `a`, (P, H) family-dtype rows, pair-major grouped by expert id,
  ##   group g covering rows offs[g-1] ..< offs[g]
  ## - `w`, (E, I, H) family-dtype pre-transpose cube, element (e, h, i)
  ##   of mat2ᵀ reading w[e, i, h]
  ## - `offs`, E inclusive end offsets, non-decreasing, offs[E-1] = P,
  ##   a repeated entry an empty group
  ##
  ## Output:
  ## - (P, I) family-dtype rows, out[r, i] = El(sum_h a[r, h]·w[e, i, h])
  ##   over the row's group e, fp32 sequential accumulation
  ##
  ## Example (H = 2, exact small values, bf16):
  ##   a[0] = [1.0, 2.0], w[e, 0, :] = [0.5, 0.25] → out[0, 0] = 1.0
  let rows = a.rows
  doAssert offs.len == w.planes, "one offset per expert"
  doAssert offs[^1].int == rows, "the last offset must close the row range"
  doAssert a.cols == w.cols, "the contraction dim must match the cube's H"
  result = Mat[uint16](rows: rows, cols: w.rows)
  result.data = newSeq[uint16](rows * w.rows)
  for e in 0 ..< w.planes:
    let lo = (if e == 0: 0 else: offs[e - 1].int)
    let hi = offs[e].int
    for r in lo ..< hi:
      for i in 0 ..< w.rows:
        var acc = 0.0'f32
        for h in 0 ..< a.cols:
          acc += gmmWiden(fam, a.data[r * a.cols + h]) *
            gmmWiden(fam, w.data[(e * w.rows + i) * w.cols + h])
        result.data[r * w.rows + i] = gmmRoundEl(fam, acc)

proc groupedMmSums*(fam: GmmFamily; a: Mat[uint16]; w: Cube[uint16];
    offs: seq[int32]): Mat[float32] =
  ## Returns the unrounded fp32 accumulations of `groupedMm`, same contract,
  ## for checks that need the pre-store value.
  doAssert a.cols == w.cols, "the contraction dim must match the cube's H"
  let rows = a.rows
  doAssert offs.len == w.planes, "one offset per expert"
  doAssert offs[^1].int == rows, "the last offset must close the row range"
  result = Mat[float32](rows: rows, cols: w.rows)
  result.data = newSeq[float32](rows * w.rows)
  for e in 0 ..< w.planes:
    let lo = (if e == 0: 0 else: offs[e - 1].int)
    let hi = offs[e].int
    for r in lo ..< hi:
      for i in 0 ..< w.rows:
        var acc = 0.0'f32
        for h in 0 ..< a.cols:
          acc += gmmWiden(fam, a.data[r * a.cols + h]) *
            gmmWiden(fam, w.data[(e * w.rows + i) * w.cols + h])
        result.data[r * w.rows + i] = acc

# ─── Fused decoder layer's stage ops ──────────────────────────────────

## Stage references of the fused GDN decoder layer composition, the stage ops the mega kernel's reference side walks:
##
## - one bias-one RMSNorm pass with residual add, both norm sites one proc
## - q/k l2 normalization, causal conv + silu, recurrence values, GEMV
## - softmax top-K router, MoE activation chain
##
## Storage contract, every proc here:
##
## | rule         | behavior                                                                                                                                                                                             |
## | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | storage      | family-dtype (bf16 or fp16) operands and results are stored as their uint16 bit patterns, widened exactly to fp32 for arithmetic                                                                     |
## | accumulation | fp32 sequential over the row, one RNE bf16 round at each storage handoff the kernel chain rounds at                                                                                                  |
## | rsqrt        | the fp32 rsqrt is the correctly-rounded `1.0 / sqrt(x)`, the Metal approximate `rsqrt` builtin a different rounding class, that difference is a composition-band item, judged by the comparison tier |

# Host libm log1p under its C float32 spelling, std/math spells no log1p.
proc log1p(x: float32): float32 {.importc: "log1pf", header: "<math.h>", noSideEffect.}

const Log2E* = 1.4426950408889634'f32

func softplus*(x: float32): float32 =
  ## Softplus in the ATen `softplus(x, 1, 20)` shape, linear past the threshold, `log(1 + exp(x))` under it.
  if x > 20.0'f32: x else: log1p(exp(x))

func sigmoid*(x: float32): float32 =
  ## Returns `1 / (1 + exp(-x))` in fp32.
  1.0'f32 / (1.0'f32 + exp(-x))

func famWiden*(fam: GmmFamily, h: uint16): float32 =
  ## Returns the exact fp32 widening of a family-dtype bit pattern.
  gmmWiden(fam, h)

func famRound*(fam: GmmFamily, x: float32): uint16 =
  ## Returns the family-dtype round-to-nearest-even bit pattern of an fp32 value.
  gmmRoundEl(fam, x)

proc rmsNormRes*(xPrev, rPrev, w: seq[uint16]; H: int; eps: float32;
    fam: GmmFamily = gmmBf16): tuple[stream, normed: seq[uint16]] =
  ## One bias-one RMSNorm pass over the residual add, both the decoder layer's
  ## stage 1 and stage 11 (the same proc for each norm site).
  ##
  ## Returns:
  ##
  ## | output    | value                                             |
  ## | --------- | ------------------------------------------------- |
  ## | stream    | s[e] = bf16(x[e] + r[e]), the new residual stream |
  ## | acc       | sum_e widen(s[e])², fp32 serial over the row      |
  ## | rstd      | 1/sqrt(acc/H + eps), fp32, no round               |
  ## | normed[e] | bf16(widen(s[e])·rstd·(widen(w[e]) + 1))          |
  ##
  ## Example (H = 2, exact small values)
  ## x = [1.0, 0.0], r = [0.0, 0.0], w = [0.0, 0.0], eps = 0
  ## gives s = [1.0, 0.0], acc = 1.0, rstd = sqrt(2), normed = [sqrt(2), 0.0] up to the store's bf16 round.
  doAssert xPrev.len == H and rPrev.len == H and w.len == H
  result.stream = newSeq[uint16](H)
  result.normed = newSeq[uint16](H)
  var acc = 0.0'f32
  for e in 0 ..< H:
    let s = famRound(fam, famWiden(fam, xPrev[e]) + famWiden(fam, rPrev[e]))
    result.stream[e] = s
    acc += famWiden(fam, s) * famWiden(fam, s)
  let rstd = 1.0'f32 / sqrt(acc / float32(H) + eps)
  for e in 0 ..< H:
    result.normed[e] = famRound(fam,
      famWiden(fam, result.stream[e]) * rstd * (famWiden(fam, w[e]) + 1.0'f32))

proc l2NormRow*(x: seq[uint16]; cols: int; fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One l2-normalized row, the q/k normalization's rounding pipeline.
  ##
  ## Returns:
  ##
  ## | part   | value                                                                |
  ## | ------ | -------------------------------------------------------------------- |
  ## | acc    | sum_c El(widen(x[c])²), each square rounds to the family dtype first |
  ## | sumFam | El(acc), the sum's own round at the model's `.sum` output dtype      |
  ## | inv    | El(1/sqrt(El(widen(sumFam) + 1e-6)))                                 |
  ## | out    | out[c] = El(widen(x[c])·widen(inv))                                  |
  ##
  ## Recorded-chain rounding, none of it belongs to the mathematical l2 norm:
  ## - each square rounds to the family dtype elementwise
  ## - the fp32 sum rounds to the family dtype at the model's `.sum` output dtype
  ## - the eps add rounds again before the reciprocal
  doAssert x.len == cols
  var acc = 0.0'f32
  for c in 0 ..< cols:
    let xi = famWiden(fam, x[c])
    acc += famWiden(fam, famRound(fam, xi * xi))
  let sumBf = famRound(fam, acc)
  let inv = famRound(fam, 1.0'f32 /
    sqrt(famWiden(fam, famRound(fam, famWiden(fam, sumBf) + 1.0e-6'f32))))
  result = newSeq[uint16](cols)
  for c in 0 ..< cols:
    result[c] = famRound(fam, famWiden(fam, x[c]) * famWiden(fam, inv))

proc denseLinear*(x: seq[uint16]; w: seq[uint16]; N, K: int;
    fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One row's dense projection, the GEMV form of an (N, K) row-major weight.
  ##
  ## Returns:
  ## - out[n] = bf16(sum_k widen(x[k])·widen(w[n·K + k])), fp32 sequential
  ##
  ## The kernel's 16-wide mma chunk chain reassociates this sum, the reassociation
  ## budget belongs to the comparison tier.
  result = newSeq[uint16](N)
  for n in 0 ..< N:
    var acc = 0.0'f32
    for k in 0 ..< K:
      acc += famWiden(fam, x[k]) * famWiden(fam, w[n * K + k])
    result[n] = famRound(fam, acc)

proc causalConvSiluStep*(convW: seq[uint16]; ring: var seq[uint16];
    xCol: seq[uint16]; ConvDim, kernel: int; fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One decode conv step over `ConvDim` channels at kernel width `kernel`,
  ## the ring carrying the `kernel - 1` history taps.
  ##
  ## Returns:
  ## - acc[c] = widen(convW[c·kernel + j])·widen(ring[c·(kernel-1) + j]) for j < kernel-1
  ##   plus widen(convW[c·kernel + kernel-1])·widen(xCol[c])
  ## - out[c] = El(silu(El(acc[c])))
  ## - ring[c] is shifted down one slot in place, xCol[c] the newest tap
  ##
  ## The tapped dot rounds once at the family dtype, the silu runs in fp32 over
  ## the widened tapped value, one family-dtype round at the output, per channel
  ## independent of its neighbors.
  let taps = kernel - 1
  doAssert convW.len == ConvDim * kernel
  doAssert ring.len == ConvDim * taps and xCol.len == ConvDim
  result = newSeq[uint16](ConvDim)
  for c in 0 ..< ConvDim:
    var acc = 0.0'f32
    for j in 0 ..< taps:
      acc += famWiden(fam, convW[c * kernel + j]) *
        famWiden(fam, ring[c * taps + j])
    acc += famWiden(fam, convW[c * kernel + taps]) * famWiden(fam, xCol[c])
    let tapped = famRound(fam, acc)
    result[c] = famRound(fam,
      famWiden(fam, tapped) / (1.0'f32 + exp(-famWiden(fam, tapped))))
  for c in 0 ..< ConvDim:
    for j in 0 ..< taps - 1:
      ring[c * taps + j] = ring[c * taps + j + 1]
    ring[c * taps + taps - 1] = xCol[c]

proc gdnGates*(aRow, bRow, dtBias: seq[uint16]; aLog: seq[float32]; H: int;
    fam: GmmFamily = gmmBf16):
    tuple[g: seq[float32], beta: seq[uint16]] =
  ## Recurrence values over the H value heads, one head per index.
  ##
  ## Returns:
  ## - g[h]    = -exp(A_log[h])·softplus(widen(a[h]) + widen(dtBias[h]))
  ##   fp32 end to end, no round
  ## - beta[h] = El(sigmoid(widen(b[h])))
  doAssert aRow.len == H and bRow.len == H and dtBias.len == H and aLog.len == H
  result.g = newSeq[float32](H)
  result.beta = newSeq[uint16](H)
  for h in 0 ..< H:
    let x = famWiden(fam, aRow[h]) + famWiden(fam, dtBias[h])
    result.g[h] = -exp(aLog[h]) * softplus(x)
    result.beta[h] = famRound(fam, sigmoid(famWiden(fam, bRow[h])))

proc softmaxTopKRouter*(x: seq[uint16]; routerW: seq[uint16];
    E, H, K: int; scale: float32; fam: GmmFamily = gmmBf16):
    tuple[ids: seq[int32], w: seq[float32]] =
  ## One token's top-K expert ids and routing weights, the softmax form.
  ##
  ## Returns:
  ##
  ## | step   | value                                                                 |
  ## | ------ | --------------------------------------------------------------------- |
  ## | logits | logits[e] = bf16(sum_k widen(x[k])·widen(routerW[e·H + k])), fp32 dot |
  ## | p      | softmax over the widened logits, fp32                                 |
  ## | top-K  | by score, the lowest index on a tie                                   |
  ## | w      | w[slot] = El(p[id]/sum(top-K p)·scale), renormalized over             |
  ## |        | the selected set only, the model's routeToExperts contract            |
  ##
  ## Example (E = 4, K = 2, exact small values, all logits distinct):
  ##   logits [3.0, 1.0, 2.0, 0.0] → p ∝ [e³, e¹, e², 1] → ids [0, 2],
  ##   w = [e³/(e³+e²), e²/(e³+e²)], each rounded to the family dtype.
  doAssert routerW.len == E * H
  var logits = newSeq[float32](E)
  for e in 0 ..< E:
    var acc = 0.0'f32
    for k in 0 ..< H:
      acc += famWiden(fam, x[k]) * famWiden(fam, routerW[e * H + k])
    logits[e] = famWiden(fam, famRound(fam, acc))
  var p = newSeq[float32](E)
  for e in 0 ..< E:
    p[e] = exp(logits[e] - max(logits))
  result.ids = newSeq[int32](K)
  result.w = newSeq[float32](K)
  var used = newSeq[bool](E)
  var topSum = 0.0'f32
  var picked = newSeq[int](K)
  for slot in 0 ..< K:
    var best = -1
    var bestP = -1.0'f32
    for e in 0 ..< E:
      if not used[e] and (best < 0 or p[e] > bestP):
        bestP = p[e]
        best = e
    used[best] = true
    picked[slot] = best
    result.ids[slot] = int32(best)
    topSum += p[best]
  # renormalize over the selected set only, then one bf16 round per weight
  for slot in 0 ..< K:
    result.w[slot] = famWiden(fam, famRound(fam, p[picked[slot]] / topSum * scale))

proc siluMulEl*(g, u: float32; fam: GmmFamily = gmmBf16): uint16 =
  ## MoE expert activation element, the mega chain's rounding form over the fp32 g/up accumulator operands.
  ##
  ## Returns:
  ## - El(El(silu(g))·u)
  ##
  ## The silu result rounds at the family dtype first, then the product with the fp32 up
  ## operand rounds once at the store.
  let s = g / (1.0'f32 + exp(-g))
  famRound(fam, famWiden(fam, famRound(fam, s)) * u)

proc sharedGate*(x, sharedGateVecW: seq[uint16]; H: int;
    fam: GmmFamily = gmmBf16): float32 =
  ## One token's shared-expert scalar, the sigmoid of the raw fp32 scalar logit,
  ## one bf16 round, returned widened.
  ##
  ## Returns:
  ## - l32 = sum_k widen(x[k])·widen(sharedGateVecW[k]), fp32 sequential
  ## - the returned value = El(sigmoid(l32))
  doAssert sharedGateVecW.len == H
  var l32 = 0.0'f32
  for k in 0 ..< H:
    l32 += famWiden(fam, x[k]) * famWiden(fam, sharedGateVecW[k])
  famWiden(fam, famRound(fam, sigmoid(l32)))

proc rmsNormGated*(y, z, w: seq[uint16]; Dv: int; eps: float32;
    fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One gated RMSNorm row, the o_norm epilogue's rounding chain.
  ##
  ## Returns:
  ##
  ## | step     | value                                                                 |
  ## | -------- | --------------------------------------------------------------------- |
  ## | rstd     | 1/sqrt(sum_d widen(y[d])²/Dv + eps), fp32 squares, fp32 sum, no round |
  ## | normed   | normed[d] = El(widen(y[d])·rstd)                                      |
  ## | weighted | weighted[d] = El(widen(w[d])·widen(normed[d]))                        |
  ## | out[d]   | El(widen(weighted[d])·silu32(widen(z[d])))                            |
  ##
  ## The squares stay fp32 here (unlike the l2-normalized row's family-dtype squares),
  ## matching the o_norm tile core's fp32 square-and-reduce.
  doAssert y.len == Dv and z.len == Dv and w.len == Dv
  var sumSq = 0.0'f32
  for d in 0 ..< Dv:
    let yv = famWiden(fam, y[d])
    sumSq += yv * yv
  let rstd = 1.0'f32 / sqrt(sumSq / float32(Dv) + eps)
  result = newSeq[uint16](Dv)
  for d in 0 ..< Dv:
    let normed = famRound(fam, famWiden(fam, y[d]) * rstd)
    let weighted = famRound(fam, famWiden(fam, w[d]) * famWiden(fam, normed))
    let g = famWiden(fam, z[d])
    let silu = g / (1.0'f32 + exp(-g))
    result[d] = famRound(fam, famWiden(fam, weighted) * silu)

proc moeMerge*(partial: seq[float32]; K, H: int; fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One token's MoE merge, the fp32 partial rows summed in slot order
  ## (the shared contribution last), one family-dtype round at the store.
  ##
  ## Returns:
  ## - out[e] = bf16(sum_slot widen(partial[slot·H + e])), slot order
  doAssert partial.len == (K + 1) * H
  result = newSeq[uint16](H)
  for e in 0 ..< H:
    var acc = 0.0'f32
    for slot in 0 .. K:
      acc += partial[slot * H + e]
    result[e] = famRound(fam, acc)

# ─── One-token layer composition ──────────────────────────────────────

## One token's full GDN decoder layer pass, mirroring the mega kernel's 13-stage order over the stage
## ops above, bf16 storage handoffs between stages, fp32 state.
##
## | stage | role        | stage | role         |
## | ----- | ----------- | ----- | ------------ |
## | 1     | add + norm1 | 8     | GDN step     |
## | 2     | qkv GEMV    | 9     | o_norm       |
## | 3     | z GEMV      | 10    | out_proj     |
## | 4     | a/b GEMV    | 11    | fold + norm2 |
## | 5     | conv + ring | 12    | MoE decode   |
## | 6     | q/k l2norm  | 13    | merge        |
## | 7     | g/beta      |       |              |

const
  RefH* = 2048
    ## Layer width, the norm rows, the MoE hidden and the out_proj rows.
  RefConvDim* = 8192
    ## Fused qkv projection width.
  RefHv* = 32
  RefHk* = 16
  RefDk* = 128
  RefDv* = 128
  RefConvKernel* = 4
  RefTopK* = 8
  RefNumExperts* = 256
  RefInter* = 512
  RefHkRatio* = 2

type LayerOut* = object
  ## One launch's intermediate and output sections, all bf16 bit patterns
  ## except the fp32 fields, named after the mega module's arena sections:
  ##
  ## | field    | section                                        |
  ## | -------- | ---------------------------------------------- |
  ## | stream   | the residual add, the new residual stream (H)  |
  ## | norm1    | norm1 output row (H)                           |
  ## | qkvCol   | fused qkv projection column (ConvDim)          |
  ## | z        | o_norm z row (Hv·Dv)                           |
  ## | a        | decay projection row (Hv)                      |
  ## | b        | beta projection row (Hv)                       |
  ## | conv     | conv + silu output column (ConvDim)            |
  ## | qn       | l2-normalized q heads (Hk·Dk)                  |
  ## | kn       | l2-normalized k heads (Hk·Dk)                  |
  ## | g        | log-decay values (Hv), fp32 end to end         |
  ## | beta     | beta values (Hv)                               |
  ## | y        | GDN core output rows (Hv·Dv)                   |
  ## | normed   | o_norm output rows, the out_proj input (Hv·Dv) |
  ## | blockOut | out_proj row before the fold (H)               |
  ## | h1       | folded residual row (H), the next addend       |
  ## | normed2  | post-LN normed row, the MoE input (H)          |
  ## | moeOut   | merged MoE output row (H)                      |
  stream*: seq[uint16]
  norm1*: seq[uint16]
  qkvCol*: seq[uint16]
  z*: seq[uint16]
  a*: seq[uint16]
  b*: seq[uint16]
  conv*: seq[uint16]
  qn*: seq[uint16]
  kn*: seq[uint16]
  g*: seq[float32]
  beta*: seq[uint16]
  y*: seq[uint16]
  normed*: seq[uint16]
  blockOut*: seq[uint16]
  h1*: seq[uint16]
  normed2*: seq[uint16]
  moeOut*: seq[uint16]

proc moeDecodeBody*(x: seq[uint16];
    routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16];
    fam: GmmFamily = gmmBf16):
    tuple[ids: seq[int32], w: seq[float32], partial: seq[float32], moeOut: seq[uint16]] =
  ## One token's MoE decode body, the mega's slot-group walk over the grouped
  ## GEMM reference:
  ##
  ## | part          | contract                                                                          |
  ## | ------------- | --------------------------------------------------------------------------------- |
  ## | projections   | fp32 sums (the mma accumulator form), one-expert cubes over the weight rows       |
  ## | activation    | h = bf16(bf16(silu(g))·u) per element                                             |
  ## | down walk     | fp32 into the partial row, scaled by the routing weight                           |
  ## | shared expert | the scalar's gated contribution in the last partial row, the merge one bf16 round |
  ##
  ## Returns:
  ## - ids, w, the recomputed router's top-K expert ids and weights
  ## - partial, the (K+1, H) fp32 partial rows in the mega's partial contract
  ## - moeOut, the merged output row
  let (ids, w) = softmaxTopKRouter(x, routerW, RefNumExperts, RefH, RefTopK, 1.0'f32, fam)
  result.partial = newSeq[float32]((RefTopK + 1) * RefH)
  for slot in 0 ..< RefTopK:
    let id = ids[slot].int
    # gate/up walk, one-expert cube over the fused (2I, H) weight rows
    let guData = gateUpW[(id * 2 * RefInter) * RefH ..< ((id + 1) * 2 * RefInter) * RefH]
    let guCube = Cube[uint16](planes: 1, rows: 2 * RefInter, cols: RefH, data: guData)
    let gu = groupedMmSums(fam, Mat[uint16](rows: 1, cols: RefH, data: x), guCube, @[1'i32])
    var hBits = newSeq[uint16](RefInter)
    for i in 0 ..< RefInter:
      hBits[i] = siluMulEl(gu.data[i], gu.data[RefInter + i], fam)
    # down walk, one-expert cube (H, I) over the expert's (H, I) weight rows
    let dnData = downW[id * RefH * RefInter ..< (id + 1) * RefH * RefInter]
    let dnCube = Cube[uint16](planes: 1, rows: RefH, cols: RefInter, data: dnData)
    let dn = groupedMmSums(fam, Mat[uint16](rows: 1, cols: RefInter, data: hBits), dnCube, @[1'i32])
    for e in 0 ..< RefH:
      result.partial[slot * RefH + e] = w[slot] * dn.data[e]
  # shared expert walk, the scalar then the separate projections
  let gateVal = sharedGate(x, sharedGVW, RefH, fam)
  let sg = groupedMmSums(fam, Mat[uint16](rows: 1, cols: RefH, data: x),
    Cube[uint16](planes: 1, rows: RefInter, cols: RefH, data: sharedGW), @[1'i32])
  let su = groupedMmSums(fam, Mat[uint16](rows: 1, cols: RefH, data: x),
    Cube[uint16](planes: 1, rows: RefInter, cols: RefH, data: sharedUW), @[1'i32])
  var hsBits = newSeq[uint16](RefInter)
  for i in 0 ..< RefInter:
    hsBits[i] = siluMulEl(sg.data[i], su.data[i], fam)
  let sd = groupedMmSums(fam, Mat[uint16](rows: 1, cols: RefInter, data: hsBits),
    Cube[uint16](planes: 1, rows: RefH, cols: RefInter, data: sharedDW), @[1'i32])
  for e in 0 ..< RefH:
    result.partial[RefTopK * RefH + e] = gateVal * sd.data[e]
  result.ids = ids
  result.w = w
  result.moeOut = moeMerge(result.partial, RefTopK, RefH, fam)

proc qwen35GdnLayer*(state: var Cube[float32], ring: var seq[uint16];
    x, r: seq[uint16];
    norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: seq[uint16];
    routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16];
    aLog: seq[float32], dtBias: seq[uint16]; eps: float32;
    fam: GmmFamily = gmmBf16): LayerOut =
  ## One token's layer pass over the stage ops above, the mega's 13-stage order,
  ## the state and ring updated in place.
  ##
  ## Returns the LayerOut record, every arena section the comparison tier
  ## reads per stage.
  doAssert x.len == RefH and r.len == RefH, "x/r width mismatch"
  doAssert qkvW.len == RefConvDim * RefH, "qkv weight shape mismatch"
  doAssert convW.len == RefConvDim * RefConvKernel, "conv weight shape mismatch"

  # Stage 1, add + norm1
  let n1 = rmsNormRes(x, r, norm1W, RefH, eps, fam)
  result.stream = n1.stream
  result.norm1 = n1.normed

  # Stage 2, qkv GEMV (one 32-row block at M = 1)
  result.qkvCol = denseLinear(n1.normed, qkvW, RefConvDim, RefH, fam)

  # Stage 3, z GEMV
  result.z = denseLinear(n1.normed, zW, RefHv * RefDv, RefH, fam)

  # Stage 4, a/b GEMVs
  result.a = denseLinear(n1.normed, aW, RefHv, RefH, fam)
  result.b = denseLinear(n1.normed, bW, RefHv, RefH, fam)

  # Stage 5, conv + ring roll, the conv input column the fused qkv projection column
  result.conv = causalConvSiluStep(convW, ring, result.qkvCol, RefConvDim, RefConvKernel, fam)

  # Stage 6, q/k l2norm, 16 heads × 128 each, q from rows 0..2048, k from 2048..4096
  result.qn = newSeq[uint16](RefHk * RefDk)
  result.kn = newSeq[uint16](RefHk * RefDk)
  for h in 0 ..< RefHk:
    result.qn[h * RefDk ..< (h + 1) * RefDk] =
      l2NormRow(result.conv[h * RefDk ..< (h + 1) * RefDk], RefDk, fam)
    result.kn[h * RefDk ..< (h + 1) * RefDk] =
      l2NormRow(result.conv[(RefHk * RefDk) + h * RefDk ..< (RefHk * RefDk) + (h + 1) * RefDk], RefDk, fam)

  # Stage 7, g/beta
  let gates = gdnGates(result.a, result.b, dtBias, aLog, RefHv, fam)
  result.g = gates.g
  result.beta = gates.beta

  # Stage 8, GDN step, the value rows read from the conv column's value channels
  var qMat = Mat[float32](rows: RefHk, cols: RefDk)
  qMat.data = newSeq[float32](RefHk * RefDk)
  var kMat = Mat[float32](rows: RefHk, cols: RefDk)
  kMat.data = newSeq[float32](RefHk * RefDk)
  var vMat = Mat[float32](rows: RefHv, cols: RefDv)
  vMat.data = newSeq[float32](RefHv * RefDv)
  var betaF = newSeq[float32](RefHv)
  for i in 0 ..< RefHk * RefDk:
    qMat.data[i] = gmmWiden(fam, result.qn[i])
    kMat.data[i] = gmmWiden(fam, result.kn[i])
  for bh in 0 ..< RefHv:
    betaF[bh] = gmmWiden(fam, result.beta[bh])
    for d in 0 ..< RefDv:
      vMat.data[bh * RefDv + d] =
        gmmWiden(fam, result.conv[(2 * RefHk * RefDk) + bh * RefDv + d])
  var yMat = Mat[float32](rows: RefHv, cols: RefDv)
  yMat.data = newSeq[float32](RefHv * RefDv)
  gdnDecodeStep(state, yMat, qMat, kMat, vMat, betaF, result.g, RefHv, RefHk, RefHkRatio)
  result.y = newSeq[uint16](RefHv * RefDv)
  for i in 0 ..< RefHv * RefDv:
    result.y[i] = gmmRoundEl(fam, yMat.data[i])

  # Stage 9, o_norm, one gated RMSNorm row per value head
  result.normed = newSeq[uint16](RefHv * RefDv)
  for bh in 0 ..< RefHv:
    result.normed[bh * RefDv ..< (bh + 1) * RefDv] = rmsNormGated(
      result.y[bh * RefDv ..< (bh + 1) * RefDv],
      result.z[bh * RefDv ..< (bh + 1) * RefDv],
      onormW[bh * RefDv ..< (bh + 1) * RefDv], RefDv, eps, fam)

  # Stage 10, out_proj
  result.blockOut = denseLinear(result.normed, outprojW, RefH, RefHv * RefDv, fam)

  # Stage 11, fold + norm2, the fold's sum the new residual
  let n2 = rmsNormRes(result.stream, result.blockOut, norm2W, RefH, eps, fam)
  result.h1 = n2.stream
  result.normed2 = n2.normed

  # Stages 12 + 13, the MoE decode then the merge
  let moe = moeDecodeBody(result.normed2, routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW, fam)
  result.moeOut = moe.moeOut
