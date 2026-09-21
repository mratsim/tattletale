# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Plain delta-rule reference for the positron naive test tier, the g = 0
## reduction target of both delta-rule families:
##
##   S ← S + k ⊗ (β·(v − S·k))
##   y ← S'·(q/√Dk)
##
## An independent spelling with no decay term, a g = 0 run of either gated
## family must match it element for element.

import std/math
import naive_tensors

proc deltaRuleWalk*[F: float32|float64](
    state: var NaiveCube[F],   # (B·Hv, Dv, Dk), updated in place
    y: var NaiveCube[F],       # (B·Hv, T, Dv)
    q, k: NaiveCube[F],        # (B·Hk, T, Dk)
    v: NaiveCube[F],           # (B·Hv, T, Dv)
    beta: NaiveMat[F],         # (B·Hv, T) beta per value head
    Hv, Hk, hkRatio: int) =
  ## Walks the plain delta-rule recurrence over T tokens, state in place:
  ##
  ## Contract:
  ## - kv[r] = Σ_dk state[r, dk]·k[dk], delta[r] = β·(v[r] − kv[r])
  ## - state[r, dk] += k[dk]·delta[r]
  ## - y[r] = Σ_dk state[r, dk]·(q[dk]/√Dk)
  let bhMax = state.planes
  let dv = state.rows
  let dk = state.cols
  let T = beta.cols
  doAssert q.planes == Hk and q.rows == T and q.cols == dk, "q shape mismatch"
  doAssert k.planes == Hk and k.rows == T and k.cols == dk, "k shape mismatch"
  doAssert v.planes == bhMax and v.rows == T and v.cols == dv, "v shape mismatch"
  doAssert y.planes == bhMax and y.rows == T and y.cols == dv, "y shape mismatch"
  doAssert beta.rows == bhMax, "beta head count mismatch"
  doAssert hkRatio >= 1, "hkRatio must be at least 1"
  let qScale = sqrt(F(dk))
  var kvRow = newSeq[F](dv)
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for t in 0 ..< T:
      for row in 0 ..< dv:
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += state.at(bh, row, dkc) * k.data[(hk * T + t) * dk + dkc]
        kvRow[row] = acc
      for row in 0 ..< dv:
        let delta = beta.data[bh * T + t] * (v.data[(bh * T + t) * dv + row] - kvRow[row])
        for dkc in 0 ..< dk:
          state.at(bh, row, dkc) = state.at(bh, row, dkc) +
            k.data[(hk * T + t) * dk + dkc] * delta
        var acc: F = 0
        for dkc in 0 ..< dk:
          acc += state.at(bh, row, dkc) *
            (q.data[(hk * T + t) * dk + dkc] / qScale)
        y.at(bh, t, row) = acc
