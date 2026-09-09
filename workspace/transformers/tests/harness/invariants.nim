# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Analytic invariants for transformer ops.
##
## Each proc expresses one property that must hold for any correct
## implementation. Property mode checks seeded random inputs. Wiring mode
## checks committed fixtures. Every check echoes its worst deviation
## before asserting, so a failure report carries the margin. Slack
## parameters are explicit: bf16 tensors round to 1 ulp = 2^-8 relative.
## f32 accumulations drift 1-2 ulp (bench_rmsnorm.nim evidence). Defaults
## leave headroom for honest rounding, far below fault corpus sizes
## listed in selftest.nim.
##
## Slacks used by the checks (detection floors are in selftest.nim):
## - RMSNorm definition: rtol 1e-2 on bf16 fixture outputs, 1e-6 in f32
## - RMSNorm scale preservation: 3-sigma kurtosis slack, see SPEC.md
## - RoPE pair norms: rtol 2e-2 on bf16, 1e-6 in f32
## - softmax row sums: slack 1e-2 on bf16 logits, 1e-5 in f32

import
  std/math,
  workspace/libtorch as F,
  workspace/libtorch_testutils

proc rmsNormReference*(x, weight: Tensor, eps: float64,
    biasOne = false): Tensor =
  ## RMSNorm recomputed at f32 from the same inputs over the last
  ## dimension. biasOne = false scales with x·w·rsqrt(mean(x²)+eps).
  ## biasOne = true means the checkpoint stores a weight offset. The scale
  ## is (1 + w) as in RmsNormOne.
  let x32 = x.to(F.kFloat32)
  let w32 = weight.to(F.kFloat32)
  let rstd = x32.square().mean(axis = -1, keepdim = true, dtype = F.kFloat32)
    .add(eps).rsqrt()
  if biasOne:
    (x32 * rstd) * (w32 + 1.0'f32)
  else:
    (x32 * rstd) * w32

proc checkRmsNormDefinition*(
    x, weight, output: Tensor,
    eps: float64,
    biasOne = false,
    rtol = 1e-6'f64, abstol = 1e-6'f64,
    msg = "rmsnorm definition") =
  ## RMSNorm output against its defining identity, recomputed in f32.
  ## bf16 fixture outputs carry bf16 rounding, pass rtol 1e-2 there.
  let expected = rmsNormReference(x, weight, eps, biasOne)
  assertAllClose(output.to(F.kFloat32), expected,
    rtol = rtol, abstol = abstol, msg = msg)

proc checkRmsNormScalePreservation*(
    x, weight, output: Tensor,
    biasOne = false,
    msg = "rmsnorm scale preservation") =
  ## RMS(out) ≈ RMS(scale) for uncorrelated x and weight, with correlation
  ## slack. scale is w, or 1 + w for the bias-one checkpoint layout.
  ## Identity: RMS(x·s)/RMS(x) = RMS(s) holds only
  ## in expectation over uncorrelated x. The slack is a 3-sigma bound.
  ## Its relative variance is (kx·ks + kx − 1)/d at normalized width d.
  ## kx and ks are the empirical kurtosis E[z^4]/E[z^2]^2 of x
  ## and of the scale. A 1% uniform output fault stays inside this slack
  ## for moderate d. Fault detection for norm outputs uses
  ## checkRmsNormDefinition instead. See SPEC.md.
  let x32 = x.to(F.kFloat32)
  let out32 = output.to(F.kFloat32)
  let scale = if biasOne: weight.to(F.kFloat32).add(1.0'f32)
              else: weight.to(F.kFloat32)
  let rmsOut = out32.square().mean().sqrt().item(float64)
  let rmsScale = scale.square().mean().sqrt().item(float64)
  # Empirical kurtosis (E[x^4]/E[x^2]^2) of x and the scale. The finite
  # sample deviation of RMS(x·s)/RMS(x)/RMS(s) from 1 is ~3 sigma.
  # Relative variance is (kx·ks + kx − 1)/d at normalized width d.
  let kx = x32.square().square().mean().item(float64) /
    pow(x32.square().mean().item(float64), 2.0)
  let ks = scale.square().square().mean().item(float64) /
    pow(scale.square().mean().item(float64), 2.0)
  let d = float64(x.size(-1))
  let slack = 3.0 * sqrt((kx * ks + kx - 1.0) / d)
  let dev = abs(rmsOut / rmsScale - 1.0)
  echo "  ", msg, ": |RMS(out)/RMS(scale) - 1| = ", dev, " (slack ", slack, ")"
  doAssert dev <= slack, msg & ": scale drift " & $dev & " exceeds slack " & $slack

proc checkRopePairNormPreservation*(
    q, k, qRot, kRot: Tensor,
    rotaryDim: int,
    rtol = 2e-2'f64, abstol = 2e-2'f64,
    msg = "rope pair norms") =
  ## RoPE preserves the norm of each NEOX rotation pair (i, i+half)
  ## within the rotated columns. Pass-through columns are checked
  ## bit-exact by the suites. Position-0 identity check:
  ## checkRopePositionZeroIdentity.
  doAssert rotaryDim mod 2 == 0, "rotaryDim must be even"
  for (inT, rotT, name) in [(q, qRot, "q"), (k, kRot, "k")]:
    let rIn = inT.to(F.kFloat32).narrow(-1, 0, rotaryDim)
    let rOut = rotT.to(F.kFloat32).narrow(-1, 0, rotaryDim)
    let half = rotaryDim div 2
    let normIn = rIn.narrow(-1, 0, half).square()
      .add(rIn.narrow(-1, half, half).square())
    let normOut = rOut.narrow(-1, 0, half).square()
      .add(rOut.narrow(-1, half, half).square())
    assertAllClose(normOut, normIn, rtol = rtol, abstol = abstol,
      msg = msg & " (" & name & ")")

proc checkRopePositionZeroIdentity*(
    q, k, qRot, kRot, sin: Tensor,
    rotaryDim: int,
    msg = "rope position-0 identity") =
  ## Position 0 rotates by identity: cos row 0 is 1 and sin row 0 is 0
  ## (asserted precondition), so the rotated columns at sequence position 0
  ## are bit-exact equal. Expected layout: (batch, seq, head, head_dim).
  let sin0 = sin.to(F.kFloat32).narrow(0, 0, 1)
  doAssert sin0.abs().max().item(float64) == 0.0,
    msg & ": precondition sin row 0 must be all zero"
  for (inT, rotT, name) in [(q, qRot, "q"), (k, kRot, "k")]:
    let rotSlice = rotT.narrow(1, 0, 1).narrow(-1, 0, rotaryDim)
    let inSlice = inT.narrow(1, 0, 1).narrow(-1, 0, rotaryDim)
    assertAllClose(rotSlice, inSlice, rtol = 0.0, abstol = 0.0,
      msg = msg & " (" & name & ")")

proc checkFwhtParseval*(
    input: Tensor,
    fwht: proc(x: var Tensor) {.closure.},
    rtol = 1e-5'f64,
    msg = "fwht parseval") =
  ## Parseval identity for the unnormalized FWHT: ||H·x||² = d·||x||²
  ## blockwise, checked in f32.
  let x32 = input.to(F.kFloat32)
  var y = x32.clone()
  fwht(y)
  let energyIn = x32.square().sum().item(float64)
  let energyOut = y.square().sum().item(float64)
  let dim = float64(x32.size(-1))
  let dev = abs(energyOut / energyIn - dim) / dim
  echo "  ", msg, ": relative energy deviation = ", dev
  doAssert dev <= rtol, msg & ": parseval deviation " & $dev

proc checkSoftmaxRowSums*(
    output: Tensor,
    slack = 1e-5'f64,
    msg = "softmax row sums") =
  ## Every softmax output row sums to 1 within slack, computed in f32.
  let out32 = output.to(F.kFloat32)
  let worst = out32.sum(axis = -1).add(-1.0'f32).abs().max().item(float64)
  echo "  ", msg, ": worst row-sum deviation = ", worst, " (slack ", slack, ")"
  doAssert worst <= slack, msg & ": row-sum deviation " & $worst & " exceeds slack " & $slack
