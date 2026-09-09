# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run: nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on
##   --hints:off --warnings:off --passC:"-std=c++20"
##   --outdir:build/tests/rel --nimcache:nimcache/tests/rel -r
##   workspace/transformers/tests/synthetic/t_block_sparse_batch_property.nim [cpu|mps]

import
  std/math,
  std/strformat,
  std/os,
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/ffn

proc ulpBf16(m: float32): float32 =
  ## One bf16 ulp at magnitude m (7 significand bits), zero maps to zero.
  if m <= 0.0'f32:
    return 0.0'f32
  result = pow(2.0'f32, floor(log2(m)) - 7.0'f32)

proc buildMoe(device: F.DeviceKind): GatedBlockSparseFFN =
  let opts = F.tensorOptions(F.kBFloat16, device)
  let gateUpProj = F.randn(12, 48, 32, opts)
  let downProj = F.randn(12, 32, 24, opts)
  let routerWeight = F.randn(12, 32, opts)
  let sharedExpert = GatedDenseFFN.init(
    F.randn(24, 32, opts), F.randn(24, 32, opts), F.randn(32, 24, opts))
  let sharedGateWeight = F.randn(1, 32, opts)
  GatedBlockSparseFFN.init(
    gateUpProj, downProj, routerWeight, sharedExpert, sharedGateWeight, 3)

proc batchVsSingles(moe: GatedBlockSparseFFN, t: int, device: F.DeviceKind): bool =
  ## forward([T, H]) must equal row-by-row forward([1, H]): the batched
  ## prefill path against the single-token decode path on identical rows.
  let opts = F.tensorOptions(F.kBFloat16, device)
  let hidden = F.randn(t, 32, opts)
  let batchOut = moe.forward(hidden)

  var worstDiff = 0.0'f32
  var worstMag = 0.0'f32
  for i in 0 ..< t:
    let row = hidden.narrow(0, i.int64, 1)
    let singleOut = moe.forward(row)
    let diff = (batchOut.narrow(0, i.int64, 1) - singleOut).abs()
    worstDiff = max(worstDiff, diff.max().item(float32))
    worstMag = max(worstMag, singleOut.abs().max().item(float32))
  # Implementation-variant roundoff: the batched path accumulates the
  # experts in routing order, the per-row path in ascending-expert order,
  # budget two bf16 ulps at the output magnitude.
  let budget = 2.0'f32 * ulpBf16(worstMag)
  echo &"  T={t}: worst diff {worstDiff:.3e}, budget {budget:.3e} (max mag {worstMag:.1f})"
  result = worstDiff <= budget

proc main() =
  let device =
    if paramCount() > 0 and paramStr(1) == "mps": F.kMPS
    else: F.kCPU
  echo &"Block-sparse FFN batch-vs-single property, device {device}"
  var failures = 0
  for draw in 1 .. 4:
    echo &"weight draw {draw}:"
    let moe = buildMoe(device)
    for t in [2, 5, 9]:
      if not batchVsSingles(moe, t, device):
        failures += 1
        echo &"  FAIL at draw {draw}, T={t}"
  if failures == 0:
    echo "block-sparse batch property: all cases PASS"
  else:
    echo &"block-sparse batch property: {failures} case(s) FAIL"
    quit(1)

when isMainModule:
  main()
