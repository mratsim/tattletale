# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal budget-row measurement instrument.
##
## The budget table v0 rows were measured with CPU computations.
## This bench measures the same per-op rows when the computation runs
## on the M4 Max Metal device: the production path computes on Metal
## over the same input bytes, the cpu output of the identical path
## serves as the reference, and the ulp report gives the per-element
## count of ulp distances against the row caps. A cap-respecting row replaces
## the provisional value with the measured cap. A row past its cap must stop
## the change until the cause is found (a stale assumption, a code defect, or
## the wrong model), never a reason to widen the row.
##
## Derivation: one op computes once, so a single-op
## cross-device drift is one reordering step in the 1 to 2 bf16 ulp
## class of the same-device floors. The chain replay measurements show
## per-block cross-device drift at or under 0.16 of the activation
## bulk over 18 checkpoints. A single-op drift therefore stays inside
## a fraction of one band. Prediction: the per-element ulp distances sit at
## the same-device floors, the 2 ulp / 4 ulp caps hold, mismatch fractions
## far below 1e-3.
##
## Run manually, never in the sweep (the filename carries no t_ prefix):
##
##   nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on \
##     --debugger:native --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests/rel --nimcache:nimcache/tests/rel \
##     workspace/transformers/tests/harness/bench_metal_budget.nim
##   LD_LIBRARY_PATH=.venv/lib:.venv/lib/python3.14/site-packages/torch/lib \
##     ./build/tests/rel/bench_metal_budget

import
  std/importutils,
  std/options,
  std/os,
  std/strformat,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

from workspace/libtorch/src/raw_libtorch import manual_seed

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  RopeFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Qwen3.5-0.8B-layer-3"
  ChainFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" /
    "Qwen3-0.6B"
  ChainSeqLen = 6
  MaxSeq = 256

proc ulpCensus(actual, expected: Tensor, capUlps: int):
    tuple[maxUlp: int64, past: int] =
  ## Per-element count of bf16 ulp distances, the Metal output against the
  ## cpu reference: worst ulp distance plus the element count past the cap.
  let a = actual.contiguous().to(F.kCPU)
  let e = expected.contiguous().to(F.kCPU)
  doAssert a.numel() == e.numel(), "ulp count element count mismatch"
  doAssert a.scalarType() == F.kBfloat16, "the ulp count expects bf16 tensors"
  doAssert e.scalarType() == F.kBfloat16, "the ulp count expects bf16 tensors"
  let av = a.view(F.kInt16).contiguous()
  let ev = e.view(F.kInt16).contiguous()
  let ra = cast[ptr UncheckedArray[int16]](av.data_ptr(int16))
  let re = cast[ptr UncheckedArray[int16]](ev.data_ptr(int16))
  let n = a.numel()
  for i in 0 ..< n:
    let fa = bf16ToF32(uint16(ra[i]))
    let fe = bf16ToF32(uint16(re[i]))
    if fa == fe: continue
    let d = ulpDistance16(fa, fe)
    if d > result.maxUlp: result.maxUlp = d
    if d > capUlps: inc result.past

proc quantileDrift(actual, expected: Tensor): int64 =
  ## Returns the worst quantile ulp distance of the two fingerprints, the
  ## quantity the ctDistribution rows cap.
  let sa = tensorStats(actual, withHistogram = false)
  let se = tensorStats(expected, withHistogram = false)
  for i in 0 ..< QuantileCount:
    let d = ulpDistance16(sa.quantiles[i], se.quantiles[i])
    if d > result: result = d

proc chainBandUsage(actual, expected: Tensor, depth: int): float64 =
  ## Returns the band usage of the cross-device drift under the chain
  ## checkpoint row at the given depth, the bulk-anchored absolute term.
  let mw = maxBandWidths(actual, expected, ChainCheckpointRtol,
    chainCheckpointAbstol(depth, meanAbsValue(expected)))
  mw.worst

proc histL1(actual, expected: Tensor): float64 =
  ## Normalized histogram L1 between the two fingerprints.
  normalizedL1(tensorStats(actual), tensorStats(expected))

proc report(name: string, capUlps: int, capFrac: float64,
    c: tuple[maxUlp: int64, past: int], n: int, q: int64,
    histL1 = -1.0, chainUsage = -1.0) =
  ## One report line per measured row. The verdict names the row class
  ## the number falls in: the same-device ulp caps, the quantile
  ## fingerprint, or the cross-device chain band.
  let frac = c.past.float64 / n.float64
  var line = &"{name:<16} maxUlp {c.maxUlp:>4} (cap {capUlps}), mismatch past cap " &
    &"{c.past}/{n} = {frac:.2e} (cap {capFrac}), quantiles {q} ulp"
  if histL1 >= 0.0:
    line.add &", hist L1 {histL1:.2e}"
  if chainUsage >= 0.0:
    line.add &", depth-1 chain band usage {chainUsage:.3f}"
  if c.maxUlp <= capUlps.int64 and frac <= capFrac:
    line.add ", inside the same-device row"
  else:
    line.add ", cross-device class: bulk-anchored rows apply"
  echo line

proc main() =
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  let dev = testDevice()
  echo "measurement device: " & deviceName(dev)
  if dev == F.kCPU:
    echo "no Metal device under TTT_TEST_ON, the per-op Metal rows cannot measure here"
    return

  # obRmsNorm: production RmsNorm, Metal computation vs cpu computation
  # over identical inputs. Row cap: 2 ulp, zero mismatch.
  Torch.manual_seed(0x5EED'u64)
  block:
    let x = F.randn(256, 1024, F.tensorOptions(F.kBFloat16, F.kCPU))
    let w = F.randn(1024, F.tensorOptions(F.kBFloat16, F.kCPU))
    let refOut = RmsNorm.init(w).forward(x)
    let got = RmsNorm.init(w.to(dev)).forward(x.to(dev))
    let c = ulpCensus(got, refOut, 2)
    report("obRmsNorm", 2, 0.0, c, refOut.numel(),
      quantileDrift(got, refOut))

  # obRope: production applyRopeImpl over the recorded rope fixture,
  # Metal computation vs the recorded cpu q_rot. Row cap: 2 ulp, zero
  # mismatch.
  block:
    var st = Safetensor.open(RopeFixtureDir / "rope-Qwen3.5-0.8B-00.safetensor")
    let q = st.getTensorOwned("q").to(dev)
    let k = st.getTensorOwned("k").to(dev)
    let cos = st.getTensorOwned("cos").to(dev)
    let sin = st.getTensorOwned("sin").to(dev)
    let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
    let qExpected = st.getTensorOwned("q_rot")
    let kExpected = st.getTensorOwned("k_rot")
    let cq = ulpCensus(qRot, qExpected, 2)
    report("obRope q", 2, 0.0, cq, qExpected.numel(),
      quantileDrift(qRot, qExpected))
    let ck = ulpCensus(kRot, kExpected, 2)
    report("obRope k", 2, 0.0, ck, kExpected.numel(),
      quantileDrift(kRot, kExpected))

  # obAttention: SDPA over identical inputs, Metal computation vs cpu
  # computation. Row cap: 4 ulp, 1e-3 mismatch fraction.
  Torch.manual_seed(0x5EED'u64 + 1)
  block:
    let q = F.randn(2, 8, 64, 128, F.tensorOptions(F.kBFloat16, F.kCPU))
    let k = F.randn(2, 2, 64, 128, F.tensorOptions(F.kBFloat16, F.kCPU))
    let v = F.randn(2, 2, 64, 128, F.tensorOptions(F.kBFloat16, F.kCPU))
    let refOut = F.scaled_dot_product_attention(q, k, v, enable_gqa = true)
    let got = F.scaled_dot_product_attention(
      q.to(dev), k.to(dev), v.to(dev), enable_gqa = true)
    let c = ulpCensus(got, refOut, 4)
    report("obAttention", 4, 1e-3, c, refOut.numel(),
      quantileDrift(got, refOut),
      histL1 = histL1(got, refOut),
      chainUsage = chainBandUsage(got, refOut, 1))

  # obPostResidual: one 0.6B decoder block on Metal over the recorded
  # block input, against the recorded cpu block output. Row cap: 4 ulp,
  # 1e-3 mismatch fraction.
  block:
    var st0 = Safetensor.open(ChainFixtureDir / "block-00.safetensor")
    let hidden = st0.getTensorOwned("after_attn_norm_residual").to(dev)
    let expected = st0.getTensorOwned("hf_layer_output")
    let model = loadQwen3ModelRaw(ModelPath, dev)
    var ctx = InferenceContext.init(
      num_layers = 1, batch_size = 1,
      kv_heads = model.config.num_key_value_heads,
      max_seq = MaxSeq, head_dim = model.config.head_dim)
    let pool = PagePool.init(
      64, num_layers = 1,
      kv_heads = model.config.num_key_value_heads,
      head_dim = model.config.head_dim,
      dtype = F.kBFloat16, device = dev)
    let numPages = ceilDiv(MaxSeq, TokensPerPage)
    for _ in 0 ..< numPages:
      ctx.pages.add(pool.borrow())
    ctx.kv_position = 0
    ctx.position_ids = F.arange(ChainSeqLen,
      F.tensorOptions(F.kInt64, F.kCPU))
    ctx.setRopeForPositions(model.layers[0].sequence_mixer.rotary)
    let (blkOut, res) = model.layers[0](ctx, hidden, none(Tensor))
    let checkpoint = blkOut + res
    let c = ulpCensus(checkpoint, expected, 4)
    report("obPostResidual", 4, 1e-3, c, expected.numel(),
      quantileDrift(checkpoint, expected),
      histL1 = histL1(checkpoint, expected),
      chainUsage = chainBandUsage(checkpoint, expected, 1))

when isMainModule:
  main()
