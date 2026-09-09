# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Analytic invariants on seeded random inputs and committed fixtures.
## Property mode catches algorithmic breaks independent of recordings.
## Wiring mode checks the same properties on real checkpoint tensors.
##
## nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/t_harness_invariants \
##   --nimcache:nimcache/tests/t_harness_invariants \
##   workspace/transformers/tests/harness/t_harness_invariants.nim

import
  std/options,
  std/os,
  std/sequtils,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/positron/src/kernels/portable/hadamard_transforms {.all.},
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/tests/harness

from workspace/libtorch/src/raw_libtorch import manual_seed

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures"
  LayerFixtureDir = FixtureDir / "bf16-01-layer-internals" / "Qwen3.5-0.8B-layer-3"

proc main() =
  runCppTest "property: RMSNorm definition identity in f32":
    proc(): bool =
      Torch.manual_seed(0x5EED'u64)
      let x = F.randn(8, 256, F.kFloat32)
      let w = F.randn(256, F.kFloat32)
      let got = F.rms_norm(x, 256, w, eps = 1e-6)
      checkRmsNormDefinition(x, w, got, eps = 1e-6)
      true

  runCppTest "property: RMSNorm scale preservation with correlation slack":
    proc(): bool =
      let x = F.randn(8, 256, F.kFloat32)
      let w = F.randn(256, F.kFloat32)
      let got = F.rms_norm(x, 256, w, eps = 1e-6)
      checkRmsNormScalePreservation(x, w, got, biasOne = true)
      true

  runCppTest "wiring: RMSNorm definition on the Qwen3.5 norm fixture":
    proc(): bool =
      var st = Safetensor.open(
        LayerFixtureDir / "norm-Qwen3.5-0.8B-00.safetensor")
      let x = st.getTensorOwned("input")
      let w = st.getTensorOwned("weight")
      let got = st.getTensorOwned("output")
      checkRmsNormDefinition(x, w, got, eps = 1e-6, biasOne = true,
        rtol = 1e-2, abstol = 1e-2)
      checkRmsNormScalePreservation(x, w, got, biasOne = true)
      true

  runCppTest "property: RoPE pair-norm preservation in f32":
    proc(): bool =
      Torch.manual_seed(0x5EED'u64)
      let q = F.randn(1, 8, 8, 64, F.kFloat32)
      let k = F.randn(1, 8, 2, 64, F.kFloat32)
      let pos = F.arange(8, F.kFloat32).view(8, 1)
      let theta = F.arange(32, F.kFloat32).view(1, 32).add(0.1'f32)
      let angles = pos * theta                   # (8, 32)
      let cosHalf = F.cos(angles)
      let sinHalf = F.sin(angles)
      let cos = F.cat([cosHalf, cosHalf], -1)    # (8, 64), NEOX repetition
      let sin = F.cat([sinHalf, sinHalf], -1)
      let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
      checkRopePairNormPreservation(q, k, qRot, kRot, rotaryDim = 64,
        rtol = 1e-5, abstol = 1e-5)
      checkRopePositionZeroIdentity(q, k, qRot, kRot, sin, rotaryDim = 64)
      true

  runCppTest "wiring: RoPE pair norms and position-0 identity on the rope fixture":
    proc(): bool =
      var st = Safetensor.open(LayerFixtureDir / "rope-Qwen3.5-0.8B-00.safetensor")
      let q = st.getTensorOwned("q")            # (1, 8, 8, 256)
      let k = st.getTensorOwned("k")            # (1, 8, 2, 256)
      let cos = st.getTensorOwned("cos")        # (8, 64)
      let sin = st.getTensorOwned("sin")        # (8, 64)
      let qRot = st.getTensorOwned("q_rot")
      let kRot = st.getTensorOwned("k_rot")
      checkRopePairNormPreservation(q, k, qRot, kRot, rotaryDim = 64)
      checkRopePositionZeroIdentity(q, k, qRot, kRot, sin, rotaryDim = 64)
      true

  runCppTest "property: FWHT Parseval in f32":
    proc(): bool =
      Torch.manual_seed(0x5EED'u64)
      let x = F.randn(256, 128, F.kFloat32)
      checkFwhtParseval(x, fwht_128)
      true

  runCppTest "wiring: hadamard rotate energy preservation on the fixture":
    proc(): bool =
      var st = Safetensor.open(FixtureDir / "exl3-00-hadamard" /
        "hadamard_single_block.safetensor")
      let input = st.getTensorOwned("input")
      let suh = st.getTensorOwned("suh")
      let expNone = st.getTensorOwned("output_none")
      let expPre = st.getTensorOwned("output_pre")
      # Energy preservation through the unnormalized FWHT. With the default
      # 1/sqrt(128) norm, ||out||^2 = ||in||^2. With pre_scale the input
      # energy scales elementwise before the transform.
      let in32 = input.to(F.kFloat32)
      let none32 = expNone.to(F.kFloat32)
      let eIn = in32.square().sum().item(float64)
      let eNone = none32.square().sum().item(float64)
      echo "  none-case energy ratio deviation: ",
        abs(eNone / eIn - 1.0)
      doAssert abs(eNone / eIn - 1.0) <= 2e-2,
        "output_none energy not preserved"
      let pre32 = (in32 * suh.to(F.kFloat32))
      let ePre = expPre.to(F.kFloat32).square().sum().item(float64)
      let eScaled = pre32.square().sum().item(float64)
      echo "  pre-case energy ratio deviation: ",
        abs(ePre / eScaled - 1.0)
      doAssert abs(ePre / eScaled - 1.0) <= 2e-2,
        "output_pre energy not equal to scaled-input energy"
      true

  runCppTest "property: softmax row sums in f32 and bf16":
    proc(): bool =
      Torch.manual_seed(0x5EED'u64)
      let logits32 = F.randn(4, 1024, F.kFloat32)
      checkSoftmaxRowSums(F.softmax(logits32, dim = -1), slack = 1e-5)
      let logits16 = F.randn(4, 1024, F.kFloat32).to(F.kBFloat16)
      checkSoftmaxRowSums(F.softmax(logits16, dim = -1), slack = 1e-2)
      true

when isMainModule:
  main()
