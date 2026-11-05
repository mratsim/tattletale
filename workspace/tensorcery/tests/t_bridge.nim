# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## BridgeTensor boundary: translation into the kernel image and one Metal op
## path staged through it.
##
## Test that toView applies the padded image contract with shape padding 1s,
## strides padding 0s and step keeping the last real axis stride.
##
## Test that Metal gelu over a slice view staged through BridgeTensor computes
## the tanh approximation at the view's offset.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_bridge.nim

import std/math

import
  workspace/tensorcery,
  workspace/tensorcery/src/kernels/k_view,
  workspace/tensorcery/src/raw/bridge,
  workspace/tensorcery/src/backends/metal,
  workspace/data_structures/src/small_seqs

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

proc main() =
  block toViewContract:
    var shape: SmallSeq[4, int32]
    shape = SmallSeq[4, int32].initFrom(@[4'i32, 8])
    var strides: SmallSeq[4, int32]
    strides = SmallSeq[4, int32].initFrom(@[8'i32, 1])
    let b = BridgeTensor[float32](base: 0x1000'u64, offset: 7, shape: shape, strides: strides)
    let v = b.toView()
    doAssert v.base == 0x1000'u64
    doAssert v.off == 7
    doAssert v.shape == [4'i32, 8, 1, 1], "shape pads with 1s"
    doAssert v.strides == [8'i32, 1, 0, 0], "strides pad with 0s"
    doAssert v.step == 1, "step keeps the last real axis stride"

  block metalGeluThroughBridge:
    # gelu over a slice view: the bridge stages the view offset into the kernel
    let n = 8
    var xdata: seq[float32]
    for i in 0 ..< n:
      xdata.add frand(i + 11)
    let ctx = Ctx()
    let x = toTensor(ctx, xdata, bkMetal, n)
    let v = x.slice(2 .. 7)
    let outT = gelu(v)
    ctx.eval()

    var expected: seq[float64]
    for i in 2 ..< n:
      let u = 0.7978845608028654 * (float64(xdata[i]) + 0.044715 * float64(xdata[i]) *
          float64(xdata[i]) * float64(xdata[i]))
      expected.add 0.5 * float64(xdata[i]) * (1.0 + tanh(u))
    for i in 0 ..< 6:
      doAssert abs(float64(toSeq[float32](outT)[i]) - expected[i]) < 1e-5,
          "bridged gelu element " & $i

  stdout.writeLine "t_bridge: OK"

main()
