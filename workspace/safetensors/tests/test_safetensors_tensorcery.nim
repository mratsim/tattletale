# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Safetensors loading into tensorcery tensors (owned form).
##
## Blocks:
## - CPU: load the F64 vandermonde fixture, check shape, dtype, device
##   and values through the erased face
## - Metal: transfer the F32 fixture, check devMetal as the device
## - BF16: check the refusal names the dtype
## Save blocks:
## - Roundtrip: F64+F32 through saveSafetensors, compare values
## - Device: serialize a Metal-resident tensor, reload on CPU, compare
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --nimcache:nimcache/t_st workspace/safetensors/tests/test_safetensors_tensorcery.nim

import std/os
import std/math
import std/importutils
import std/strutils

import pkg/iface
import workspace/safetensors/src/safetensors {.all.}
import workspace/safetensors/src/collections
import workspace/safetensors/src/safetensors_tensorcery
import workspace/data_structures/src/small_seqs
import workspace/tensorcery
import workspace/tensorcery/src/backends/metal

privateAccess(SafetensorObj)
privateAccess(Tensor)
privateAccess(TensorImpl[float32, bkCpu])
privateAccess(TensorImpl[float64, bkCpu])
privateAccess(TensorImpl[float32, bkMetal])

proc main() =
  let fixturePath = currentSourcePath().parentDir() / "fixtures" / "vandermonde.safetensors"
  doAssert fileExists(fixturePath), "fixture not found: " & fixturePath

  block cpuLoadF64:
    var st = Safetensor.open(fixturePath)
    doAssert st.hasTensor("F64_vandermonde_5x5")

    let ctx = Ctx()
    let t = getTensorOwned(ctx, st, "F64_vandermonde_5x5")
    doAssert t.getShape() == [5'i32, 5'i32]
    doAssert t.getDtype() == dtFloat64
    doAssert t.getDevice() == devCpu

    # v[i, j] = i^(j+1), i and j from 1
    let back = t.impl.to(TensorImpl[float64, bkCpu])
    for i in 0 ..< 5:
      for j in 0 ..< 5:
        let expected = pow(float64(i + 1), float64(j + 1))
        doAssert back.storage.data[i * 5 + j] == expected,
            "vandermonde[" & $i & "," & $j & "]"

  block metalTransfer:
    var st = Safetensor.open(fixturePath)
    let ctx = Ctx()
    ensureMetal(ctx)
    let t = getTensorOwned(ctx, st, "F32_vandermonde_5x5", bkMetal)
    doAssert t.getDevice() == devMetal
    doAssert t.getDtype() == dtFloat32
    doAssert t.getShape() == [5'i32, 5'i32]

  block unsupportedDtypeRefuses:
    var st = Safetensor.open(fixturePath)
    let ctx = Ctx()
    try:
      discard getTensorOwned(ctx, st, "BF16_vandermonde_5x5")
      doAssert false, "BF16 load should have raised"
    except AssertionDefect as e:
      doAssert "BF16" in e.msg, "refusal must name the dtype, got: " & e.msg

  block saveLoadRoundtrip:
    let ctx = Ctx()
    var st = Safetensor.open(fixturePath)
    let v64 = getTensorOwned(ctx, st, "F64_vandermonde_5x5")
    let v32 = getTensorOwned(ctx, st, "F32_vandermonde_5x5")

    let outPath = getTempDir() / "tensorcery_roundtrip.safetensors"
    saveSafetensors(outPath, [("f64_v", v64), ("f32_v", v32)])
    doAssert fileExists(outPath)

    var rt = Safetensor.open(outPath)
    doAssert rt.hasTensor("f64_v")
    doAssert rt.hasTensor("f32_v")
    let back64 = getTensorOwned(ctx, rt, "f64_v")
    let back32 = getTensorOwned(ctx, rt, "f32_v")
    doAssert back64.getShape() == [5'i32, 5'i32]
    doAssert back32.getShape() == [5'i32, 5'i32]
    doAssert back64.getDtype() == dtFloat64
    doAssert back32.getDtype() == dtFloat32

    let a = toSeq[float64](back64)
    let b = toSeq[float64](v64)
    for i in 0 ..< a.len:
      doAssert a[i] == b[i], "roundtrip value mismatch at " & $i
    let c = toSeq[float32](back32)
    let d = toSeq[float32](v32)
    for i in 0 ..< c.len:
      doAssert c[i] == d[i], "roundtrip value mismatch at " & $i

  block saveMetalWeights:
    let ctx = Ctx()
    ensureMetal(ctx)
    var st = Safetensor.open(fixturePath)
    let m = getTensorOwned(ctx, st, "F32_vandermonde_5x5", bkMetal)
    let outPath = getTempDir() / "tensorcery_metal_save.safetensors"
    saveSafetensors(outPath, "metal_w", m)

    var rt = Safetensor.open(outPath)
    let ctx2 = Ctx()
    let back = getTensorOwned(ctx2, rt, "metal_w")
    let a = toSeq[float32](back)
    let b = toSeq[float32](getTensorOwned(ctx, st, "F32_vandermonde_5x5"))
    for i in 0 ..< a.len:
      doAssert a[i] == b[i], "metal save value mismatch at " & $i

  echo "test_safetensors_tensorcery: ALL GREEN"

main()
