# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Safetensors loading into tensorcery tensors (owned form).
##
## - CPU: load the F64 vandermonde fixture, check shape, dtype, device
##   and values through the erased face
## - Metal: transfer the F32 fixture, check devMetal as the device
## - BF16: check the refusal names the dtype
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

  echo "test_safetensors_tensorcery: ALL GREEN"

main()
