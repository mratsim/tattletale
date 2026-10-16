# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Value-lowered magic builtin: `system.default(T)` inside a kernel body:
##
## - `default` is `{.magic: "Default"}`: no Nim body to parse, no backend name to forward
## - a kernel call therefore lowers to the zero-init of the result type, phantom or scalar
## - before the fix the call reached the external-proc registration path
## - that path accepts `{.builtin.}`/`{.importc.}` procs only, and the assert fired
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_default_magic.nim

import workspace/crucible

type
  StaticInt[N: static int] = object
    ## Phantom scalar: the value lives in the type parameter, the runtime
    ## struct is empty (the `Int[N]` shape of ceramic tile layouts).

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  block:
    const defaultMagicMsl = metal:
      proc defaultMagicKernel(C: ptr UncheckedArray[float32]) {.global.} =
        let s = default(StaticInt[16])
        let t = default((StaticInt[16], StaticInt[16]))
        let z = default(int32)
        C[0] = float32(z) + 1.0'f32

    var engine = bkMetal.init()
    engine.ingest(defaultMagicMsl)
    let oDev = engine.allocDevice(sizeof(float32))
    engine.launchKernel("defaultMagicKernel", oDev, ())
    var res: array[1, float32]
    engine.copyToHostFromDevice(addr res[0], oDev, sizeof(float32))
    engine.freeDevice(oDev)
    doAssert res[0] == 1.0'f32, "the zero-init of int32 must be 0, got " & $res[0]
    echo "  OK: default(T) on a phantom, a phantom tuple, and a scalar"

when isMainModule:
  runTest()
  echo "  All default-magic-builtin tests passed."
