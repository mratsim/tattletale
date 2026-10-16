# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## launchDeviceFn static/env classification:
## - a phantom-typed formal (its value lives in the type) bakes into
##   the {.global.} bridge body and contributes no bridge kernel param
## - a pointer formal (tensor view analog) stays an env member, delivered as
##   a device buffer
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_phantom_bake.nim

import std/strutils
import workspace/crucible

type
  Phantom[N: static int] = object
    discard

template phantomValue[N: static int](p: Phantom[N]): int = N

proc addPhantom(outBuf, inBuf: ptr UncheckedArray[uint32], x: Phantom[8]) =
  outBuf[0] = inBuf[0] + uint32(phantomValue(x))

proc kernelSignatures(artifact: string): seq[string] =
  ## `kernel void` entry lines of an ingested Metal artifact
  for line in artifact.splitLines():
    if line.startsWith("kernel void"):
      result.add(line)

proc runTest() =
  var engine = bkMetal.init()
  let inDev = engine.allocDevice(sizeof(uint32))
  let outDev = engine.allocDevice(sizeof(uint32))
  var hostIn: array[1, uint32] = [100'u32]
  engine.copyToDeviceFromHost(inDev, addr hostIn[0], sizeof(uint32))

  engine.launchDeviceFn addPhantom(outDev, inDev, Phantom[8]()) do:
    grid = (1, 1)
    blk = (1, 1)

  var res: array[1, uint32]
  engine.copyToHostFromDevice(addr res[0], outDev, sizeof(uint32))
  doAssert res[0] == 108'u32,
    "the kernel must add the phantom formal's type-carried value: got " & $res[0]

  let sigs = kernelSignatures(engine.getArtifact())
  doAssert sigs.len == 1, "expected exactly one kernel entry, got: " & $sigs.len
  let sig = sigs[0]
  doAssert "outBuf [[buffer(0)]]" in sig and "inBuf [[buffer(1)]]" in sig,
    "the pointer formals must stay env members, bound as device buffers: " & sig
  doAssert "Phantom8" notin sig,
    "the phantom formal must not be a bridge kernel param, " &
    "its value lives in the type and bakes into the bridge body: " & sig

  block:  # a `static T` formal bakes the same way, the value crosses in the bridge body
      proc addStatic(outBuf, inBuf: ptr UncheckedArray[uint32], k: static int32) =
        outBuf[0] = inBuf[0] + uint32(k)
      engine.launchDeviceFn addStatic(outDev, inDev, 5) do:
        grid = (1, 1)
        blk = (1, 1)
      engine.copyToHostFromDevice(addr res[0], outDev, sizeof(uint32))
      doAssert res[0] == 105'u32,
        "the kernel must add the static formal's baked value: got " & $res[0]
      let staticArt = engine.getArtifact()
      let staticSig = kernelSignatures(staticArt)[0]
      doAssert "constant int& k" notin staticSig and "constant int k" notin staticSig,
        "the static formal must not be a bridge kernel param: " & staticSig
      doAssert "(uint)5" in staticArt,
        "the static formal's value must bake into the bridge body: " & staticArt

  engine.freeDevice(inDev)
  engine.freeDevice(outDev)

when isMainModule:
  runTest()
