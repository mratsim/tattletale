# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## launchDeviceFn cache contract:
## - a literal passed to a non-static formal stays an env member
## - two launches of one kernel with the literals 1 and 2 hit the same
##   cached {.global.} bridge (the artifact is not re-ingested) and produce
##   their respective results
## - baking per-literal would monomorphize one bridge per distinct value
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_literal_bridge.nim

import std/strutils
import workspace/crucible

proc addScalar(outBuf, inBuf: ptr UncheckedArray[uint32], k: int32) =
  outBuf[0] = inBuf[0] + uint32(k)

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

  engine.launchDeviceFn addScalar(outDev, inDev, 1) do:
    grid = (1, 1)
    blk = (1, 1)
  var res1: array[1, uint32]
  engine.copyToHostFromDevice(addr res1[0], outDev, sizeof(uint32))
  doAssert res1[0] == 101'u32, "first launch must add the literal 1: got " & $res1[0]
  let art1 = engine.getArtifact()

  engine.launchDeviceFn addScalar(outDev, inDev, 2) do:
    grid = (1, 1)
    blk = (1, 1)
  var res2: array[1, uint32]
  engine.copyToHostFromDevice(addr res2[0], outDev, sizeof(uint32))
  doAssert res2[0] == 102'u32, "second launch must add the literal 2: got " & $res2[0]

  doAssert engine.getArtifact() == art1,
    "both launches must reuse one cached bridge, the artifact must not be re-ingested"
  let sigs = kernelSignatures(art1)
  doAssert sigs.len == 1, "expected exactly one kernel entry, got: " & $sigs.len
  doAssert "constant int& k" in sigs[0],
    "the literal's formal must stay an env member bridge param, not bake: " & sigs[0]

  block:  # a second kernel gets its own bridge, both reachable in one artifact run
      proc addOffset(outBuf, inBuf: ptr UncheckedArray[uint32], k: int32) =
        outBuf[0] = inBuf[0] + uint32(k) + 7'u32
      engine.launchDeviceFn addOffset(outDev, inDev, 1) do:
        grid = (1, 1)
        blk = (1, 1)
      var res3: array[1, uint32]
      engine.copyToHostFromDevice(addr res3[0], outDev, sizeof(uint32))
      doAssert res3[0] == 108'u32,
        "the second kernel must compute its own result: got " & $res3[0]
      let sigs2 = kernelSignatures(engine.getArtifact())
      doAssert sigs2.len == 1, "each ingest replaces the artifact: " & $sigs2.len
      doAssert "addOffset" in sigs2[0] and "addScalar" notin sigs2[0],
        "the second kernel's bridge is a distinct kernel: " & sigs2[0]

  engine.freeDevice(inDev)
  engine.freeDevice(outDev)

when isMainModule:
  runTest()
