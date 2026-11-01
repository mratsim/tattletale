## Metal refuses an assignment into relocated program-scope constant storage.
##
## - a `{.const_mem.}` local prints as a `constant` global
## - a later assignment to that name refuses in the constant-write validation
## - v1 had no check: the write reached the emitted `constant` global and died
##   in the Metal compiler with no crucible diagnostic
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_const_mem_write_refused.nim

import workspace/crucible

const code = metal:
  proc k(outBuf: ptr UncheckedArray[float32]) {.global.} =
    var g {.const_mem.}: float32 = 3.0'f32
    g = 5.0'f32
    outBuf[0] = g

proc runTest() =
  # ingest must refuse: the kernel assigns into read-only program-scope storage, and Metal rejects `g = 5.0f` on a `constant float` global
  var engine = newEngine(runMetal)
  engine.ingest(code)
  echo "UNREACHABLE — the constant-storage write must be refused at compile time"

when isMainModule:
  runTest()
