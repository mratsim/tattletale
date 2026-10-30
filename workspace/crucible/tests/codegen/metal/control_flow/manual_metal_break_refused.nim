## Metal: `break` in device code refuses at compile time.
##
## Test that a `metal:` kernel with a `break` inside a loop fails to compile
## with the frontend refusal, mid-loop exits are unsupported in device code.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_break_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var i = 0'u32
    while true:
      if i == 4'u32:
        break
      i = i + 1'u32
    output[0] = i

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(code)
  echo "UNREACHABLE — break in device code must refuse at compile time"

when isMainModule:
  runTest()
