## Metal: `continue` in device code refuses at compile time.
##
## Test that a `metal:` kernel with a `continue` inside a for loop fails to compile
## with the frontend refusal: mid-loop jumps are unsupported in device code.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_continue_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var acc = 0'u32
    for i in 0 ..< 10:
      acc = acc + 1'u32
      continue
    output[0] = acc

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(code)
  echo "UNREACHABLE — continue in device code must refuse at compile time"

when isMainModule:
  runTest()
