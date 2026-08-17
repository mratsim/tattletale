## Metal engine loud-quit regressions: ingest and dispatch failures
## exit non-zero with the engine's message on stderr. Each case builds as a separate binary
## selected with a -d: define. The default case compiles the 31-binding boundary kernel
## and dispatches it, exiting zero.
##
##   -d:caseBindings  32-binding kernel quits at ingest, before the Metal compile error
##   -d:caseBlk       blk 2048 quits with the Apple Silicon thread limit message
##   -d:caseGrid      grid 0 quits with the grid validation message
##   -d:caseScalar    24-byte scalar quits with the 16-byte-slot message
##   (default)        31-binding kernel compiles and dispatches, exit zero
##
## Run each case like the command below, with the -d: define swapped in.
## The non-zero exit code and the engine message on stderr are the assertions.
##   nim c -r --hints:off --warnings:off -d:caseBindings \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_quit_bindings.nim

import workspace/crucible

const addMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void addKernel(device uint* output [[buffer(0)]],
                      device const uint* a [[buffer(1)]],
                      device const uint* b [[buffer(2)]])
{
  output[0] = a[0] + b[0];
}
"""

# Hand-written MSL: checkBindingLimit locates kernels by the literal "kernel " token,
# which must match the printer's emission or the ingest check silently stops firing.
proc gen32Buffers(): string =
  ## 32 buffer bindings: output [[buffer(0)]] plus 31 inputs at indices 1..31.
  result = """
#include <metal_stdlib>
using namespace metal;

kernel void tooMany(device uint* out [[buffer(0)]],
"""
  for i in 0 ..< 31:
    result.add "  device const uint* a" & $i & " [[buffer(" & $(i + 1) & ")]],\n"
  result.add "  uint3 tid [[thread_position_in_threadgroup]])\n{\n  out[0] = 1u;\n}\n"

proc gen31Buffers(): string =
  ## 31 buffer bindings: output [[buffer(0)]] plus 30 inputs at indices 1..30,
  ## the legal boundary (Metal's limit is 31 bindings at indices 0..30).
  result = """
#include <metal_stdlib>
using namespace metal;

kernel void boundary(device uint* out [[buffer(0)]],
"""
  for i in 0 ..< 30:
    result.add "  device const uint* a" & $i & " [[buffer(" & $(i + 1) & ")]],\n"
  result.add "  uint3 tid [[thread_position_in_threadgroup]])\n{\n  out[0] = 1u;\n}\n"

type
  BigScalar = object
    ## 24 bytes: larger than the 16-byte constant-buffer slot.
    a, b, c, d, e, f: float32

proc runTest() =   # private: the -d: define selects the case and engines are destroyed at return
  when defined(caseBindings):
    # A 32-buffer kernel must quit at ingest, before Metal's own compile error appears.
    var engine = bkMetal.init()
    engine.ingest(gen32Buffers())
    echo "UNEXPECTED: ingest succeeded with 32 buffers"
  elif defined(caseBlk):
    # blk 2048 exceeds the Apple Silicon limit of 1024 threads per threadgroup.
    var engine = bkMetal.init()
    engine.ingest(addMsl)
    var res: array[1, uint32]
    engine.run<<(1, 2048)>>("addKernel", res, ([2'u32], [3'u32]))
    echo "UNEXPECTED: run succeeded with blk 2048"
  elif defined(caseGrid):
    # grid axis 0 must be rejected before any dispatch.
    var engine = bkMetal.init()
    engine.ingest(addMsl)
    var res: array[1, uint32]
    engine.run<<(0, 64)>>("addKernel", res, ([2'u32], [3'u32]))
    echo "UNEXPECTED: run succeeded with grid 0"
  elif defined(caseScalar):
    # Reject a 24-byte scalar at packing time, before it corrupts the next slot.
    var engine = bkMetal.init()
    engine.ingest(addMsl)
    var res: array[1, uint32]
    var big: BigScalar
    engine.run("addKernel", res, (big, 7'i32))
    echo "UNEXPECTED: run succeeded with a 24-byte scalar"
  else:
    # 31-binding legal boundary: the ingest check passes, Metal compiles the kernel,
    # and the dispatch succeeds.
    var engine = bkMetal.init()
    engine.ingest(gen31Buffers())
    var res: array[1, uint32]
    var ins = newSeq[uint32](30)
    engine.run("boundary", res, (ins,))
    if res[0] != 1'u32:
      stderr.write("UNEXPECTED: boundary kernel returned " & $res[0] & ", expected 1\n")
      quit 1
    echo "OK: 31-binding kernel compiled and dispatched"

when isMainModule:
  runTest()
