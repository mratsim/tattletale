## CuTe: forward-referenced types (B23)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_cute_circular_type.nim
##
## Tests that types referencing each other (defined in order at Nim level)
## resolve correctly in GPU codegen. True circular types (A contains B
## contains A) have infinite size and are caught by the Nim compiler.
##
## Note: `ref` and GC types don't exist on GPU, so pointer-based cycles
## use `ptr` (raw CUDA pointer) instead.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

# Forward-referenced types (defined at Nim level, adjacent in scope)
type
  Node[N: static int] = object
    val: uint32

const kernelCode = cuda:
  proc fwdRefKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Node[4](val: 42'u32)
    output[0] = x.val

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("fwdRefKernel", buf, ())
doAssert buf[0] == 42, &"fwd ref: {buf[0]}"
echo "  OK — forward-referenced types (B23)"
