## Minimal repro: evalOnceAs-generated constexpr leaks into let RHS
##
## Suspect pattern (what make_layout's evalOnceAs expands to):
##   let L =
##     const tmp {.genSym.} = (Int[8](), Int[16]())
##     tmp
##
## Crucible may emit this as:
##   Layout_... L = constexpr Tuple_... tmp = {{}, {}};
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_repro_tensorview_codegen.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

type
  Int*[V: static int] = object
    discard

type
  Tuple2*[A, B] = object
    f0: A
    f1: B

# ── Pattern A: direct tuple ──
const kernelDirect = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = Tuple2[Int[8], Int[16]]()
    C[0] = 1'u32

# ── Pattern B: const + let (simulates evalOnceAs) ──
const kernelConstLet = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

# ── Pattern C: let with block that defines const then yields value ──
#   This is what evalOnceAs actually expands to under the hood.
const kernelBlock = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
      tmp
    C[0] = 1'u32

suite "Crucible - TensorView codegen repro":
  test "Pattern A — direct tuple let":
    var output: array[1, uint32]
    var nv = initNvrtc(kernelDirect)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 1

  test "Pattern B — const + let":
    let code = kernelConstLet
    echo code
    var nv = initNvrtc(code)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    check true

  test "Pattern C — block with const then yield":
    let code = kernelBlock
    echo code
    var nv = initNvrtc(code)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    check true
