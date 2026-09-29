## copyFrom / size() inside cuda: — narrow down the AST pattern
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/ceramic/tests/gpu/test_copyFrom.nim

import std/[unittest]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_copy_registry
import workspace/ceramic/src/hardware/h_copy_properties
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/ceramic/src/kernels/k_layout_copy_gpu

# All static — should work
const test1 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((Int[8](), Int[16]()))
    var tv = make_view(C, L)
    let sz = size(tv)
    for i in 0 ..< sz: discard

# shape with runtime component from kernel param
const test2 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]; N: int32) {.global.} =
    let sh = (Int[8](), Int[16](), int(N))
    let st = (Int[1](), Int[8](), int(N))
    let L = make_layout(sh, st)
    var tv = make_view(C, L)
    let sz = size(tv)
    for i in 0 ..< sz: discard

# copyFrom with all-static layout — should work
const test3 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((Int[32](), Int[8]()))
    var tv = make_view(C, L)
    copyFrom(tv, tv)

# copyFrom with runtime-sized layout
const test4 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]; N: int32) {.global.} =
    let sh = (Int[32](), Int[8](), int(N))
    let L = make_layout(sh, (Int[1](), Int[32](), int(N)))
    var tv = make_view(C, L)
    copyFrom(tv, tv)

# The gemm_cta copy leg, one copy-atom chunk per predicate unit,
# then the group commit and wait resolved from the element type
# (the universal atom abstraction).
# - CUDA takes the async sm80 cp.async atom, committed and waited in groups
# - OpenCL takes the universal blocking atom, a portable elementwise chunk copy,
#   the commit and wait slots discard
# The chunk views stand in for partition_S
# and partition_D, the real smem/gmem tile views carry the addresses.
# The legs use different element types, Nim caches a generic instantiation
# per (proc, T) so the first backend that instantiates getCopyAsyncAtom(T)
# fixes its atom for the whole compilation.
# The OpenCL leg instantiates float32 (the universal blocking atom),
# the CUDA leg uint32 (the sm80 atom).
const gemmCopyLegOpencl = opencl:
  proc copyLegKernel(gmemA: ptr UncheckedArray[float32]) {.global.} =
    const
      tileM = 32
      tileK = 32
      packedW = 4        # the chunk width in float32, 16 div sizeof(float32)
      blockSize = 128
      units = tileM * tileK div (packedW * blockSize)
    var smem {.smem.}: array[tileM * tileK, float32]
    var dstChunks = make_view(addr smem[0], make_layout((1, units)))
    let srcChunks = make_view(gmemA, make_layout((1, units)))
    var pred: array[tileM * tileK div (packedW * blockSize), bool]
    let predChunks = make_view(addr pred[0], make_layout((1, units)))
    copyFromIfAsync(dstChunks, srcChunks, predChunks)
    float32.commit_group()
    float32.wait_group(0)

const gemmCopyLegCuda = cuda:
  proc copyLegKernel(gmemA: ptr UncheckedArray[uint32]) {.global.} =
    # units, the thread's cp.async chunks,
    # (tileM * tileK) div (numPacked(CpAsyncAtom[uint32]) * 128) = 1024 div 512
    const
      tileM = 32
      tileK = 32
      packedW = 4        # numPacked(CpAsyncAtom[uint32]), the chunk width
      blockSize = 128
      units = tileM * tileK div (packedW * blockSize)
    var smem {.smem.}: array[tileM * tileK, uint32]
    var dstChunks = make_view(addr smem[0], make_layout((1, units)))
    let srcChunks = make_view(gmemA, make_layout((1, units)))
    var pred: array[tileM * tileK div (packedW * blockSize), bool]
    let predChunks = make_view(addr pred[0], make_layout((1, units)))
    copyFromIfAsync(dstChunks, srcChunks, predChunks)
    uint32.commit_group()
    uint32.wait_group(0)

suite "size() / copyFrom in cuda:":
  test "all-static size":
    discard cstring(test1)

  test "mixed static/dynamic size":
    discard cstring(test2)

  test "copyFrom all-static":
    discard cstring(test3)

  test "copyFrom with runtime component":
    discard cstring(test4)

  test "gemm copy leg compiles for CUDA":
    discard cstring(gemmCopyLegCuda)

  test "gemm copy leg compiles for OpenCL":
    discard cstring(gemmCopyLegOpencl)

  test "commit_group rejects a wait-kind atom":
    doAssert not compiles(commit_group(SM80_CP_ASYNC_WAIT))

  test "wait_group rejects a copy-kind atom and honors the wait depth":
    doAssert not compiles(wait_group(SM80_CP_ASYNC_COMMIT, 0))
    doAssert not compiles(wait_group(SM80_CP_ASYNC_WAIT, 3))
    doAssert compiles(wait_group(SM80_CP_ASYNC_WAIT, 2))

  test "the universal blocking atom discards commit and wait":
    doAssert compiles(commit_group(UNIVERSAL_COPY))
    doAssert compiles(wait_group(UNIVERSAL_COPY, 0))
