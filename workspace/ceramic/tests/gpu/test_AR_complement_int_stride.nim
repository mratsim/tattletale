## Ceramic × Crucible anti-regression: `Int[N]` stride + `local_partition` on GPU
##
## Failure mode this test prevents:
##   Ceramic's complement path (via `local_partition`) emits `max(1, flatten(stride))`
##   where the flattened scalar stride is a static `Int[1]`. Crucible treats `max`
##   as an "ambiguous builtin" and FORWARDS it to CUDA verbatim WITHOUT lowering the
##   static `Int[N]` value, so the operand is the emitted empty struct and NVRTC
##   rejects the kernel:
##     no instance of overloaded function "max" matches the argument list
##     argument types are: (int, Int1)
##   This is the sgemm_1 port blocker: any layout complement that touches a static
##   Int stride emits a kernel that cannot NVRTC-compile.
##
## Why the empty type is NOT the bug:
##   `Int[N]` is a compile-time type-level integer: its value lives in the type
##   parameter, so emitting it as an empty struct (`struct Int1 { char _; };`) is
##   correct and intentional — exactly like CuTe's `Int<N>`. It only becomes a
##   problem when something uses it as a RUNTIME number without the value being
##   lowered first. The fix is crucible lowering the static value at emission.
##
## This is a real end-to-end slice of the ceramic partition path (view + layout
## with a static Int stride + local_partition), exercised through a `cuda:` kernel
## compiled and run by NVRTC.
##
## Anti-regression contract: the kernel must compile and run, i.e. crucible
## must lower the static Int value so `max(1, stride)` resolves instead of
## reaching CUDA as `max(1, <empty struct>)` (which fails NVRTC compilation).
##
## Run:
##   cd tattletale
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##     nim cpp -r --hints:off --warnings:off \
##       --outdir:build/tests/gpu --nimcache:nimcache/tests/gpu \
##       workspace/ceramic/tests/gpu/test_AR_complement_int_stride.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc
import workspace/crucible/src/codegen/gpu_compiler
import workspace/ceramic/src/int_tuples {.all.}
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

const kernelPartition = cuda:
  proc partKernel(Buf: ptr UncheckedArray[float32]) {.global.} =
    # View over Buf with a static Int[1] stride on the leading mode.
    let a = make_view(Buf, make_layout((8, 8), (Int[1](), 8)))
    let tl = make_layout((Int[8](), Int[8]()))
    # local_partition -> complement -> max(1, Int1) at emission.
    let t = local_partition(a, tl, int(threadIdx.x))
    Buf[0] = 1.0'f32   # written only if the kernel compiles and runs

suite "Ceramic × Crucible — complement with a static Int stride":

  test "NVRTC compiles and runs a local_partition with an Int[1] stride":
    # The complement must emit concrete ints (static Int value lowered) so the
    # kernel compiles and executes; a regression causes nv.compile() to abort.
    var Buf: array[64, float32]
    var nv = initNvrtc(kernelPartition)
    nv.numBlocks = 1
    nv.threadsPerBlock = 8
    nv.compile()                 # aborts with NVRTC_ERROR_COMPILATION on a max(1, <empty struct>) regression
    nv.getPtx()
    nv.execute("partKernel", Buf, ())
    check Buf[0] == 1.0'f32