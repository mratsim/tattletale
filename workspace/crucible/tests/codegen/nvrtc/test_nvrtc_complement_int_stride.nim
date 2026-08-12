## NVRTC anti-regression — crucible-only (stdlib + crucible, NO ceramic import)
##
## Failure mode this test prevents:
##   Crucible emits a static `Int[N]` (a compile-time type-level integer) as an
##   EMPTY struct and FORWARDS `max` — an "ambiguous builtin" (nim_builtins.nim:14)
##   — to CUDA's native max WITHOUT lowering the static `Int[N]` value. A layout
##   complement then produces `max(1, <int value>)` whose operand is the empty
##   struct, and NVRTC rejects it:
##     no instance of overloaded function "max" matches the argument list
##     argument types are: (int, Int1)
##   This blocks any kernel whose complement/layout path includes a static Int
##   stride (this is the sgemm_1 port blocker).
##
## Why the empty type is NOT the bug:
##   `Int[N]` is a compile-time type-level integer: its value lives in the type
##   parameter, so emitting it as an empty struct is correct and intentional —
##   exactly like CuTe's `Int<N>`. It only becomes a problem when something uses
##   it as a RUNTIME number without the value being lowered first.
##
## How the test detects the failure mode:
##   It NVRTC-compiles a minimal kernel built from the same frontend shape as
##   ceramic's `complementScalar` (a `max(int, Int[V]) -> int` overload so Nim
##   accepts `max(1, Int[1])`, plus an untyped binder `max(1, stride)` spliced
##   into a plain proc that crucible pulls in). If crucible stops lowering the
##   static value, engine.compile() fails with the NVRTC error above. When the value
##   is lowered the kernel compiles and runs.
##
## Anti-regression contract: the kernel must compile and run with a concrete
## `max(1, stride) = 1`, i.e. crucible must lower the static `Int[N]` value at
## max emission. A regression that forwards `max(1, <empty struct>)` makes
## engine.compile() fail with the NVRTC error above.
##
## Run:
##   cd tattletale
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##     nim cpp -r --hints:off --warnings:off \
##       --outdir:build/tests/nvrtc --nimcache:nimcache/tests/nvrtc \
##       workspace/crucible/tests/codegen/nvrtc/test_nvrtc_complement_int_stride.nim

import std/[unittest, macros]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines
import workspace/crucible/src/codegen/gpu_compiler

# Int[N]-style empty object: value lives only in the static type parameter.
type StaticInt[N: static int] = object

# genBinOp(max)-equivalent overload: lets Nim accept max(int, StaticInt[V]).
func max[V: static int](a: int; b: StaticInt[V]): int = max(a, V)

type Layout = object
  stride: StaticInt[1]

# complementScalar-equivalent: gap = max(1, stride) as an UNTYPED node.
macro myComplementNode(st: typed): untyped =
  newCall(bindSym"max", newLit(1), st)

# Plain proc pulled in by crucible; its body carries the untyped max node.
proc complementLike(l: Layout): auto =
  myComplementNode(l.stride)    # -> max(1, l.stride) forwarded by crucible

# The partition size this kernel computes. When the static Int value is lowered,
# the complement resolves and the size is concrete; if the value is not lowered
# the generated CUDA is `max(1, StaticInt1{})` and NVRTC fails to compile.
const kernelComplement = cuda:
  proc kComplement(Buf: ptr UncheckedArray[int32]) {.global.} =
    let l = Layout(stride: StaticInt[1]())
    let r = complementLike(l)   # crucible pulls body -> max(1, IntN1{})
    Buf[0] = int32(r)           # concrete value when the static Int is lowered

suite "crucible — complement via a static Int stride (max emission)":

  test "NVRTC compiles and runs a complement with a static Int stride":
    # The static value must be lowered so max(1, stride) resolves to 1 and the
    # kernel compiles; a regression causes engine.compile() to abort below.
    var Buf: array[1, int32]
    var engine = bkCuda.init()
    engine.ingest(kernelComplement)
    engine.run<<(1, 1)>>("kComplement", Buf, ())
    check Buf[0] == 1