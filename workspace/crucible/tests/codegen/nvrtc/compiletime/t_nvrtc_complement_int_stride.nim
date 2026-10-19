## NVRTC: layout complement with a static Int[N] stride value (crucible only)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/compiletime/t_nvrtc_complement_int_stride.nim
import std/[unittest, macros]
import workspace/crucible

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "crucible — complement via a static Int stride (max emission)":

    test "NVRTC compiles and runs a complement with a static Int stride":
      # The static value must be lowered so max(1, stride) resolves to 1 and the
      # kernel compiles. A regression causes engine.ingest() to abort below.
      var Buf: array[1, int32]
      var engine = bkCuda.init()
      engine.ingest(kernelComplement)
      let oDev = engine.allocDevice(int32, Buf.len)
      engine.launchKernel<<(1, 1)>>("kComplement", oDev, ())
      engine.copyToHostFromDevice(addr Buf[0], oDev, Buf.len)
      check Buf[0] == 1

when isMainModule:
  runTest()
