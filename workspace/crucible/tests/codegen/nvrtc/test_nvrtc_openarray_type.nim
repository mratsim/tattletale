## Regression: `ntyOpenArray` not supported in type resolver
##
## Two kernels, both calling an external func with `openArray[T]`:
##
##   kernel 1 — double a full openArray
##   kernel 2 — double a half-view via the `+%` template
##              (ceramic's ptr_arithmetic.nim), which expands to
##              toOpenArray(data, off, data.len - 1)
##
## Bug: resolvers.nim `resolveType` has no `of ntyOpenArray:` arm.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_openarray_type.nim
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

# ─────────────────────────────────────────────────────────────────
#  Func that doubles every element of an openArray
#  Like ceramic's make_view(data: openArray[T], ...)
# ─────────────────────────────────────────────────────────────────

proc doubleElements(output: ptr UncheckedArray[float32];
                    data: openArray[float32]) =
  for i in 0 ..< data.len:
    output[i] = data[i] * 2.0'f32

# ─────────────────────────────────────────────────────────────────
#  +% template: exact copy from ceramic's ptr_arithmetic.nim
#    template `+%`*[E](data: openArray[E]; off: int): auto =
#      toOpenArray(data, off, data.len - 1)
# ─────────────────────────────────────────────────────────────────

template `+%`*[E](data: openArray[E]; off: int): auto =
  toOpenArray(data, off, data.len - 1)

# ─────────────────────────────────────────────────────────────────
#  Kernel 1: double the full array
# ─────────────────────────────────────────────────────────────────

const kernel1 = cuda:
  proc kernelFull(output, input: ptr UncheckedArray[float32];
                  n: int32) {.global.} =
    doubleElements(output, toOpenArray(input, 0, n - 1))

# ─────────────────────────────────────────────────────────────────
#  Kernel 2: double the second half via +%
# ─────────────────────────────────────────────────────────────────

const kernel2 = cuda:
  proc kernelHalf(output, input: ptr UncheckedArray[float32];
                  n: int32) {.global.} =
    let half = n div 2
    doubleElements(output, toOpenArray(input, 0, n - 1) +% half)

# ─────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────

# Kernel 1
var fullIn = [1.0'f32, 2.0, 3.0, 4.0]
var fullBuf: array[4, float32]
var engine1 = bkCuda.init()
engine1.ingest(kernel1)
engine1.run<<(1, 1)>>("kernelFull", fullBuf, (fullIn, 4'i32))
for i in 0 .. 3:
  let expected = float32(i + 1) * 2.0'f32
  echo &"full[{i}]: {fullBuf[i]} (expected {expected})"
  doAssert abs(fullBuf[i] - expected) < 1e-5

# Kernel 2
var halfIn = [1.0'f32, 2.0, 3.0, 4.0]
var halfBuf: array[2, float32]
var engine2 = bkCuda.init()
engine2.ingest(kernel2)
engine2.run<<(1, 1)>>("kernelHalf", halfBuf, (halfIn, 4'i32))
for i in 0 .. 1:
  let expected = float32(i + 3) * 2.0'f32
  echo &"half[{i}]: {halfBuf[i]} (expected {expected})"
  doAssert abs(halfBuf[i] - expected) < 1e-5

echo "  OK — openArray type regression test"
