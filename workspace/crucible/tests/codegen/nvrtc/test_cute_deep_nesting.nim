## CuTe: deep generic nesting (B22)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_deep_nesting.nim
##
## Tests that moderately deep generic type nesting (10+ levels) resolves
## without stack overflow. Real CuTe layouts rarely exceed 3-4 levels.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  L0 = object
    val: uint32
  L1[N: static int] = object
    inner: L0
    extra: array[N, uint32]
  L2[N: static int] = object
    inner: L1[N]
    extra: array[N, uint32]
  L3[N: static int] = object
    inner: L2[N]
    extra: array[N, uint32]
  L4[N: static int] = object
    inner: L3[N]
    extra: array[N, uint32]
  L5[N: static int] = object
    inner: L4[N]
    extra: array[N, uint32]
  L6[N: static int] = object
    inner: L5[N]
    extra: array[N, uint32]
  L7[N: static int] = object
    inner: L6[N]
    extra: array[N, uint32]
  L8[N: static int] = object
    inner: L7[N]
    extra: array[N, uint32]
  L9[N: static int] = object
    inner: L8[N]
    extra: array[N, uint32]
  L10[N: static int] = object
    inner: L9[N]
    extra: array[N, uint32]

const kernelCode = cuda:
  proc deepNestedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Construct 10-layer nesting: L10 contains L9[N] contains ... L1[N] contains L0
    let l = L10[1](inner: L9[1](inner: L8[1](inner: L7[1](inner: L6[1](
      inner: L5[1](inner: L4[1](inner: L3[1](inner: L2[1](
        inner: L1[1](inner: L0(val: 42'u32), extra: [1'u32]),
      extra: [2'u32]),
    extra: [3'u32]),
  extra: [4'u32]),
extra: [5'u32]),
extra: [6'u32]),
extra: [7'u32]),
extra: [8'u32]),
extra: [9'u32]),
extra: [10'u32])

    # Chain of `.inner` through all 10 layers to reach L0.val
    output[0] = l.inner.inner.inner.inner.inner.inner.inner.inner.inner.inner.val

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("deepNestedKernel", buf, ())
doAssert buf[0] == 42, &"deep: {buf[0]}"
echo "  OK — deep nesting (B22)"
