## CuTe: when branches producing different types (B26)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_cute_when_types.nim
##
## CuTe dispatches tile types per GPU arch via compile-time branching.
## `when` is evaluated by Nim at compile time during generic instantiation.
## Inside `cuda:`, generic procs with `when` are instantiated by Crucible,
## which receives the AST with when already resolved by Nim.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  TileSmall = object
    val: uint32
  TileLarge = object
    val: uint32
    extra: array[4, uint32]

const kernelCode = cuda:
  proc whenTypesKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Use `when` directly in kernel body — it's resolved by Nim before Crucible.
    when 8 <= 16:
      let t = TileSmall(val: 42'u32)
      output[0] = t.val
    else:
      let t = TileLarge(val: 42'u32, extra: [0'u32, 0'u32, 0'u32, 0'u32])
      output[0] = t.val + 1'u32

    when 32 <= 16:
      let t2 = TileSmall(val: 42'u32)
      output[1] = t2.val
    else:
      let t2 = TileLarge(val: 42'u32, extra: [0'u32, 0'u32, 0'u32, 0'u32])
      output[1] = t2.val + 1'u32

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("whenTypesKernel", buf, ())
doAssert buf[0] == 42, &"small tile: {buf[0]}"
doAssert buf[1] == 43, &"large tile: {buf[1]}"
echo "  OK — when dispatch types (B26)"
