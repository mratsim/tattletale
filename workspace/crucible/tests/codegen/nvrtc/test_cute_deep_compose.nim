## CuTe: deep generic composition (B11, B12, B13)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_deep_compose.nim
##
## B11: 3-level generic nesting
## B12: Type alias chain (5+ aliases)
## B13: 8+ generic params (GEMM config)
import std/strformat
import workspace/crucible

# B12: type alias chain
type
  BaseTile[N: static int] = object
    data: array[N, uint32]
  Alias1[N: static int] = BaseTile[N]
  Alias2[N: static int] = Alias1[N]
  Alias3[N: static int] = Alias2[N]
  Alias4[N: static int] = Alias3[N]
  Alias5[N: static int] = Alias4[N]

# B11: 3-level nesting
type
  Leaf[N: static int] = object
    val: uint32
  Branch[N: static int] = object
    leaf: Leaf[N]
  Tree[N: static int] = object
    branch: Branch[N]

# B13: 8+ generic params (GEMM tile config)
type
  GemmTile[M, N, K, Tm, Tn, Tk, Tmr, Tnr, Tkr: static int] = object
    data: array[M * N, uint32]

const kernelCode = cuda:
  proc deepComposeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # B12: alias chain — BaseTile and Alias5 resolve to same type
    let t = Alias5[16](data: [0'u32, 1'u32, 2'u32, 3'u32, 4'u32, 5'u32,
                              6'u32, 7'u32, 8'u32, 9'u32, 10'u32, 11'u32,
                              12'u32, 13'u32, 14'u32, 15'u32])
    output[0] = t.data[0]
    output[1] = t.data[15]

    # B11: 3-level nesting
    let tree = Tree[8](branch: Branch[8](leaf: Leaf[8](val: 42'u32)))
    output[2] = tree.branch.leaf.val

    # B13: 8+ generic params
    let gemm = GemmTile[4, 4, 4, 2, 2, 2, 1, 1, 1](data: [99'u32, 99'u32, 99'u32, 99'u32,
                                              99'u32, 99'u32, 99'u32, 99'u32,
                                              99'u32, 99'u32, 99'u32, 99'u32,
                                              99'u32, 99'u32, 99'u32, 99'u32])
    output[3] = gemm.data[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[4, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("deepComposeKernel", buf, ())
  doAssert buf[0] == 0,   &"alias[0]: {buf[0]}"
  doAssert buf[1] == 15,  &"alias[15]: {buf[1]}"
  doAssert buf[2] == 42,  &"nested val: {buf[2]}"
  doAssert buf[3] == 99,  &"gemm[0]: {buf[3]}"
  echo "  OK — deep composition (B11, B12, B13)"

when isMainModule:
  runTest()
