## Test: crucible type resolution edge cases
##
## Exercises patterns where Nim's AST representation diverges from the
## canonical form. Uses crucible-only types — NO ceramic dependency.
##
##   - ObjConstr with Empty type child — generic type instance in
##     const context loses the BracketExpr type node.
##     (resolvers.nim:initGpuGenericInst, nnkObjConstr branch)

import std/[unittest, strformat]
import workspace/crucible/src/codegen/nvrtc

type
  MyBox*[V: static int] = object
    val: uint32

suite "Crucible - type resolution edge cases":
  test "ObjConstr empty child (generic type in const)":
    const kernelCode = cuda:
      proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
        const x {.genSym.} = MyBox[1]()
        C[0] = x.val + 1'u32

    var buf: array[1, uint32]
    var nv = initNvrtc(kernelCode)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 1
