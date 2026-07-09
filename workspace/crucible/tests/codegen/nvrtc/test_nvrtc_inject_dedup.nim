## {.inject.} let-bindings from recursive template expansion:
## repeated declarations in the same scope after block flattening.
##
## Without the fix: unnest pass flattens blocks but doesn't deduplicate
## {.inject.} variable names across iterations — `acc` / `it_a` / `it_b`
## get declared twice in the same scope.
##
## With the fix: shared usedNames set across recursive unnest calls
## renames duplicates to `acc_4`, `it_a_5`, etc.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_inject_dedup.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

type
  Wrapper* = object
    val*: int32

template foldThrice(body: untyped): untyped =
  block:
    let it_a {.inject.} = Wrapper(val: 10)
    let it_b {.inject.} = Wrapper(val: 1)
    body
  block:
    let it_a {.inject.} = Wrapper(val: 20)
    let it_b {.inject.} = Wrapper(val: 2)
    body
  block:
    let it_a {.inject.} = Wrapper(val: 30)
    let it_b {.inject.} = Wrapper(val: 3)
    body

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[int32]) {.global.} =
    var acc: int32 = 0
    foldThrice:
      acc = acc + it_a.val * it_b.val
    C[0] = acc

suite "Inject variable dedup":
  test "no duplicate declarations after block flattening":
    var output: array[1, int32]
    var nv = initNvrtc(kernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    # 10*1 + 20*2 + 30*3 = 10 + 40 + 90 = 140
    check output[0] == 140
