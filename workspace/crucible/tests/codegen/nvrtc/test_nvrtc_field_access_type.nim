## Minimal repro: DotExpr crashes getGenericTypeName
##
## Crucible's getGenericTypeName (resolvers.nim:84) only handles
## nnkSym and nnkBracketExpr. A DotExpr in generic type position
## (like when a template expansion produces `tv.layout.stride`
## as a type argument) crashes.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_field_access_type.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

type
  Inner*[T] = object
    val: T

  Outer*[T] = object
    inner: Inner[T]

  Wrapper*[T] = object
    data: T

# Template that expands to a DotExpr in type position —
# simulating what the TensorView `()` operator does.
template getFieldType(obj: typed): typedesc =
  type(obj.val)

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    var x: Outer[uint32]
    type F = type(x.inner.val)
    let w = Wrapper[F]()
    C[0] = 1'u32

suite "Crucible - nested field access in type context":
  test "Generic with field-access type param inside cuda: compiles":
    let code = kernel
    echo code
    var nv = initNvrtc(code)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    check true
