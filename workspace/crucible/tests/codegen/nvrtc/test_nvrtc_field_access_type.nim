## Crucible type resolver edge cases
##
## When tracing the TensorView `()` operator, several gaps were found
## in how crucible handles type expressions involving field access.
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

# ── Case 1: generic call with DotExpr arg ──
# rank(tv.layout.stride) — generic template where arg is a field access.
# This currently works but let's keep the test for regression.
template rank(x: typed): int = 1

const kernelRank = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    var x: Outer[uint32]
    let r = rank(x.inner.val)
    C[0] = uint32(r)

# ── Case 2: type definition from typeof with field access ──
# type F = type(x.inner.val) — crashes "Unsupported type to parse fields from: nnkSym"
const kernelTypeof = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    var x: Outer[uint32]
    type F = type(x.inner.val)
    let w = Wrapper[F]()
    C[0] = 1'u32

suite "Crucible - type resolver edge cases":
  test "Case 1 — generic call with DotExpr arg":
    let code = kernelRank
    echo code
    var nv = initNvrtc(code)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    check true

  test "Case 2 — typeof field access in type definition":
    let code = kernelTypeof
    echo code
    var nv = initNvrtc(code)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    check true
