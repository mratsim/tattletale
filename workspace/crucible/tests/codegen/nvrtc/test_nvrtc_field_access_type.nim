## Crucible type resolver edge cases
##
## When tracing the TensorView `()` operator, several gaps were found
## in how crucible handles type expressions involving field access.
##
## Cases:
##   1. generic call with DotExpr arg — works (regression guard)
##   2. typeof field access in typedef — crashes parseTypeFields
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_field_access_type.nim

import std/[unittest]
import workspace/crucible

type
  Inner*[T] = object
    val: T

  Outer*[T] = object
    inner: Inner[T]

  Wrapper*[T] = object
    data: T

# ── Case 1: generic call with DotExpr arg ──
# Simulates rank(tv.layout.stride) — generic template where arg is a field access.
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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Crucible - type resolver edge cases":
    test "Case 1 — generic call with DotExpr arg":
      let code = kernelRank
      var output: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(code)
      engine.run<<(1, 1)>>("kernel", output, ())
      check output[0] == 1

    test "Case 2 — typeof field access in type definition":
      let code = kernelTypeof
      var output: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(code)
      engine.run<<(1, 1)>>("kernel", output, ())
      check output[0] == 1

when isMainModule:
  runTest()
