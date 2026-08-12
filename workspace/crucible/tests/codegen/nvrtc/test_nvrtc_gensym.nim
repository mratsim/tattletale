## Test: crucible handles genSym'd temporaries from `{.genSym.}` pragma
##
## `{.genSym.}` appears when ceramic's `evalOnceAs` macro creates
## internal constants/variables with `{.genSym.}` pragma (see
## int_tuples_compiletime.nim:315). The pragma wraps the const/let
## identifier in a `nnkPragmaExpr`, which crucible must unwind.
##
## Related: https://github.com/nim-lang/Nim/blob/version-2-2/lib/system/macros.nim#L747

import std/[unittest, strformat]
import workspace/crucible

suite "Crucible - gensym temporaries":
  ## gensym'd constant with {.genSym.} pragma — core pattern from evalOnceAs
  ## Generates: const ct_tmp {.genSym.} = 42
  test "gensym const":
    const kernelCode = cuda:
      proc kernel_const(C: ptr UncheckedArray[uint32]) {.global.} =
        const ct_tmp {.genSym.} = 42
        C[0] = ct_tmp

    var buf: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelCode)
    engine.run<<(1, 1)>>("kernel_const", buf, ())
    check buf[0] == 42

  ## gensym'd let with {.genSym.} pragma — runtime branch of evalOnceAs
  test "gensym let":
    const kernelCode = cuda:
      proc kernel_let(C: ptr UncheckedArray[uint32]; x: uint32) {.global.} =
        let rt_tmp {.genSym.} = x
        C[0] = rt_tmp

    var buf: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelCode)
    engine.run<<(1, 1)>>("kernel_let", buf, (7'u32,))
    check buf[0] == 7

  ## genSym'd var in a for loop — pattern from fillWith / gemm kernels
  test "gensym var in for range":
    const kernelCode = cuda:
      proc kernel_var(C: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 5:
          var inner_tmp {.genSym.} = uint32(i * 3)
          C[i] = inner_tmp

    var buf: array[5, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelCode)
    engine.run<<(1, 5)>>("kernel_var", buf, ())
    for i in 0 ..< 5:
      check buf[i] == uint32(i * 3)

  ## Real evalOnceAs runtime pattern: let {.genSym.} + template forwarding
  test "gensym let + template (evalOnceAs runtime pattern)":
    const kernelCode = cuda:
      proc kernel_eoa(C: ptr UncheckedArray[uint32]; x: uint32) {.global.} =
        let rt_tmp {.genSym.} = x
        template v(): untyped = rt_tmp
        C[0] = v() + 1'u32

    var buf: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelCode)
    engine.run<<(1, 1)>>("kernel_eoa", buf, (41'u32,))
    check buf[0] == 42

  ## Multiple gensym consts working together in arithmetic
  test "multiple gensym consts":
    const kernelCode = cuda:
      proc kernel_multi(C: ptr UncheckedArray[uint32]) {.global.} =
        const m0 {.genSym.} = 3
        const m1 {.genSym.} = 5
        let idx = m0 * 16 + m1
        C[0] = uint32(idx)

    var buf: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelCode)
    engine.run<<(1, 1)>>("kernel_multi", buf, ())
    check buf[0] == 53
