## All operators covered by assignOp: nnkInfix → gpuBinOp, not gpuCall
##
## assignOp maps Nim operator names to C++ operator symbols:
##   + - * /           (standard arithmetic, identity)
##   div → /  mod → %  (integer division/modulo)
##   shr → >>  shl → << (bit shifts)
##   and → &&|&  or → |||  xor → ^  (logical/bitwise)
##
## Without the fix: nnkInfix → gpuCall → maybePatchFnName renames to
## add/mul/div — non-existent function names.
## With the fix: all stay as gpuBinOp → `(a + b)` not `+(a,b)` or `add(a,b)`.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_user_defined_operator.nim

import std/[unittest, strutils]
import workspace/crucible/src/codegen/nvrtc

type
  Wrapper* = object
    val*: int32

proc `+`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val + b.val)
proc `-`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val - b.val)
proc `*`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val * b.val)
proc `div`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val div b.val)
proc `mod`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val mod b.val)
proc `shl`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val shl b.val)
proc `shr`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val shr b.val)
proc `and`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val and b.val)
proc `or`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val or b.val)
proc `xor`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val xor b.val)

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = Wrapper(val: 13)
    let b = Wrapper(val: 5)
    let x1 = a + b
    let x2 = a - b
    let x3 = a * b
    let x4 = a div b
    let x5 = a mod b
    let x6 = a shl b
    let x7 = a shr b
    let x8 = a and b
    let x9 = a or b
    let x10 = a xor b
    C[0] = float32(x1.val)

suite "User-defined operators via nnkInfix":
  test "all operators emit infix not call-style":
    # No renamed function calls
    check "add(" notin kernel
    check "sub(" notin kernel
    check "mul(" notin kernel
    check "div(" notin kernel
    check "mod(" notin kernel
    check "shr(" notin kernel
    check "shl(" notin kernel
    check "and(" notin kernel
    check "or(" notin kernel
    check "xor(" notin kernel
    # Verify infix rendering in CUDA output
    check "(a + b)" in kernel
    check "(a - b)" in kernel
    check "(a * b)" in kernel
    check "(a / b)" in kernel      # Nim div → C++ /
    check "(a % b)" in kernel      # Nim mod → C++ %
    check "(a << b)" in kernel     # Nim shl → C++ <<
    check "(a >> b)" in kernel     # Nim shr → C++ >>
    check "(a & b)" in kernel      # Nim and → C++ &
    check "(a | b)" in kernel      # Nim or → C++ |
    check "(a ^ b)" in kernel      # Nim xor → C++ ^
