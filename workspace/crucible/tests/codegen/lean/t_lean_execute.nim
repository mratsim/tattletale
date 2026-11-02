## Lean axpy execution, end-to-end
##
## Test that the emitted axpy Lean module executes a=[2,3] b=[10,20] alpha=4 into 18 and 32
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[os, strutils]
import lean_testutils

const genDir = leanProjectDir & "/generated"
const genAxpyModuleRelPath = "generated/" & autogenModule("axpyKernel")
const genAxpyExecRelPath = "generated/axpy_exec.lean"

proc runTest() =
  block: # axpy emission + elaboration: the scalar becomes a Lean UInt32 argument
    let ir = leanIr:
      proc axpyKernel(output, a, b: ptr UncheckedArray[uint32], alpha: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = alpha * a[i] + b[i]
    doAssert ir.refusal.len == 0, "unexpected refusal: " & ir.refusal
    let hasLoopDef = "def axpyLoop " in ir.text
    let hasEntryDef = "def axpyBody " in ir.text
    let hasScalarBinder = "(alpha : UInt32)" in ir.text
    let hasMultiply = "alpha * a j" in ir.text
    doAssert hasLoopDef, "the loop stays a recursive def in Lean"
    doAssert hasEntryDef, "entry-point def emitted"
    doAssert hasScalarBinder, "the scalar maps to a UInt32 argument"
    doAssert hasMultiply, "the multiply is in the stored value"

    createDir(genDir)
    writeFile(leanProjectDir / genAxpyModuleRelPath, ir.text)
    let (outp, code) = elaborate(genAxpyModuleRelPath)
    doAssert code == 0, "Lean rejected the emitted axpy module:\n" & outp
    echo "  axpy emission + elaboration (warningAsError): ok"

  block: # execution over concrete buffers: a=[2,3], b=[10,20], alpha=4
    # harness appends a `main` to the emitted module so the emitted def
    # evaluates over the concrete buffers, Lean prints, Nim asserts.
    # Nim computes the expectation from the closed form
    # `alpha * a[j] + b[j]` over uint32, index 2 is out of the loop range
    # and must come back preserved at its initial 0
    var expected: array[3, uint32]
    let a = [2'u32, 3]
    let b = [10'u32, 20]
    for j in 0 ..< 2:
      expected[j] = 4'u32 * a[j] + b[j]

    let harness = readFile(leanProjectDir / genAxpyModuleRelPath) & """

def main : IO Unit := do
  let a : Nat → UInt32 := fun i => if i = 0 then 2 else if i = 1 then 3 else 0
  let b : Nat → UInt32 := fun i => if i = 0 then 10 else if i = 1 then 20 else 0
  let outBuf : Nat → UInt32 := fun _ => 0
  let res := axpyBody a b outBuf 4
  IO.println ("res 0 = " ++ toString (res 0))
  IO.println ("res 1 = " ++ toString (res 1))
  IO.println ("res 2 = " ++ toString (res 2))
"""
    writeFile(leanProjectDir / genAxpyExecRelPath, harness)
    let (outp, code) = execute("generated/axpy_exec.lean")
    doAssert code == 0, "the Lean run of the emitted axpy failed:\n" & outp
    let gotLines = outp.splitLines()
    for j in 0 ..< 3:
      let want = "res " & $j & " = " & $expected[j]
      doAssert want in gotLines,
        "missing executed value '" & want & "' in the Lean run output:\n" & outp
    echo "  executed axpyBody over a=[2,3] b=[10,20] alpha=4: got [18, 32],",
      " index 2 preserved: ok"

  block: # loud refusals on kernels outside the axpy shape
    let twoScalars = leanIr:
      proc twoScalars(output, a, b: ptr UncheckedArray[uint32], alpha, beta: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = alpha * a[i] + b[i] + beta
    expectLeanRefusal("lean_lang: unsupported kernel - more than one uint32 scalar parameter",
      twoScalars)

    let wrongScalarType = leanIr:
      proc wrongScalarType(output, a, b: ptr UncheckedArray[uint32], alpha: int32) {.global.} =
        for i in 0 ..< 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported kernel - parameter 'alpha' is neither" &
      " a ptr UncheckedArray[uint32] buffer nor a uint32 scalar", wrongScalarType)

    # a scalar named `j` would be shadowed by the emitted update binder,
    # collision refusal happens before emitting
    let scalarNamedJ = leanIr:
      proc jKernel(output, a, b: ptr UncheckedArray[uint32], j: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = j * a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported kernel - scalar parameter 'j' collides" &
      " with the emitted update binder 'j' of the loop shape", scalarNamedJ)
    echo "  refusals on kernels outside the axpy shape: ok"

  block: # refusal on a loop start bound to the scalar parameter
    let scalarStart = leanIr:
      proc scalarStartKernel(output, a, b: ptr UncheckedArray[uint32], alpha: uint32) {.global.} =
        for i in alpha ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - start must be the literal 0",
      scalarStart)
    echo "  refusal on a loop bound that starts at the scalar parameter: ok"

when isMainModule:
  runTest()
  echo "t_lean_execute: all checks passed"
