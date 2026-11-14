## Emitted Lean module execution and scalar-name refusals
##
## Test that the emitted axpy and uint32-literal-conversion modules execute over concrete buffers and colliding scalar parameter names refuse
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[os, strutils]
import lean_testutils

const genDir = leanProjectDir & "/generated"
const genAxpyModuleRelPath = "generated/" & autogenModule("axpyKernel")
const genAxpyExecRelPath = "generated/axpy_exec.lean"
const genConvAddModuleRelPath = "generated/" & autogenModule("convAddKernel")
const genConvAddExecRelPath = "generated/convAdd_exec.lean"

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
    let (outp, code) = execute(genAxpyExecRelPath)
    doAssert code == 0, "the Lean run of the emitted axpy failed:\n" & outp
    let gotLines = outp.splitLines()
    for j in 0 ..< 3:
      let want = "res " & $j & " = " & $expected[j]
      doAssert want in gotLines,
        "missing executed value '" & want & "' in the Lean run output:\n" & outp
    echo "  executed axpyBody over a=[2,3] b=[10,20] alpha=4: got [18, 32],",
      " index 2 preserved: ok"

  block: # stored value carrying an exact uint32 conversion of an int literal
    # `uint32(1)` reaches the IR as a conversion node over the int literal
    let ir = leanIr:
      proc convAddKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i] + uint32(1)
    doAssert ir.refusal.len == 0, "unexpected refusal: " & ir.refusal

    createDir(genDir)
    writeFile(leanProjectDir / genConvAddModuleRelPath, ir.text)
    let (outp, code) = elaborate(genConvAddModuleRelPath)
    doAssert code == 0, "Lean rejected the emitted literal-conversion module:\n" & outp

    # harness appends a `main` to the emitted module so the emitted def
    # evaluates over the concrete buffers, Lean prints, Nim asserts.
    # Nim computes the expectation from the closed form `a[j] + b[j] + 1`
    # over uint32, the indices no thread wrote stay at their initial 0
    var expected: array[2, uint32]
    let a = [2'u32, 3]
    let b = [10'u32, 20]
    for j in 0 ..< 2:
      expected[j] = a[j] + b[j] + 1'u32

    let harness = readFile(leanProjectDir / genConvAddModuleRelPath) & """

def main : IO Unit := do
  let a : Nat → UInt32 := fun i => if i = 0 then 2 else if i = 1 then 3 else 0
  let b : Nat → UInt32 := fun i => if i = 0 then 10 else if i = 1 then 20 else 0
  let outBuf : Nat → UInt32 := fun _ => 0
  let r0 := convAddBody 0 2 a b outBuf
  let r1 := convAddBody 1 2 a b outBuf
  IO.println ("r0 0 = " ++ toString (r0 0))
  IO.println ("r0 2 = " ++ toString (r0 2))
  IO.println ("r1 1 = " ++ toString (r1 1))
  IO.println ("r1 0 = " ++ toString (r1 0))
"""
    writeFile(leanProjectDir / genConvAddExecRelPath, harness)
    let (runOut, runCode) = execute(genConvAddExecRelPath)
    doAssert runCode == 0,
      "the Lean run of the emitted literal-conversion module failed:\n" & runOut
    let gotLines = runOut.splitLines()
    for j in 0 ..< 2:
      let want = "r" & $j & " " & $j & " = " & $expected[j]
      doAssert want in gotLines,
        "missing executed value '" & want & "' in the Lean run output:\n" & runOut
    let r0KeepsOld = "r0 2 = 0" in gotLines
    let r1KeepsOld = "r1 0 = 0" in gotLines
    doAssert r0KeepsOld and r1KeepsOld,
      "the untouched indices must keep their old value, expected 'r0 2 = 0'" &
      " and 'r1 0 = 0' in the Lean run output:\n" & runOut
    echo "  executed convAddBody over a=[2,3] b=[10,20]: got [13, 24],",
      " untouched indices preserved: ok"

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
      " with the emitted binder 'j' (the update binder", scalarNamedJ)

    # a scalar named `k` duplicates the loop def's remaining-count binder,
    # invalid Lean that fails only at elaboration
    let scalarNamedK = leanIr:
      proc kKernel(output, a, b: ptr UncheckedArray[uint32], k: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = k * a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported kernel - scalar parameter 'k' collides" &
      " with the emitted binder 'k' (the remaining-count binder", scalarNamedK)

    # a scalar named `h` duplicates the theorems' guard hypothesis,
    # invalid Lean that fails only at elaboration
    let scalarNamedH = leanIr:
      proc hKernel(output, a, b: ptr UncheckedArray[uint32], h: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = h * a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported kernel - scalar parameter 'h' collides" &
      " with the emitted binder 'h' (the guard hypothesis", scalarNamedH)
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
