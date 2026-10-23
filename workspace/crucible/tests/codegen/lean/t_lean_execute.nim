# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Lean axpy execution, end-to-end
##
## Test that the emitted axpy Lean module executes over concrete buffers:
##
## - a=[2,3], b=[10,20], alpha=4 run through `lake env lean --run`,
##   the outputs 18 and 32 plus the preserved out-of-range index asserted in Nim
## - the axpy kernel (one loop over a constant range, one uint32 scalar)
##   elaborates under warningAsError, sorry-free, no scaffold of any kind
## - kernels outside the axpy shape (two scalars, wrong scalar type, a scalar on the guarded-thread shape, a spec pragma) refuse loudly
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[assertions, os, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/frontend/lean_pragmas
import workspace/crucible/src/codegen/backends/lean_lang
import workspace/crucible/src/codegen/frontend/nim_to_lean
import lean_testutils

const genAxpyRelPath = leanProjectDir & "/generated/axpy.lean"
const genAxpyExecRelPath = leanProjectDir & "/generated/axpy_exec.lean"

proc runTest() =
  block: # axpy emission + elaboration: the scalar becomes a Lean UInt32 argument
    let ir = irLean:
      proc axpyKernel(output, a, b: ptr UncheckedArray[uint32], alpha: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = alpha * a[i] + b[i]

    let emitted = emitLeanModule(ir.statements[0], genAxpyRelPath)
    doAssert "def axpyLoop " in emitted, "the loop stays a recursive def in Lean"
    doAssert "def axpyBody " in emitted, "entry-point def emitted"
    doAssert "(alpha : UInt32)" in emitted, "the scalar maps to a UInt32 argument"
    doAssert "alpha * a j" in emitted, "the multiply is in the stored value"

    let (outp, code) = elaborate("generated/axpy.lean")
    doAssert code == 0, "Lean rejected the emitted axpy module:\n" & outp
    doAssert "sorry" notin outp, "no sorry under warningAsError"
    echo "  axpy emission + elaboration (warningAsError): ok"

  block: # execution over concrete buffers: a=[2,3], b=[10,20], alpha=4
    # the harness appends a `main` to the emitted module: the emitted def
    # evaluates over the concrete buffers, Lean prints, Nim asserts.
    # The Nim side computes the expectation from the closed form
    # `alpha * a[j] + b[j]` over uint32, index 2 is out of the loop range
    # and must come back preserved at its initial 0
    var expected: array[3, uint32]
    let a = [2'u32, 3]
    let b = [10'u32, 20]
    for j in 0 ..< 2:
      expected[j] = 4'u32 * a[j] + b[j]

    let harness = readFile(genAxpyRelPath) & """

def main : IO Unit := do
  let a : Nat → UInt32 := fun i => if i = 0 then 2 else if i = 1 then 3 else 0
  let b : Nat → UInt32 := fun i => if i = 0 then 10 else if i = 1 then 20 else 0
  let outBuf : Nat → UInt32 := fun _ => 0
  let res := axpyBody a b outBuf 4
  IO.println ("res 0 = " ++ toString (res 0))
  IO.println ("res 1 = " ++ toString (res 1))
  IO.println ("res 2 = " ++ toString (res 2))
"""
    writeFile(genAxpyExecRelPath, harness)
    let (outp, code) = execute("generated/axpy_exec.lean")
    doAssert code == 0, "the Lean run of the emitted axpy failed:\n" & outp
    for j in 0 ..< 3:
      let want = "res " & $j & " = " & $expected[j]
      doAssert want in outp,
        "missing executed value '" & want & "' in the Lean run output:\n" & outp
    echo "  executed axpyBody over a=[2,3] b=[10,20] alpha=4: got [18, 32],",
      " index 2 preserved: ok"

  block: # loud refusals on kernels outside the axpy shape
    let twoScalars = irLean:
      proc twoScalars(output, a, b: ptr UncheckedArray[uint32], alpha, beta: uint32) {.global.} =
        for i in 0 ..< 2:
          output[i] = alpha * a[i] + b[i] + beta
    doAssertRaises(ValueError):
      discard emitLeanModule(twoScalars.statements[0], genAxpyRelPath)

    let wrongScalarType = irLean:
      proc wrongScalarType(output, a, b: ptr UncheckedArray[uint32], alpha: int32) {.global.} =
        for i in 0 ..< 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(wrongScalarType.statements[0], genAxpyRelPath)

    let guardedWithScalar = irLean:
      proc guardedWithScalar(output, a, b: ptr UncheckedArray[uint32], alpha: uint32) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(guardedWithScalar.statements[0], genAxpyRelPath)

    let withSpec = irLeanSpec:
      proc axpySpec(output, a, b: ptr UncheckedArray[uint32], alpha: uint32) {.global, invariant: "i < 2".} =
        for i in 0 ..< 2:
          output[i] = alpha * a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModuleWithSpecs(withSpec.gpu.statements[0], withSpec.specs,
        genAxpyRelPath)
    echo "  refusals on kernels outside the axpy shape: ok"

when isMainModule:
  runTest()
  echo "t_lean_execute: all checks passed"
