## IR: address-space pragmas resolve to the unified enum at declaration.
##
## Covers `collectAddressSpace` through the DSL: `{.smem.}` → asSMEM,
## `{.rmem.}` → asRMEM, `{.const_mem.}` → asConstant, and the absence of a
## pragma → asDevice (the zero default). The `const_mem` arm is the
## regression guard: Nim's identifier `normalize` strips underscores
## (`"const_mem"` → `"constmem"`), so the match must not go through it.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_collectAddressSpace.nim

import std/[macros, sequtils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

proc findVarSpaces(ir: GpuAst): seq[(string, AddressSpace)] =
  ## Collects (var name, addressSpace) for every gpuVar in the IR.
  if ir == nil: return
  if ir.kind == gpuVar:
    result.add (ir.vName.ident(), ir.addressSpace)
  for ch in ir:
    result.add findVarSpaces(ch)

let ir = toGpuAst:
  var smemBuf {.smem.}: int32
  var rmemVal {.rmem.}: int32
  var constVal {.const_mem.}: int32
  var plainVal: int32

block:
  let spaces = findVarSpaces(ir)
  let byName = spaces.toTable()
  doAssert byName["smemBuf"] == asSMEM, "{.smem.} must resolve to asSMEM, got " & $byName["smemBuf"]
  doAssert byName["rmemVal"] == asRMEM, "{.rmem.} must resolve to asRMEM, got " & $byName["rmemVal"]
  doAssert byName["constVal"] == asConstant, "{.const_mem.} must resolve to asConstant, got " & $byName["constVal"]
  doAssert byName["plainVal"] == asDevice, "unannotated var must default to asDevice, got " & $byName["plainVal"]
  echo "  OK — smem/rmem/const_mem/unannotated resolve to asSMEM/asRMEM/asConstant/asDevice"
