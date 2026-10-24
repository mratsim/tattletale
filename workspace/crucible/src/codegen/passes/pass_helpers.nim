# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Helpers shared by the transform passes.

import std/[macros, sets]
# macros: for the `error` builtin (compile-time diagnostics)
import ../ir/ast_datatypes

func unwrapSingleStmtBlocks*(n: GpuAst): GpuAst =
  ## Returns `n` with nested single-statement block wrappers stripped.
  ##
  ## toCrucibleAst shapes an if-expression branch statement as nested
  ## single-statement blocks:
  ##
  ##   `if c: (block: (t = 1.0; t + 1))`
  ##
  ## becomes block(block(assign, expr)).
  result = n
  while result.kind == gpuBlock and result.statements.len == 1:
    result = result.statements[0]

func isExprOnlyBranch*(n: GpuAst): bool =
  ## True when an if-expression branch is a bare expression after
  ## unwrapping single-statement block wrappers.
  unwrapSingleStmtBlocks(n).kind != gpuBlock

func roundUpMultiple*(x, step: int): int =
  ## Smallest multiple of `step` that is `x` or larger
  ((x + step - 1) div step) * step

func gpuScalarByteSize*(t: GpuType): int =
  ## Byte width of a scalar GpuType in the SPIR-V push and storage model.
  ## Divergences from `size` are deliberate:
  ## - gtBool is 4 bytes: push members and WGSL storage hold bools as i32
  ## - gtSize_t is 4 bytes: Nim int/uint lower to 32-bit off-graph
  case t.kind
  of gtVoid: 0
  of gtUint8: 1
  of gtUint16, gtInt16, gtFloat16: 2
  of gtBool, gtUint32, gtInt32, gtFloat32, gtSize_t: 4
  of gtUint64, gtInt64, gtFloat64: 8
  else: error("not a scalar GpuType: " & $t.kind)

proc containsReturnStmt*(n: GpuAst): bool =
  ## True when a return statement sits anywhere in `n`
  if n.kind == gpuReturn:
    return true
  for ch in n:
    if ch.containsReturnStmt():
      return true

# ═══════════════════════════════════════════════════════════════════════
# Struct kernel-param expansion
# ═══════════════════════════════════════════════════════════════════════

func containsPtrDeep*(t: GpuType): bool =
  case t.kind
  of gtPtr: true
  of gtObject, gtGenericInst:
    for f in t.fields:
      if f.typ.containsPtrDeep:
        return true
    false
  of gtArray: t.aTyp.containsPtrDeep
  else: false

type KernelParamLeaf* = object
  name*: string
  typ*: GpuType
  isBuffer*: bool

proc kernelParamLeaves*(t: GpuType, name: string): seq[KernelParamLeaf] =
  ## Leaves of `t` in field order, structs with (nested) pointer fields
  ## expand depth-first, the Metal/WebGPU/MSL resource model
  result = newSeq[KernelParamLeaf]()
  proc walk(t: GpuType, name: string, leaves: var seq[KernelParamLeaf]) =
    case t.kind
    of gtPtr:
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: true)
    of gtObject, gtGenericInst:
      if t.containsPtrDeep:
        for f in t.fields:
          if size(f.typ) > 0:
            walk(f.typ, name & "_" & f.name, leaves)
      else:
        leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
    of gtArray:
      doAssert not t.containsPtrDeep,
        "kernelParamLeaves: array field '" & name & "' carries pointers, not supported as a kernel param leaf"
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
    else:
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
  walk(t, name, result)

proc collectIdentNames*(n: GpuAst, names: var HashSet[string]) =
  ## Every identifier name appearing in `n`, declaration positions included.
  ## Fresh names must not collide with any name the body already uses.
  ##
  ## Args:
  ## - `names` receives the collected names
  ##
  ## The AST iterator yields value positions only, the dedicated fields carry
  ## declaration names (`var` names, call targets, loop variables).
  case n.kind
  of gpuIdent:
    if n.symbol != nil:
      names.incl n.symbol.name
    return
  of gpuVar:
    names.incl n.vName.symbol.name
  of gpuCall:
    if n.cName.symbol != nil:
      names.incl n.cName.symbol.name
  of gpuFor:
    names.incl n.fVar.symbol.name
  else:
    discard
  for ch in n:
    ch.collectIdentNames(names)
