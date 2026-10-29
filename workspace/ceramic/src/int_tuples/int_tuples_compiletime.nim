# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_datatypes
import workspace/ceramic/src/integer_semi_modules

# ═══════════════════════════════════════════════════════════════════════
#  AST syntax sugar
# ═══════════════════════════════════════════════════════════════════════

func tupleTypeWithoutNames(t: NimNode): NimNode {.compileTime.} =
  # Keep only the types from a named tuple like  `tuple[a: int, b: int]`
  if t.kind != nnkTupleTy:
    return t
  var fields: seq[NimNode] = @[]
  for f in t:
    if f.kind == nnkIdentDefs:
      # a grouped `tuple[a, b: int]` IdentDefs groups several
      # identifiers under one shared type, the type appends once per identifier
      for _ in 0 ..< f.len - 2:
        fields.add f[^2].tupleTypeWithoutNames()
    else:
      fields.add f
  result = nnkTupleTy.newTree(fields)

func getTypeOf*(n: NimNode): NimNode {.compileTime.} =
  case n.kind
  of nnkStmtListExpr:
    result = getTypeOf(n[^1])
  of nnkCall, nnkObjConstr:
    if n[0].kind == nnkBracketExpr:
      result = n[0]
    elif n[0].eqIdent("E"):
      # a CoordStride atom constructor, the type comes from the arguments
      var coords = nnkPar.newTree()
      for i in 1 ..< n.len:
        coords.add n[i]
      result = nnkBracketExpr.newTree(ident"CoordStride", coords)
    else:
      result = n.getTypeInst()
  of nnkSym:
    result = n.getTypeInst()
  of nnkIdent, nnkIntLit .. nnkUInt64Lit, nnkFloatLit:
    result = nil
  of nnkPar, nnkTupleConstr:
    if n.kind == nnkPar and n.len == 1:
      result = getTypeOf(n[0])
    else:
      result = nnkTupleTy.newTree()
      for child in n:
        result.add getTypeOf(child)
  of nnkBracketExpr:
    let base = getTypeOf(n[0])
    if base != nil and base.kind in {nnkTupleTy, nnkTupleConstr} and
        n[1].kind in {nnkIntLit .. nnkUInt64Lit}:
      result = base[int(n[1].intVal)]
    else:
      result = n.getTypeInst()
  else:
    result = n.getTypeInst()

func getTupleType*(n: NimNode): NimNode {.compileTime.} =
  if n.kind == nnkTupleTy:
    return n.tupleTypeWithoutNames()
  let t = n.getType()
  let inner =
    if t.kind == nnkBracketExpr and t[0].eqIdent("typeDesc"):
      t[1]
    else:
      t
  result = inner.getTypeImpl().tupleTypeWithoutNames()

func isTupleTy*(t: NimNode): bool {.compileTime.} =
  if t == nil:
    return false
  case t.kind
  of nnkTupleTy, nnkTupleConstr:
    true
  of nnkPar:
    t.len != 1 # A single elem in parenthesis is not a tuple
  of nnkCall, nnkObjConstr:
    if t[0].kind == nnkBracketExpr:
      # an Int[v]() / Layout[A, B](...) constructor is a scalar value
      false
    else:
      t.getTupleType().kind in {nnkTupleConstr, nnkTupleTy}
  of nnkIntLit, nnkInt64Lit, nnkFloatLit, nnkCharLit, nnkStrLit,
     nnkNilLit, nnkEmpty:
    # fresh macro nodes, no type to resolve and never tuples
    false
  of nnkIdentDefs:
    # a typed declaration node is a scalar, never a tuple
    false
  of nnkBracketExpr:
    # a bracket is a tuple type only in its `tuple[...]` form, index expressions are scalars
    t.len > 0 and t[0].kind in {nnkSym, nnkIdent} and t[0].eqIdent("tuple")
  else:
    t.getTupleType().kind in {nnkTupleConstr, nnkTupleTy}

proc newLetAsgn*(stmts: var NimNode, name: string, value: NimNode): NimNode {.compileTime.} =
  result = genSym(nskLet, name)
  stmts.add result.newLetStmt value
