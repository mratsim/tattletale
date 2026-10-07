# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/strutils

type
  ATuple*[AT: static tuple] = object

proc unwrapTypedesc(n: NimNode): NimNode =
  var t = n.getTypeInst
  if t.kind == nnkBracketExpr and t[0].kind == nnkSym and t[0].strVal == "typeDesc":
    t = t[1]
  if t.kind == nnkBracketExpr and t[0].kind == nnkSym and t[0].strVal == "ATuple":
    return t[1]
  if t.kind == nnkSym:
    return t.getImpl[2][1]
  error("ATuple: not an ATuple instantiation: " & t.repr, n)

proc pretty(t: NimNode, path: seq[int]): string =
  # 1. classify - depth-first count of nonzero leaves
  var
    nzCount = 0
    nzValue: BiggestInt
    nzPath: seq[int]
  proc walk(n: NimNode, p: seq[int]) =
    if n.kind == nnkTupleConstr:
      for i in 0 ..< n.len:
        walk(n[i], p & i)
    else:
      if n.intVal != 0:
        nzCount += 1
        nzValue = n.intVal
        nzPath = p
  walk(t, path)

  # 2. zero element
  if nzCount == 0:
    return "0"

  # 3. single basis leaf - value@p_n@...@p_0, outermost first
  if nzCount == 1:
    var s = $nzValue
    for i in countdown(nzPath.len - 1, 0):
      s.add("@")
      s.add($nzPath[i])
    return s

  # 4. multi-term - nested tuple, recursive pretty printing
  var parts: seq[string]
  for i in 0 ..< t.len:
    if t[i].kind == nnkTupleConstr:
      parts.add(pretty(t[i], @[]))
    else:
      parts.add($t[i].intVal)
  return "(" & parts.join(",") & ")"

macro `$`*(T: typedesc[ATuple]): string =
  result = newStrLitNode(pretty(T.unwrapTypedesc(), @[]))

macro `+`*[A: ATuple, B: ATuple](a: typedesc[A], b: typedesc[B]): typedesc =
  let ta = a.unwrapTypedesc()
  let tb = b.unwrapTypedesc()
  if ta.len != tb.len:
    error("ATuple +: rank mismatch " & $ta.len & " vs " & $tb.len, b)
  var sum = nnkTupleConstr.newTree()
  for i in 0 ..< ta.len:
    sum.add(newIntLitNode(ta[i].intVal + tb[i].intVal))
  result = nnkBracketExpr.newTree(ident("ATuple"), sum)

macro `*`*(k: static int, A: typedesc[ATuple]): typedesc =
  let ta = A.unwrapTypedesc()
  var scaled = nnkTupleConstr.newTree()
  for i in 0 ..< ta.len:
    scaled.add(newIntLitNode(k * ta[i].intVal))
  result = nnkBracketExpr.newTree(ident("ATuple"), scaled)

template `*`*(A: typedesc[ATuple], k: static int): typedesc =
  k * A

when isMainModule:
  type
    E0 = ATuple[(1, 0)]
    E1 = ATuple[(0, 1)]
    Sum = ATuple[(3, 5)]
    Nested = ATuple[((1, 0), 5)]
    ThreeD = ATuple[(1, 0, 0)]
    Deep = ATuple[(((0, 7), 0), 0)]

  echo $E0
  echo $E1
  echo $Sum
  echo $Nested
  echo $ThreeD
  echo $Deep
  echo $ATuple[(((1, 0, 0), 0), 0)]
  echo $(E0 + E1)
  echo $(4 * Sum)
  echo $ATuple[(0, 0)]
