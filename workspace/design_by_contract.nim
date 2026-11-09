# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Design-by-contract checks, each check runs where it is written
##
## ```
## proc double(x: int): int =
##   precondition: x > 0
##   result = 2 * x
##   postcondition: result == 2 * x
##
## proc addKernels(a, b: openArray[int], outp: var openArray[int], n: int) {.contracts.} =
##   {.precondition: n == a.len and n == b.len.}
##   for i in 0 ..< n:
##     {.invariant: (i >= 0 and i <= n, "the loop walks the inputs").}
##     outp[i] = a[i] + b[i]
##   {.postcondition: outp[0] == a[0] + b[0].}
## ```
##
## - the first form is live on import
## - the second needs {.contracts.} on the proc (statement pragmas are discarded at sem otherwise)
## - a (condition, "message") tuple gives the message, the condition
##   alone raises with the condition as written

import std/macros
import std/assertions
import std/strutils

macro contractCheck(kind: static string, expr: untyped) =
  ## Emits one check, a (condition, "message") tuple gives the message
  var e = expr
  if e.kind == nnkStmtList and e.len == 1:
    e = e[0]
  var cond, msg: NimNode
  if e.kind == nnkTupleConstr and e.len == 2:
    cond = e[0]
    msg = e[1]
  else:
    cond = e
    msg = newLit(e.repr.replace("\n", " ").strip())
  result = quote do:
    if not `cond`:
      raiseAssert(`kind` & " violated: " & `msg`)

template precondition*(expr: untyped) =
  contractCheck("precondition", expr)

template invariant*(expr: untyped) =
  contractCheck("invariant", expr)

template postcondition*(expr: untyped) =
  contractCheck("postcondition", expr)

proc isContractPragma(n: NimNode): bool =
  n.kind == nnkPragma and n.len == 1 and n[0].kind == nnkExprColonExpr and
    n[0][0].kind in {nnkIdent, nnkSym} and
    n[0][0].strVal in ["precondition", "invariant", "postcondition"]

proc rewriteBody(n: NimNode): NimNode =
  ## Rewrites every statement pragma into its check-emitting call
  case n.kind
  of nnkPragma:
    if isContractPragma(n):
      return newCall(newIdentNode(n[0][0].strVal), n[0][1])
    result = n
  of nnkObjConstr:
    result = n
  of nnkStmtList, nnkIfStmt, nnkElifBranch, nnkElse, nnkOfBranch,
      nnkWhileStmt, nnkForStmt, nnkTryStmt, nnkExceptBranch, nnkFinally,
      nnkBlockStmt, nnkWhenStmt:
    for i in 0 ..< n.len:
      n[i] = rewriteBody(n[i])
    result = n
  else:
    for i in 0 ..< n.len:
      if n[i].kind in {nnkStmtList, nnkIfStmt, nnkElifBranch, nnkElse,
          nnkOfBranch, nnkWhileStmt, nnkForStmt, nnkTryStmt,
          nnkExceptBranch, nnkFinally, nnkBlockStmt, nnkWhenStmt, nnkPragma}:
        n[i] = rewriteBody(n[i])
    result = n

macro contracts*(routine: untyped): untyped =
  ## Nim discards statement pragmas at sem, this rewrites them first
  result = routine
  result[6] = rewriteBody(result[6])
