import std/macros
import workspace/ceramic/src/layout_algebra/layout_constructors
import workspace/ceramic/src/layout_algebra

macro dummy(sh, st: typed): untyped =
  echo "  [dummy expanded NOW: sh=", sh.repr, " ty=", sh.getTypeInst().repr, "]"
  result = sh

macro showGetAst(layout: typed): untyped =
  var stmts = newStmtList()
  let (sh, st) = destructureLayout(stmts, layout)
  template delegate(sh2, st2) =
    dummy(sh2, st2)
  result = stmts
  result.add getAst(delegate(sh, st))
  echo "  [showGetAst emits]: ", repr(result)

block:
  echo "case1:"
  discard showGetAst(make_layout((4, 8), (1, 4)))
  echo "case3:"
  discard showGetAst(coalesce(make_layout((4, 8), (1, 4))))
