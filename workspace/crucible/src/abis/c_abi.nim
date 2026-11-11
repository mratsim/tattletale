# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/crucible/src/macros/ast_rebuilder

# ############################################################
#
#                     Binding utilities
#
# ############################################################

# Flag parameters
# ------------------------------------------------------------

type Flag*[E: enum] = distinct cint

func flag*[E: enum](e: varargs[E]): Flag[E] {.inline.} =
  ## Enum should only have power of 2 fields
  # static:
  #   for val in E:
  #     assert (ord(val) and (ord(val) - 1)) == 0, "Enum values should all be power of 2, found " &
  #                                                 $val & " with value " & $ord(val) & "."
  var flags = 0
  for val in e:
    flags = flags or ord(val)
  result = Flag[E](flags)

# Macros
# ------------------------------------------------------------

macro replacePragmasByInline(procAst: typed): untyped =
  ## Replace pragmas by the inline pragma
  ## We need a separate "typed" macro
  ## so that it is executed after the {.push mypragma.} calls
  var params: seq[NimNode]
  for i in 0 ..< procAst.params.len:
    params.add procAst.params[i]

  result = newStmtList()

  # push noconv applies multiple times, fight push with push
  result.add nnkPragma.newTree(ident"push", ident"nimcall", ident"inline")

  result.add newProc(
    name = procAst.name,
    params = params,
    body = procAst.body.rebuildUntypedAst(),
    procType = nnkProcDef,
    pragmas = nnkPragma.newTree(ident"inline", ident"nimcall")
  )

  result.add nnkPragma.newTree(ident"pop")

proc wrapOpenArrayImpl(ty, procAst: NimNode): NimNode =
  procAst.expectKind(nnkProcDef)
  let lenFirst = procAst.pragma.findChild(it.eqIdent"lenFirstArg") != nil
  var wrappeePragmas = procAst.pragma
  if lenFirst:
    wrappeePragmas = nnkPragma.newTree()
    for p in procAst.pragma:
      if not p.eqIdent"lenFirstArg":
        wrappeePragmas.add p

  var
    wrappeeParams = @[procAst.params[0]]
    wrapperParams = @[procAst.params[0]]
    wrapperBody = newCall(ident($procAst.name))

  for i in 1 ..< procAst.params.len:
    if procAst.params[i][^2].kind == nnkBracketExpr and procAst.params[i][^2][0].eqident"openarray":
      procAst.params[i].expectLen(3) # prevent `proc foo(a, b: openArray[int])`
      let
        ptrDef = newIdentDefs(
          ident($procAst.params[i][0] & "Ptr"),
          nnkPtrTy.newTree(procAst.params[i][^2][1]),
          newEmptyNode()
        )
        lenDef = newIdentDefs(
          ident($procAst.params[i][0] & "Len"),
          ty,
          newEmptyNode()
        )
        ptrArg = nnkIfExpr.newTree(
          nnkElifExpr.newTree(
            nnkInfix.newTree(
              ident"==",
              nnkDotExpr.newTree(ident($procAst.params[i][0]), bindSym"len"),
              newLit 0
            ),
            newNilLit()
          ),
          nnkElseExpr.newTree(
            newCall(
            ident"unsafeAddr",
            nnkBracketExpr.newTree(
              ident($procAst.params[i][0]),
              newLit 0
            ))
          )
        )
        lenArg = newCall(ty, nnkDotExpr.newTree(ident($procAst.params[i][0]), bindSym"len"))
      if lenFirst:
        wrappeeParams.add lenDef
        wrappeeParams.add ptrDef
        wrapperBody.add lenArg
        wrapperBody.add ptrArg
      else:
        wrappeeParams.add ptrDef
        wrappeeParams.add lenDef
        wrapperBody.add ptrArg
        wrapperBody.add lenArg
      wrapperParams.add procAst.params[i]
    else:
      wrappeeParams.add procAst.params[i]
      wrapperParams.add procAst.params[i]
      # Handle "a, b: int"
      for j in 0 ..< procAst.params[i].len - 2:
        wrapperBody.add ident($procAst.params[i][j])

  let wrappee = newProc(
    name = ident($procAst.name),                                 # Remove export marker if any
    params = wrappeeParams,
    body = procAst.body.copyNimTree(),
    procType = nnkProcDef,
    pragmas = wrappeePragmas
  )
  let wrapper = newProc(
    name = procAst[0],                                           # keep export marker if any
    params = wrapperParams,
    body = wrapperBody,
    procType = nnkProcDef,
    pragmas = nnkPragma.newTree(bindSym"replacePragmasByInline") # pragmas are for the wrappee
  )

  newStmtList(wrappee, wrapper)

macro wrapOpenArrayLenType*(ty: typedesc, procAst: untyped): untyped =
  ## Wraps pointer+len library calls in properly typed and converted openArray calls.
  ## Returns: `procAst` rewritten, each `openArray` parameter becomes a pointer + len pair.
  ##
  ## ```
  ## {.push noconv.}
  ## proc foo*(r: int, a: openArray[CustomType], b: int) {.wrapOpenArrayLenType: uint32, importc: "foo", dynlib: "libfoo.so".}
  ## {.pop.}
  ## ```
  ##
  ## is transformed into
  ##
  ## ```
  ## proc foo(r: int, a: ptr CustomType, aLen: uint32, b: int) {.noconv, importc: "foo", dynlib: "libfoo.so".}
  ##
  ## proc foo*(r: int, a: openArray[CustomType], b: int) {.inline.} =
  ##   foo(r, a[0].unsafeAddr, a.len.uint32, b)
  ## ```
  result = wrapOpenArrayImpl(ty, procAst)

macro wrapLenOpenArrayType*(ty: typedesc, procAst: untyped): untyped =
  ## Wraps len+pointer library calls in properly typed and converted openArray calls.
  ## Returns: `procAst` rewritten, the wrappee takes the length before the pointer
  ## (clSetKernelArg's `size, ptr` tail).
  ##
  ## ```
  ## {.push noconv.}
  ## proc bar*(r: int, size: uint32, a: ptr CustomType, b: int) {.wrapLenOpenArrayType: uint32, importc: "bar", dynlib: "libfoo.so".}
  ## {.pop.}
  ## ```
  ##
  ## is transformed into
  ##
  ## ```
  ## proc bar(r: int, aLen: uint32, a: ptr CustomType, b: int) {.noconv, importc: "bar", dynlib: "libfoo.so".}
  ##
  ## proc bar*(r: int, a: openArray[CustomType], b: int) {.inline.} =
  ##   bar(r, a.len.uint32, a[0].unsafeAddr, b)
  ## ```
  var ast = procAst.copyNimTree()
  if ast.pragma.kind == nnkEmpty:
    ast.pragma = nnkPragma.newTree(ident"lenFirstArg")
  else:
    ast.pragma.add ident"lenFirstArg"
  result = wrapOpenArrayImpl(ty, ast)

when isMainModule:
  expandMacros:
    {.push noconv.}

    proc foo(x: int, a: openArray[uint32], name: cstring) {.wrapOpenArrayLenType: cuint.} =
      discard

    {.pop.}
