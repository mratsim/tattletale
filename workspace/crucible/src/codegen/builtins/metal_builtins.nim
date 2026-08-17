# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, sequtils, strutils]

import ./builtins_pragmas

const MetalAttributeNames* = [
  "thread_position_in_threadgroup",
  "threadgroup_position_in_grid",
  "threads_per_threadgroup",
  "threadgroups_per_grid",
  "thread_position_in_grid",
]
  ## MSL thread-position attribute names, the single source of truth for the Metal index builtins.
  ## `declareAttributeDummies` generates the typed dummies below from this list.
  ## The metal_lang printer reads this list for its attribute-param appending
  ## and its reserved-name checks, so the names never live in the compiler as a second copy.

macro declareAttributeDummies(): untyped =
  ## Declares one dummy object type and one `{.builtin, compileTime.}` let per attribute name in `MetalAttributeNames`.
  ## The dummies make the attribute identifiers typable inside the `metal:` macro,
  ## where `.x/.y/.z` resolve to `uint32` fields.
  ## The `{.builtin.}` pragma marks them as backend builtins for which no code is generated,
  ## following the cuda_builtins and wgsl_builtins pattern.
  result = newStmtList()
  for name in MetalAttributeNames:
    let typeName = ident("Metal" & name.split('_').mapIt(it.capitalizeAscii).join)
    let dummyName = ident(name)
    let fields = newNimNode(nnkRecList)
    for f in ["x", "y", "z"]:
      fields.add newTree(nnkIdentDefs,
        newTree(nnkPostfix, ident("*"), ident(f)),
        ident("uint32"),
        newEmptyNode())
    result.add newTree(nnkTypeSection,
      newTree(nnkTypeDef,
        newTree(nnkPostfix, ident("*"), typeName),
        newEmptyNode(),
        newTree(nnkObjectTy, newEmptyNode(), newEmptyNode(), fields)))
    result.add newTree(nnkLetSection,
      newTree(nnkIdentDefs,
        newTree(nnkPragmaExpr,
          newTree(nnkPostfix, ident("*"), dummyName),
          newTree(nnkPragma, ident("builtin"), ident("compileTime"))),
        newEmptyNode(),
        newTree(nnkCall, typeName)))

declareAttributeDummies()
