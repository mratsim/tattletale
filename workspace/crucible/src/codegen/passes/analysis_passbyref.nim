# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./pass_datatypes

# ############################################################
#
#          Pass-by-reference parameter analysis
#
# ############################################################

func hasArrayLeaf*(t: CirType): bool =
  case t.kind
  of ctStruct:
    for f in t.fields:
      if f.typ.hasArrayLeaf:
        return true
    false
  of ctArray: true
  else: false

func isLargeStruct*(t: CirType, minStructSize: int): bool =
  minStructSize >= 0 and t != nil and t.kind == ctStruct and
    computeCirSizeBytes(t) > minStructSize

proc tagPassByRef*(m: CirModule, minStructSize: int = -1, tagAllArrays: bool = false) =
  for fn in m.procs:
    if fn.isGlobalFn:
      continue
    for _, p in fn.params.mpairs:
      p.passByRef = p.typ.isLargeStruct(minStructSize) or (tagAllArrays and p.typ.hasArrayLeaf)

template registerTagPassByRef*(reg: var PassRegistry, passName: string, minStructSize: int, tagAllArrays: bool) =
  reg.register(passName, pkTransform,
    "Tags device-fn formals passing as hidden const pointers",
    run = proc(m: CirModule) {.nimcall.} =
      m.tagPassByRef(minStructSize, tagAllArrays)
  )
