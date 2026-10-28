# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Lowering: kernel struct-argument explosion for the flat resource targets,
## Metal, WebGPU, and the MSL family.
##
## A kernel parameter of struct type with a (nested) pointer field expands into flat ptr/scalar leaf parameters.
## Pure-value structs cross whole: they bind and copy without a resource.
##
## Emission contract:
## - pointer leaves name the device buffers, value leaves cross as values
## - one buffer slot per leaf in field order, matching the launch packing
## - the kernel body opens with a local rebuilt from the leaves

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerExplodeKernelStructs*(reg: var PassRegistry) =
  ## Registers the kernel struct-argument explosion pass, the lowering stage.
  reg.register("explodeKernelStructs", pkTransform, phaseMain,
    "Expands kernel parameters of struct type with pointer fields into flat leaves",
    run = explodePass)

proc explodePass(m: CirModule): string =
  ## Registry adapter: `explodeKernelStructs` changes the module in place,
  ## the registry run signature returns the refusal string.
  explodeKernelStructs(m)
  return ""


proc buildLeafConstr(t: CirType, name: string): CirExpr =
  ## Rebuilds the exploded struct over its leaf parameters: field `f` reads the leaf `name_f`,
  ## a nested pointer-holding struct recurses one level deeper.
  ##
  ## For `AxpyView{p, n}` exploded under `y`:
  ##   oconstr AxpyView{p: sym y_p, n: sym y_n}
  var fields: seq[CirKv]
  for f in t.fields:
    if f.typ.kind == ctVoid:
      continue
    if f.typ.kind == ctStruct and f.typ.containsPtrDeep:
      fields.add kv(f.name, buildLeafConstr(f.typ, name & "_" & f.name))
    else:
      fields.add kv(f.name, sym(name & "_" & f.name, f.typ))
  result = oconstrE(t, fields)

proc explodeKernelStructs(m: CirModule) =
  ## Expands every kernel parameter of struct type with a pointer field into leaves, in place.
  ##
  ## Contract:
  ## - buffer leaves name the device address space, value leaves the thread space
  ## - the kernel body opens with the rebuilt struct local
  ## - a leaf name colliding with an existing kernel parameter refuses
  for fn in m.procs:
    if not fn.isKernel:
      continue
    var taken: seq[string]
    for p in fn.params:
      taken.add p.name
    var params: seq[CirParam]
    var preamble: seq[CirStmt]
    for p in fn.params:
      if p.typ.kind == ctStruct and p.typ.containsPtrDeep:
        let leaves = p.typ.kernelParamLeaves(p.name)
        for leaf in leaves:
          if leaf.name in taken:
            doAssert false,
              "explodeKernelStructs: leaf '" & leaf.name & "' collides with an existing kernel parameter"
          taken.add leaf.name
          let typ =
            if leaf.isBuffer:
              case leaf.typ.kind
              of ctPtr: tPtr(caDevice, leaf.typ.pElem)
              of ctAptr: tAptr(caDevice, leaf.typ.pElem)
              else: leaf.typ
            else:
              leaf.typ
          params.add CirParam(name: leaf.name, typ: typ)
        preamble.add varD(p.name, p.typ, buildLeafConstr(p.typ, p.name))
      else:
        params.add p
    if preamble.len > 0:
      fn.params = params
      fn.body = preamble & fn.body
