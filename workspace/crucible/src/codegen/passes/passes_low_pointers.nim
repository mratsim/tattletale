# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# Pointer deref insertion and struct kernel-param expansion
#
# ############################################################
#
# Pointers to local memory dereference through their base.
# Struct kernel parameters expand depth-first for backends that cannot take them whole.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerTagPointerLocals*(reg: var PassRegistry) =
  reg.register("tagPointerLocals", pkTransform, phaseMain,
    "Retypes pointer locals from the storage their assignment designates",
    dependsOn = @[],
    run = tagPointerLocals
  )

func registerExplodeStructWithPtrFields*(reg: var PassRegistry) =
  ## Metal/WebGPU/MSL expand, CUDA/OpenCL/GLSL cross structs natively
  ## (GLSL pointer fields cross as 64-bit buffer device addresses)
  reg.register(
    name = "explodeGlobalStructWithPtrFields",
    kind = pkTransform,
    phase = phaseMain,
    description = "Expands kernel params of structs carrying (nested) pointer fields into ptr/scalar leaves and rebuilds the struct in the body",
    dependsOn = @["ensureBlock", "pruneCompileTimeParams"],
    run = explodeGlobalStructWithPtrFields
  )

import std / [hashes, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../ir/ast_type_constructors
import ./pass_datatypes
import ./pass_registry
import ./pass_helpers

func registerInsertPointerDeref*(reg: var PassRegistry) =
  reg.register("insertPointerDeref", pkTransform, phaseMain,
    "Wraps pointer-typed index and field parents in explicit gpuDeref",
    dependsOn = @["rewriteIndexDeref"],
    run = insertPointerDeref
  )


# ═══════════════════════════════════════════════════════════════════════
# Pointer deref insertion
# ═══════════════════════════════════════════════════════════════════════

proc insertPointerDerefImpl(ctx: var GpuContext, n: var GpuAst) =
  ## Wraps pointer-typed expressions in explicit `gpuDeref` at value positions:
  ##
  ## - array indexing over a pointer to an array
  ## - field access over a pointer
  ## - deref over a pointer
  ##
  ## Address-of subtrees are untouched, the reference itself is the value there.
  ## Backends that cannot index pointers render the deref as a block-member access, C-family backends collapse it.
  case n.kind
  of gpuAddr:
    discard
  of gpuIndex:
    insertPointerDerefImpl(ctx, n.iArr)
    insertPointerDerefImpl(ctx, n.iIndex)
    if n.iArr.kind != gpuDeref:
      let t = ctx.getOperandType(n.iArr)
      if not t.isNil and t.kind == gtPtr and not t.to.isNil:
        n.iArr = GpuAst(kind: gpuDeref, dOf: n.iArr)
  of gpuDot:
    insertPointerDerefImpl(ctx, n.dParent)
    insertPointerDerefImpl(ctx, n.dField)
    if n.dParent.kind != gpuDeref:
      let t = ctx.getOperandType(n.dParent)
      if not t.isNil and t.kind == gtPtr:
        n.dParent = GpuAst(kind: gpuDeref, dOf: n.dParent)
  of gpuDeref:
    # Recurse into the deref'd subtree only: this node is already the wrap
    # for its operand. A re-wrap chains derefs and adds one extra `.x[0]`
    # per visit in the backends that render them
    insertPointerDerefImpl(ctx, n.dOf)
  else:
    for ch in mitems(n):
      insertPointerDerefImpl(ctx, ch)

proc insertPointerDeref(ctx: var GpuContext) =
  for _, fn in ctx.allFnTab.mpairs:
    insertPointerDerefImpl(ctx, fn.pBody)

# ═══════════════════════════════════════════════════════════════════════
# Reserved keyword renaming
# ═══════════════════════════════════════════════════════════════════════

proc buildLeafConstr(t: GpuType, name: string): GpuAst =
  ## Rebuilds the exploded struct over name-keyed leaf idents
  result = GpuAst(kind: gpuObjConstr, ocType: t)
  for f in t.fields:
    if size(f.typ) == 0:
      result.ocFields.add GpuFieldInit(name: f.name, value: GpuAst(kind: gpuDiscard), typ: f.typ)
    elif f.typ.kind in {gtObject, gtGenericInst} and f.typ.containsPtrDeep:
      result.ocFields.add GpuFieldInit(name: f.name, value: buildLeafConstr(f.typ, name & "_" & f.name), typ: f.typ)
    else:
      result.ocFields.add GpuFieldInit(name: f.name, value: newGpuIdent(name & "_" & f.name), typ: f.typ)

proc explodeGlobalStructWithPtrFields(ctx: var GpuContext) =
  ## Structs with (nested) pointer fields expand into ptr/scalar leaves,
  ## pure-value structs cross whole, leaves via kernelParamLeaves
  for _, fn in ctx.allFnTab.mpairs:
    if fn.kind != gpuProc or attGlobal notin fn.pAttributes:
      continue
    var taken: seq[string]
    for p in fn.pParams:
      taken.add p.ident.ident()
    var params: seq[GpuParam]
    for p in fn.pParams:
      if p.typ.kind in {gtObject, gtGenericInst} and p.typ.containsPtrDeep:
        let leaves = p.typ.kernelParamLeaves(p.ident.ident())
        for leaf in leaves:
          doAssert leaf.name notin taken,
            "explodeGlobalStructWithPtrFields: leaf '" & leaf.name & "' collides with an existing kernel parameter"
          let sym = newSymbol(leaf.name, typ = leaf.typ, symKind = gsGlobalKernelParam)
          let space = if leaf.isBuffer: asDevice else: asRMEM
          params.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                              typ: leaf.typ, addressSpace: space)
        let constr = buildLeafConstr(p.typ, p.ident.ident())
        let local = GpuAst(kind: gpuVar, vName: newGpuIdent(p.ident.ident()),
                           vType: p.typ, vInit: constr, vMutable: false,
                           addressSpace: asRMEM)
        doAssert fn.pBody.kind == gpuBlock, "explodeGlobalStructWithPtrFields: kernel body is not a block"
        fn.pBody.statements.insert(local, 0)
      else:
        params.add p
    fn.pParams = params

# ═══════════════════════════════════════════════════════════════════════
# Pointer local assignment
# ═══════════════════════════════════════════════════════════════════════

proc tagPointerLocalsIn(ctx: var GpuContext, n: GpuAst, locals: var Table[string, GpuAst]) =
  ## Retypes pointer locals from their assignments
  if n.isNil:
    return
  case n.kind
  of gpuVar:
    if n.vType.kind == gtPtr:
      locals[n.vName.symbol.name] = n
  of gpuAssign:
    if n.aLeft.kind == gpuIdent and not n.aLeft.symbol.isNil and
        not n.aLeft.symbol.typ.isNil and n.aLeft.symbol.typ.kind == gtPtr:
      let space = ctx.resolveAddressSpace(n.aRight)
      if space != n.aLeft.symbol.typ.addressSpace:
        let newTyp = n.aLeft.symbol.typ.withAddressSpace(space)
        n.aLeft.symbol.typ = newTyp
        let owned = locals.getOrDefault(n.aLeft.symbol.name, nil)
        if not owned.isNil:
          owned.vType = newTyp
  else:
    discard
  for ch in n.items:
    ctx.tagPointerLocalsIn(ch, locals)

proc tagPointerLocals(ctx: var GpuContext) =
  for _, fn in ctx.allFnTab.mpairs:
    if fn.kind == gpuProc:
      var locals = initTable[string, GpuAst]()
      ctx.tagPointerLocalsIn(fn.pBody, locals)
