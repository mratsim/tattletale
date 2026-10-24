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
    run = proc(ctx: var GpuContext): void =
      ctx.explodeKernelParams(deepOnly = true)
  )

func registerExplodeKernelParams*(reg: var PassRegistry, deps: seq[string] = @[]) =
  ## SPIR-V route, the push block renders flat members only:
  ## every object kernel param expands, pure-value objects included
  reg.register(
    name = "explodeKernelParams",
    kind = pkTransform,
    phase = phaseMain,
    description = "Expands every kernel param of object type into ptr/scalar leaf params and rebuilds the struct in the body, leaf writes refuse (push constants are read-only)",
    dependsOn = deps,
    run = proc(ctx: var GpuContext): void =
      ctx.explodeKernelParams(deepOnly = false)
  )

import std / [hashes, macros, sequtils, sets, strutils, tables]
# macros: for the `error` builtin (compile-time diagnostics)
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

func containsPtrDeep(t: GpuType): bool =
  ## True when a pointer is reachable through the type's fields
  case t.kind
  of gtPtr: true
  of gtObject, gtGenericInst:
    for f in t.fields:
      if f.typ.containsPtrDeep:
        return true
    false
  of gtArray: t.aTyp.containsPtrDeep
  else: false

type KernelParamLeaf = object
  ## One signature entry of an expanded struct kernel param
  name: string
  typ: GpuType
  isBuffer: bool

proc kernelParamLeaves(t: GpuType, name: string, deepOnly: static bool): seq[KernelParamLeaf] =
  ## Metal/WebGPU reject struct-with-pointer kernel args, the signature expands
  ## depth-first in field order, matching the serializer binding order.
  ## `deepOnly` expands only structs carrying pointers, the flat route
  ## (SPIR-V push block) expands every object into flat leaves and refuses
  ## leaves narrower than 32 bits at pass time
  result = newSeq[KernelParamLeaf]()
  proc walk(t: GpuType, name: string, leaves: var seq[KernelParamLeaf]) =
    case t.kind
    of gtPtr:
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: true)
    of gtObject, gtGenericInst:
      when deepOnly:
        if t.containsPtrDeep:
          for f in t.fields:
            if size(f.typ) > 0:
              walk(f.typ, name & "_" & f.name, leaves)
        else:
          leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
      else:
        for f in t.fields:
          if size(f.typ) > 0:
            walk(f.typ, name & "_" & f.name, leaves)
    of gtArray:
      doAssert not t.containsPtrDeep,
        "kernelParamLeaves: array field '" & name & "' carries pointers, not supported as a kernel param leaf"
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
    else:
      when deepOnly:
        leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
      else:
        if t.kind in {gtBool, gtUint32, gtInt32, gtFloat32, gtSize_t,
                      gtUint64, gtInt64, gtFloat64}:
          leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
        else:
          error "explodeKernelParams: kernel parameter member `" & name &
            "` is outside the SPIR-V push subset: " & $t.kind &
            " (push members are 32-bit scalars, pointers and arrays of them)"
  walk(t, name, result)

proc buildLeafConstr(t: GpuType, name: string, deepOnly: static bool, leafIdents: Table[string, GpuAst]): GpuAst =
  ## Builds the reconstruction construction over the leaf params.
  ##
  ## Flat route: field values reuse the leaf params' symbol nodes.
  ## Each node stores the leaf type, resolution keys on iSym.
  ## Deep route: fresh name-only idents, text printers key on names.
  result = GpuAst(kind: gpuObjConstr, ocType: t)
  for f in t.fields:
    if size(f.typ) == 0:
      result.ocFields.add GpuFieldInit(name: f.name, value: GpuAst(kind: gpuDiscard), typ: f.typ)
    elif f.typ.kind in {gtObject, gtGenericInst}:
      when deepOnly:
        if f.typ.containsPtrDeep:
          result.ocFields.add GpuFieldInit(name: f.name, value: buildLeafConstr(f.typ, name & "_" & f.name, deepOnly, leafIdents), typ: f.typ)
        else:
          result.ocFields.add GpuFieldInit(name: f.name, value: newGpuIdent(name & "_" & f.name), typ: f.typ)
      else:
        # the flat route expands every object field, recursing to the leaves the walk produced
        result.ocFields.add GpuFieldInit(name: f.name, value: buildLeafConstr(f.typ, name & "_" & f.name, deepOnly, leafIdents), typ: f.typ)
    else:
      when deepOnly:
        result.ocFields.add GpuFieldInit(name: f.name, value: newGpuIdent(name & "_" & f.name), typ: f.typ)
      else:
        let ident = leafIdents.getOrDefault(name & "_" & f.name, nil)
        doAssert not ident.isNil,
          "buildLeafConstr: leaf `" & (name & "_" & f.name) & "` has no param"
        result.ocFields.add GpuFieldInit(name: f.name, value: ident, typ: f.typ)

proc refuseLeafWrites(n: GpuAst, locals: Table[string, GpuType], kernel: string) =
  ## Refuses assignments that write an exploded param's storage.
  ## The reconstruction local is a copy and the push storage is read-only.
  ## A field write updates only the copy and silently drops the value.
  ## Pointer-leaf writes reach device memory

  proc refuseTarget(n: GpuAst) =
    case n.kind
    of gpuIdent:
      if n.symbol != nil and n.symbol.iSym in locals:
        error "explodeKernelParams: assignment to the whole parameter object `" &
          n.symbol.name & "` (kernel `" & kernel &
          "`): push constants are read-only"
    of gpuDot:
      if n.dParent.kind == gpuIdent and n.dParent.symbol != nil and
          n.dParent.symbol.iSym in locals:
        error "explodeKernelParams: assignment to the field `" &
          n.dParent.symbol.name & "." & n.dField.symbol.name &
          "` (kernel `" & kernel & "`): push constants are read-only"
      refuseTarget(n.dParent)
    of gpuIndex:
      # a store through a pointer leaf member reaches device memory,
      # a store through a value leaf member (scalar or array) drops
      if n.iArr.kind == gpuDot:
        var chain = n.iArr
        var fields: seq[string]
        while chain.kind == gpuDot:
          fields.insert(chain.dField.symbol.name)
          chain = chain.dParent
        if chain.kind == gpuIdent and chain.symbol != nil and
            chain.symbol.iSym in locals:
          var t = locals[chain.symbol.iSym]
          var resolved = true
          for f in fields:
            var found = false
            for fld in t.fields:
              if fld.name == f:
                t = fld.typ
                found = true
                break
            if not found:
              resolved = false
              break
          if resolved and t.kind notin {gtPtr, gtUncheckedArray, gtVoidPtr}:
            error "explodeKernelParams: assignment through the value member `" &
              chain.symbol.name & "." & fields.join(".") & "` (kernel `" & kernel &
              "`): push constants are read-only"
    else:
      discard
  case n.kind
  of gpuBlock:
    for s in n.statements:
      s.refuseLeafWrites(locals, kernel)
  of gpuAssign:
    n.aLeft.refuseTarget()
  of gpuIf:
    n.ifThen.refuseLeafWrites(locals, kernel)
    for el in n.ifElifs:
      el.body.refuseLeafWrites(locals, kernel)
    if n.ifElse.kind != gpuDiscard:
      n.ifElse.refuseLeafWrites(locals, kernel)
  of gpuFor:
    n.fBody.refuseLeafWrites(locals, kernel)
  of gpuWhile:
    n.wBody.refuseLeafWrites(locals, kernel)
  else:
    discard

proc explodeKernelParams(ctx: var GpuContext, deepOnly: static bool) =
  ## One shared core, monomorphized per route:
  ## - deepOnly: only structs carrying (nested) pointer fields expand,
  ##   the Metal/WebGPU/MSL behavior
  ## - flat route: every object param expands, the SPIR-V push block
  ##   renders flat members only
  ## - the reconstruction local reuses the param's symbol so body idents
  ##   resolve by iSym without name lookups
  for _, fn in ctx.allFnTab.mpairs:
    if fn.kind != gpuProc or attGlobal notin fn.pAttributes:
      continue
    var taken: seq[string]
    for p in fn.pParams:
      taken.add p.ident.ident()
    when not deepOnly:
      # leaf names must not collide with any name the body uses,
      # declarations included: a colliding leaf loads a stale local,
      # misreading the kernel
      var used: HashSet[string]
      for name in taken:
        used.incl name
      fn.pBody.collectIdentNames(used)
    var expansionLocals = initTable[string, GpuType]()
    var params: seq[GpuParam]
    for p in fn.pParams:
      if p.typ.kind notin {gtObject, gtGenericInst}:
        params.add p
        continue
      when deepOnly:
        if not p.typ.containsPtrDeep:
          params.add p
          continue
      let leaves = p.typ.kernelParamLeaves(p.ident.ident(), deepOnly)
      var leafIdents = initTable[string, GpuAst]()
      for leaf in leaves:
        when deepOnly:
          doAssert leaf.name notin taken,
            "explodeKernelParams: leaf '" & leaf.name & "' collides with an existing kernel parameter"
        else:
          if leaf.name in used:
            error "explodeKernelParams: leaf `" & leaf.name &
              "` collides with an existing name in kernel `" & fn.pName.ident() & "`"
        let sym = newSymbol(leaf.name, typ = leaf.typ, symKind = gsGlobalKernelParam)
        let space = if leaf.isBuffer: asDevice else: asRMEM
        when not deepOnly:
          leafIdents[leaf.name] = GpuAst(kind: gpuIdent, symbol: sym)
        params.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                            typ: leaf.typ, addressSpace: space)
      let constr = buildLeafConstr(p.typ, p.ident.ident(), deepOnly, leafIdents)
      when deepOnly:
        let local = GpuAst(kind: gpuVar, vName: newGpuIdent(p.ident.ident()),
                           vType: p.typ, vInit: constr, vMutable: false,
                           addressSpace: asRMEM)
      else:
        let local = GpuAst(kind: gpuVar, vName: p.ident,
                           vType: p.typ, vInit: constr, vMutable: false,
                           addressSpace: asRMEM)
        expansionLocals[p.ident.symbol.iSym] = p.typ
      doAssert fn.pBody.kind == gpuBlock, "explodeKernelParams: kernel body is not a block"
      fn.pBody.statements.insert(local, 0)
    when not deepOnly:
      if expansionLocals.len > 0:
        fn.pBody.refuseLeafWrites(expansionLocals, fn.pName.ident())
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
  ## A pointer local declared without an initializer is untagged at birth,
  ## its assignment decides the storage the declaration must carry
  for _, fn in ctx.allFnTab.mpairs:
    if fn.kind == gpuProc:
      var locals = initTable[string, GpuAst]()
      ctx.tagPointerLocalsIn(fn.pBody, locals)
