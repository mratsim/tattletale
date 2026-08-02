## Phase 5: annotateTypesForCodegen pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_annotateTypes.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. gpuTypeToDesc for primitive types
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let desc = gpuTypeToDesc(int32)
  doAssert desc.kind == tdkInt32, "Expected tdkInt32, got " & $desc.kind
  echo "  OK — gtInt32 maps to tdkInt32"

block:
  let f64 = GpuType(kind: gtFloat64)
  let desc = gpuTypeToDesc(f64)
  doAssert desc.kind == tdkFloat64, "Expected tdkFloat64, got " & $desc.kind
  echo "  OK — gtFloat64 maps to tdkFloat64"

block:
  let void = GpuType(kind: gtVoid)
  let desc = gpuTypeToDesc(void)
  doAssert desc.kind == tdkVoid, "Expected tdkVoid, got " & $desc.kind
  echo "  OK — gtVoid maps to tdkVoid"

# ═══════════════════════════════════════════════════════════════════════════
# 2. gpuTypeToDesc for pointer types
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let desc = gpuTypeToDesc(ptrTyp)
  doAssert desc.kind == tdkPtr, "Expected tdkPtr, got " & $desc.kind
  doAssert not desc.tdImplicit, "Ptr should not be implicit"
  doAssert desc.tdTo.kind == tdkInt32, "Pointed-to type should be tdkInt32"
  echo "  OK — gtPtr(gtInt32) maps to tdkPtr(tdkInt32)"

# ═══════════════════════════════════════════════════════════════════════════
# 3. gpuTypeToDesc for array types
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arr = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
  let desc = gpuTypeToDesc(arr)
  doAssert desc.kind == tdkArray, "Expected tdkArray, got " & $desc.kind
  doAssert desc.tdLen == 4, "Array length should be 4"
  doAssert desc.tdElem.kind == tdkInt32, "Element type should be tdkInt32"
  echo "  OK — gtArray[gtInt32, 4] maps to tdkArray(tdkInt32, 4)"

# ═══════════════════════════════════════════════════════════════════════════
# 4. gpuTypeToDesc for struct types
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let f32 = GpuType(kind: gtFloat32)
  let obj = GpuType(kind: gtObject, name: "Foo",
    oFields: @[GpuTypeField(name: "x", typ: int32), GpuTypeField(name: "y", typ: f32)])
  let desc = gpuTypeToDesc(obj)
  doAssert desc.kind == tdkStruct, "Expected tdkStruct, got " & $desc.kind
  doAssert desc.tdStructName == "Foo", "Struct name should be Foo"
  doAssert desc.tdFields.len == 2, "Should have 2 fields"
  doAssert desc.tdFields[0].name == "x", "First field should be x"
  doAssert desc.tdFields[0].typ.kind == tdkInt32, "x should be int32"
  doAssert desc.tdFields[1].name == "y", "Second field should be y"
  doAssert desc.tdFields[1].typ.kind == tdkFloat32, "y should be float32"
  echo "  OK — gtObject(Foo{x: int32, y: float32}) maps to tdkStruct"

# ═══════════════════════════════════════════════════════════════════════════
# 5. getTypeDesc for gpuIdent
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("x", iSym = "x_h5", typ = int32)
  var ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  let desc = ctx.getTypeDesc(ident)
  doAssert desc.kind == tdkInt32, "Expected tdkInt32 for ident, got " & $desc.kind
  echo "  OK — getTypeDesc for gpuIdent returns correct type"

# ═══════════════════════════════════════════════════════════════════════════
# 6. getTypeDesc for gpuAddr / gpuDeref
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let sym = newSymbol("p", iSym = "p_h6", typ = ptrTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let addrOf = GpuAst(kind: gpuAddr, aOf: ident)
  let deref = GpuAst(kind: gpuDeref, dOf: ident)
  var ctx = GpuContext()

  let addrDesc = ctx.getTypeDesc(addrOf)
  doAssert addrDesc.kind == tdkPtr, "addr should be ptr"
  doAssert addrDesc.tdTo.kind == tdkPtr, "addr of ptr should be ptr-to-ptr"

  let derefDesc = ctx.getTypeDesc(deref)
  doAssert derefDesc.kind == tdkInt32, "deref of ptr(int32) should be int32"
  echo "  OK — getTypeDesc for gpuAddr/gpuDeref returns correct types"

echo ""
echo "  All annotateTypes tests passed."
