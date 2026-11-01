# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Declarative registry of the text printers' target data.
##
## A row is a const object of plain data: strings, bools, ints, enums and tables of those.
## No field stores a proc or a placeholder whose value depends on a body walk.
##
## | family | row        | contents                                      |
## | ------ | ---------- | --------------------------------------------- |
## | scalar | ScalarRow  | type names and support cells by (kind, width) |
## | lit    | LitRow     | literal token forms                           |
## | kind   | KindRow    | walk policy: loops, ternary, asm, routing set |
## | space  | SpaceRow   | address-space qualifiers per context          |
## | conv   | ConvRow    | conversion name tables                        |
## | coord  | CoordRow   | coordinate builtin forms                      |
## | sig    | SigRow     | signature static parts, keyed per flavor      |
## | vector | VectorRule | a target's real vector coverage               |
##
## `genTargets` (targets_macros) parses row blocks into these types and generates one lookup proc per family and target.
## Checks run at registration as one compile-time block: full (kind, width) coverage per key, refused cells with a reason class,
## cross-target name collisions, and the ledgered cross-target divergences.
## A missing or drifted row is a compile error.

import std/tables
import ../ir/cirv2_lang
import ./emit_common

type
  ReasonClass* = enum
    ## Static class of a refused registry cell.
    ## A printer's refusal message composes at the refusal site from this class.
    rcNone
    ## Marks a supported cell.
    rcParseError
    ## No parser form for the type at all.
    rcCapability
    ## Target language lacks the capability the type needs.
    rcOutsideSubset
    ## Target language has the type, this compiler's subset excludes it.

  ScalarCell* = object
    ## One (kind, width) cell of a scalar row.
    kind*: CirTypeKind
    width*: int
    ## Bit width, 0 for the widthless kinds (void, bool).
    spelling*: string
    ## Type name the target prints, "" for a refused cell.
    supported*: bool
    reasonClass*: ReasonClass
    ## Static class of the refusal, `rcNone` for a supported cell.

  ScalarRow* = object
    ## Scalar type names of one target by (kind, width) cell.
    ## Every cell of `ScalarCellAxis` appears exactly once per target.
    ## Widths name different C types on different hosts: 64-bit `long` on LP64 hosts, 32-bit on LLP64 hosts.
    ## Each target states its own names, no cell inherits a name from another target.
    cells*: seq[ScalarCell]

  LitRow* = object
    ## Literal token forms of one target.
    ## No exponent or hex-float fields: no printer handles either form.
    int*: IntLitTokens
    f16Prefix*, f16Suffix*: string
    f32Suffix*: string
    f64Prefix*, f64Suffix*: string
    nilToken*: string
    nonFinite*: FloatLitTokens

  AsmForm* = enum
    ## Inline assembly policy of one target.
    afWrap
    ## Text wrapped in `asm(...)`.
    afSpliceRaw
    ## Text spliced into the output verbatim.
    afRefuse
    ## Inline assembly refused.

  TernaryForm* = enum
    ## Conditional expression form of one target.
    tfQuestion
    ## `cond ? then : else`
    tfSelectCall
    ## `select(else, then, cond)`

  ConstDeclForm* = enum
    ## Local constant declaration keyword of one target.
    cdfConstexpr
    cdfConst
    cdfLet

  ConstructorForm* = enum
    ## Aggregate construction form of one target.
    cofBraced
    ## Braced init `{a, b}`.
    cofCall
    ## Constructor call `Type(a, b)`.
    cofMixed
    ## Vector calls plus flat braces, HLSL's split form.

  KindRow* = object
    ## Per-target policy where the shared expression and statement walks branch on target.
    ## No for-range, switch or sizeof field: no printer emits the first two, sizeof routes per target through `printTargetExpr`.
    loop*: LoopStyle
    ternary*: TernaryForm
    foldsIteElse*: bool
    ## A single-statement else body folds into an `else if` chain.
    emitOperandRendersExpr*: bool
    ## A `{.emit.}` operand renders as an expression when true, splices its symbol name when false.
    asmForm*: AsmForm
    constDecl*: ConstDeclForm
    constructor*: ConstructorForm
    targetExprKinds*: set[CirExprKind]
    ## Optional expression kinds the printer routes through its own `printTargetExpr` proc.
    ## Eight common kinds route unconditionally and are not members of this set.

  SpaceRow* = object
    ## Address-space qualifiers of one target, in the contexts its printer uses.
    ## Qualifiers are bare names, emission sites add separators, a context the target does not have stays "".
    ptrQualifier*: array[CirAddrSpace, string]
    ## Qualifier before a pointer type in declarations and casts.
    localPrefix*: array[CirAddrSpace, string]
    ## Qualifier before a local variable's declared type.
    kernelParamScalar*: string
    ## Qualifier on a kernel's by-reference scalar parameters.
    kernelParamPointer*: string
    ## Qualifier on a kernel's pointer parameters.
    smemParam*: string
    ## Qualifier on a parameter carrying relocated shared-memory storage.
    kernelParamPointerRestrict*: bool = false
    ## Restrict qualifier on a kernel's device-space pointer parameter
    ## (`__global T* restrict p`), a function-parameter qualifier in OpenCL C.
    ptrSpace*: array[CirAddrSpace, string]
    ## Address-space word inside the target's pointer type (`ptr<storage, T>`),
    ## empty when the space has no pointer form and the printer refuses at the site.
    ptrAccessMode*: string
    ## Access mode after the pointee type inside a storage pointer (`ptr<storage, T, read_write>`),
    ## empty when the target's pointer types take no access mode.
    ## An array pointer has no pointer form at all, its binding is the array and the pointer form refuses at the site.
    declVarPrefix*: string
    ## Space prefix of the hoisted module-scope variable declaration of relocated threadgroup
    ## storage (`var<workgroup>`), empty when the target does not relocate threadgroup storage
    ## into a space-qualified module-scope variable.
    bindingVarPrefix*: string
    ## Storage-class prefix of a module-level bound value variable (`var<storage, read>`), empty when the target binds values without one.
    bindingBufferPrefix*: string
    ## Storage-class prefix of a bound device buffer (`var<storage, read_write>`), empty when the target binds buffers as resources.

  ConvRow* = object
    ## Name tables of one target: conversion data that is a pure name-to-text mapping.
    ## Conversions needing operand type analysis or multi-statement emission stay procedural in the printers.
    builtinNames*: Table[string, string]
    ## `{.builtin.}` numeric conversion name to call text, `$1` stands for the call's argument list.
    fpRefusals*: seq[string]
    ## `{.builtin.}` fp-conversion names the target refuses.

  CoordWholeForm* = enum
    ## Whole-value read form of a coordinate builtin.
    cwfVector
    ## Whole-vector constructor form.
    cwfParam
    ## Bare parameter reference, a scalar builtin converts implicitly.
    cwfRefuse
    ## Whole-value reads refuse.
    cwfFlatIndex
    ## Whole-value read composes the x-major flat thread index from the target's
    ## vector builtins. The glue proc emits the text.
    cwfCast
    ## Whole-value read wraps a reference to the cell's name in the cell's cast
    ## (`int(gl_LocalInvocationIndex)`).

  CoordComponentForm* = enum
    ## Form of one coordinate component read.
    ccField
    ## Field access on the referenced name (`name.x`).
    ccDimCall
    ## Call with the component's dimension as a literal argument
    ## (`get_global_id(0)`).
    ccWhole
    ## Every read composes the cell's whole form: the builtin is scalar and no read addresses a component.
    ccWorkgroupDim
    ## A component read is the baked workgroup dimension at the component index, interpolated at emission from the entry's workgroup size.

  CoordCell* = object
    ## One coordinate builtin kind's static data of one target.
    kind*: CoordBuiltinKind
    paramName*: string
    ## Identifier a read references: an attributed parameter's name or a native builtin name, "" for the canonical builtin name.
    ## Names are kind-static, never body-dependent.
    paramType*: string
    ## Attributed parameter's type name, "" when the target has no form for the kind.
    paramAnnotation*: string
    ## Entry-point annotation of the builtin's synthesized kernel parameter:
    ## the WGSL `@builtin(...)` attribute name or the HLSL input semantic,
    ## "" when the kind has no synthesized parameter.
    whole*: CoordWholeForm
    componentCast*: string
    ## Type a component read casts to, "" for none.
    componentForm*: CoordComponentForm = ccField
    ## Form of the cell's component reads.

  CoordRow* = object
    ## Coordinate builtin data of one target.
    ## Whole-vector forms interpolate the parameter name at emission time in the glue procs, never in the row.
    cells*: seq[CoordCell]

  SigRow* = object
    ## Static parts of a target's function signatures.
    ## Parameter-list emission is procedural, parameters depend on the body walk, none of it is row data.
    ## Keyed per flavor where a target has flavors.
    entryLinkage*: string
    ## Prefix before the return type of a kernel entry.
    deviceFnLinkage*: string
    ## Prefix before the return type of a plain device function.
    inlineLinkage*: string
    ## Prefix before the return type of an inline device function.
    voidRet*: string
    ## Name of the void return type, "" emits no return type.
    workgroupAttr*: string
    ## Workgroup-size attribute form, one `$n` placeholder per dimension, "" when the target does not bake it into the shader.
    workgroupDefault*: array[3, int]
    ## Workgroup size baked in when the kernel declares none, read only when `workgroupAttr` is non-empty.
    entryAttr*: string
    ## Entry-point attribute prefix, "" when the target has none.
    bindingPrefix*: string
    ## Resource binding form, `$1` stands for the binding slot, "" when the target has no binding form.
    groupPrefix*: string
    ## Binding group form, `$1` stands for the group, "" when absent.
    entryTakesParams*: bool = true
    ## True when the entry signature takes the kernel's parameter list.
    ## False when the parameters pack elsewhere and the entry takes
    ## synthesized inputs instead: none for GLSL (push-constant block),
    ## `@builtin(...)` coordinate parameters for WGSL, semantics inputs for HLSL.

  VectorNameForm* = enum
    ## Composition of a vector type name from the element's prefix and the lane count.
    vnfPrefixed
    ## `prefix & $laneCount` forms the C-family names (`vec3`, `float4`). An empty prefix composes the element's scalar name from the scalar row.
    vnfGenericElem
    ## `prefix & $laneCount & '<' & elementName & '>'` forms the generic-element names (`vec3<f32>`).

  VectorContext* = enum
    ## Where a target's printer emits vector types.
    vcNone
    vcCoordinate
    ## Only in coordinate-builtin context.
    vcEverywhere

  VectorRule* = object
    ## A target's real vector coverage, as its printer emits it today.
    context*: VectorContext
    elems*: seq[tuple[kind: CirTypeKind, width: int, prefix: string]]
    ## Element types the target names, with the vector name's prefix.
    ## The name composes from the prefix and the lane count per `nameForm`.
    ## Empty for `vcNone`.
    lanes*: seq[int]
    ## Lane counts the target emits.
    nameForm*: VectorNameForm = vnfPrefixed
    ## Composition of the vector type name from the element's prefix and the lane count.

const ScalarCellAxis*: array[13, tuple[kind: CirTypeKind, width: int]] = [
  (ctVoid, 0), (ctBool, 0),
  (ctInt, 8), (ctInt, 16), (ctInt, 32), (ctInt, 64),
  (ctUint, 8), (ctUint, 16), (ctUint, 32), (ctUint, 64),
  (ctFloat, 16), (ctFloat, 32), (ctFloat, 64)]

proc checkScalarRow*(row: ScalarRow, target: string) {.compileTime.} =
  ## Registration checks over one target's scalar row:
  ##
  ## - every cell of `ScalarCellAxis` appears exactly once, a second row for an existing cell fails the compile
  ## - a supported cell has a type name and no reason class, a refused cell has a reason class and no name
  ## - a type name denotes at most one cell, a bool cell's name may repeat another
  ##   cell's name (OpenCL prints bool locals as `int` for the 4-byte launch ABI)
  ##
  ## Raises `AssertionDefect` at compile time on the first violated check.
  for axisCell in ScalarCellAxis:
    var count = 0
    for c in row.cells:
      if c.kind == axisCell.kind and c.width == axisCell.width:
        inc count
    doAssert count == 1, "target " & target & ": scalar cell (" &
      $axisCell.kind & ", " & $axisCell.width & ") appears " & $count &
      " times, exactly once required"
  for c in row.cells:
    if c.supported:
      doAssert c.reasonClass == rcNone and c.spelling.len > 0,
        "target " & target & ": supported cell (" & $c.kind & ", " & $c.width &
        ") needs a name and no reason class"
    else:
      doAssert c.reasonClass != rcNone and c.spelling.len == 0,
        "target " & target & ": refused cell (" & $c.kind & ", " & $c.width &
        ") needs a reason class and no name"
  for i, a in row.cells:
    if a.spelling.len == 0 or a.kind == ctBool:
      continue
    for j in i + 1 ..< row.cells.len:
      if row.cells[j].kind == ctBool:
        continue
      doAssert row.cells[j].spelling != a.spelling,
        "target " & target & ": name '" & a.spelling & "' denotes two cells, (" &
        $a.kind & ", " & $a.width & ") and (" & $row.cells[j].kind & ", " &
        $row.cells[j].width & ")"

proc checkScalarRowsTogether*(
    rows: openArray[tuple[name: string, row: ScalarRow]]) {.compileTime.} =
  ## Cross-target scalar check:
  ##
  ## - a name two targets share must denote the same (kind, width) cell
  ## - divergent names for one cell are explicit data, msl `char`/`long` against
  ##   cuda `signed char`/`long long`
  ## - ledgered divergences are checked by `checkScalarDivergences`
  ## - a bool cell's name may repeat a non-bool cell's name, see `checkScalarRow`
  ##
  ## Returns after catching a name that drifted onto a cell of another width or kind.
  ## Raises `AssertionDefect` naming both targets at compile time.
  for i in 0 ..< rows.len:
    for j in i + 1 ..< rows.len:
      for a in rows[i].row.cells:
        if a.spelling.len == 0:
          continue
        for b in rows[j].row.cells:
          if b.spelling == a.spelling:
            if a.kind == ctBool or b.kind == ctBool:
              continue
            doAssert a.kind == b.kind and a.width == b.width,
              "name '" & a.spelling & "' denotes (" & $a.kind & ", " &
              $a.width & ") on " & rows[i].name & " but (" & $b.kind & ", " &
              $b.width & ") on " & rows[j].name

proc checkScalarDivergences*(
    rows: openArray[tuple[name: string, row: ScalarRow]],
    divergent: openArray[tuple[a, b: string, kind: CirTypeKind, width: int]]) {.compileTime.} =
  ## Divergence ledger over target pairs: a ledgered cell keeps different names
  ## on the two targets.
  ##
  ## - the ledger records that the pair names a cell on purpose differently,
  ##   msl `char`/`long` against cuda `signed char`/`long long`
  ##   (the LP64/LLP64 `long` hazard)
  ## - the ledger stores no names, so the check cannot ossify row values
  ##
  ## Raises `AssertionDefect` at compile time when a ledgered cell has no name
  ## on one target or one shared name on both targets.
  ##
  ## Usage: `checkScalarDivergences([("msl", mslScalarRow()), ("cuda", cudaScalarRow())], ledger)`
  for d in divergent:
    var spellings: array[2, string]
    for entry in rows:
      let slot = if entry.name == d.a: 0 elif entry.name == d.b: 1 else: 2
      if slot < 2:
        for c in entry.row.cells:
          if c.kind == d.kind and c.width == d.width:
            spellings[slot] = c.spelling
    doAssert spellings[0].len > 0 and spellings[1].len > 0,
      "target pair (" & d.a & ", " & d.b & "): ledgered cell (" & $d.kind & ", " &
      $d.width & ") has no name on " & (if spellings[0].len == 0: d.a else: d.b)
    doAssert spellings[0] != spellings[1],
      "target pair (" & d.a & ", " & d.b & "): cell (" & $d.kind & ", " &
      $d.width & ") is ledgered as divergent but both targets name it '" &
      spellings[0] & "'"

proc checkCoordRow*(row: CoordRow, target: string) {.compileTime.} =
  ## Registration check over one target's coordinate row.
  ## Every coordinate builtin kind except `cbkNone` appears exactly once.
  ## Raises `AssertionDefect` at compile time on the first missing or duplicate kind.
  for k in CoordBuiltinKind:
    if k == cbkNone:
      continue
    var count = 0
    for c in row.cells:
      if c.kind == k:
        inc count
    doAssert count == 1, "target " & target & ": coordinate builtin " & $k &
      " appears " & $count & " times, exactly once required"

proc scalarCellRefusal*(row: ScalarRow, kind: CirTypeKind, width: int): string =
  ## Returns the refusal class of one (kind, width) cell as text, "" for a supported cell.
  ## A refusal message interpolating runtime values composes at the refusal site from this class.
  ##
  ## Usage: `let reason = scalarCellRefusal(row, ctFloat, 64)` tests fp64 support.
  for c in row.cells:
    if c.kind == kind and c.width == width:
      return if c.supported: "" else: $c.reasonClass
  return "<missing cell>"
