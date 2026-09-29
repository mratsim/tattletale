## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Copy-atom declaration machinery.
## The `declareCopyAtoms:` macro parses h_copy_registry.nim's declarative block
## into the CopyAtom enum and per-atom consts.

type
  CopyKind* = enum
    ## Copy-atom slot kinds
    ## - `ckCopy` moves a chunk between two memory spaces
    ## - `ckCommit` closes the group of copies prepared since the last commit
    ## - `ckWait` blocks until all but a group depth of recent commit groups have completed
    ckCopy, ckCommit, ckWait

  MemSpace* = enum
    ## Memory space a copy-atom operand lives in. spAny leaves the space
    ## unconstrained (the blocking universal copy, commit/wait slots).
    spAny, spGmem, spSmem, spReg

  CacheHint* = enum
    ## Copy-atom cache behavior
    ## - `cache_default` default caching
    ## - `cache_always` cached in L1 and L2
    ## - `cache_global_bypass_L1` global cached with L1 bypass (16-byte chunks only)
    cache_default, cache_always, cache_global_bypass_L1

import std/[macros, strutils]
import workspace/ceramic/src/int_tuples

{.experimental: "dynamicBindSym".}
# bindSym with a computed name (`$atom & "_" & key`) from a static macro
# parameter needs this experimental dimension (same as h_mma_dispatch.nim).

# ═════════════════════════════════════════════════════════════════════════
#  Data types used by the parser




# ═════════════════════════════════════════════════════════════════════════
#  declareCopyAtoms — parser + generator
# ═════════════════════════════════════════════════════════════════════════

const CopyAtomPropKeys* = ["kind", "srcSpace", "dstSpace", "vecBytes",
                           "minAlign", "zeroFill", "cache", "transpose",
                           "minCudaArch", "instr", "waitDepth",
                           "commitAtom", "waitAtom"]
  ## The property keys a copy atom may declare, in declaration order.
  ## The generated const name is `NAME_key`.

const RequiredCopyAtomKeys = ["kind", "instr"]

type
  CopyAtomParams = object
    name: NimNode
    props: seq[(string, NimNode)]   ## (property key, value AST) as written

proc parseCopyAtomDecls(defs: var seq[CopyAtomParams]; body: NimNode) =
  ## Collects the copy-atom declarations into `defs`, validating the keys.
  ##
  ## Expected AST per atom
  ##   Command(Ident"atom", Ident"NAME", StmtList(Call(Ident"key", StmtList(value)) …))
  ##
  ## Value leaves (one per property)
  ## - a string literal
  ## - an int literal
  ## - an ident (enum member)
  body.expectKind(nnkStmtList)
  for atomDesc in body:
    atomDesc.expectKind(nnkCommand)
    doAssert atomDesc[0].eqIdent"atom", "expected `atom NAME:` declaration"
    let name = atomDesc[1]
    name.expectKind(nnkIdent)
    for existing in defs:
      doAssert $existing.name != $name,
        "declareCopyAtoms: duplicate atom name `" & $name & "`"
    let propsNode = atomDesc[2]
    propsNode.expectKind(nnkStmtList)
    var params = CopyAtomParams(name: name)
    var seen: seq[string]
    for prop in propsNode:
      prop.expectKind(nnkCall)
      let key = prop[0]
      key.expectKind(nnkIdent)
      let valNode = prop[1]
      valNode.expectKind(nnkStmtList)
      let keyStr = $key
      let v = valNode[0]
      doAssert keyStr in CopyAtomPropKeys,
        "declareCopyAtoms: unknown property `" & keyStr & "` on atom `" & $name & "`"
      doAssert keyStr notin seen,
        "declareCopyAtoms: duplicate property `" & keyStr & "` on atom `" & $name & "`"
      seen.add keyStr
      case keyStr
      of "kind":
        v.expectKind(nnkIdent)
        doAssert v.strVal in ["ckCopy", "ckCommit", "ckWait"],
          "declareCopyAtoms: invalid kind `" & v.strVal & "` on atom `" & $name & "`"
      of "srcSpace", "dstSpace":
        v.expectKind(nnkIdent)
        doAssert v.strVal in ["spAny", "spGmem", "spSmem", "spReg"],
          "declareCopyAtoms: invalid memory space `" & v.strVal & "` on atom `" & $name & "`"
      of "vecBytes", "minAlign", "minCudaArch", "waitDepth":
        v.expectKind(nnkIntLit)
        doAssert v.intVal >= 0,
          "declareCopyAtoms: property `" & keyStr & "` on atom `" & $name &
          "` must be non-negative, got " & $v.intVal
        if keyStr == "vecBytes":
          doAssert v.intVal in [4, 8, 16],
            "declareCopyAtoms: vecBytes on atom `" & $name & "` must be 4, 8 or 16, got " &
            $v.intVal
        if keyStr == "waitDepth":
          doAssert v.intVal <= 2,
            "declareCopyAtoms: waitDepth on atom `" & $name & "` must be 0, 1 or 2, got " &
            $v.intVal
      of "zeroFill", "transpose":
        v.expectKind(nnkIdent)
        doAssert v.strVal in ["true", "false"],
          "declareCopyAtoms: property `" & keyStr & "` on atom `" & $name &
          "` must be true or false"
      of "cache":
        v.expectKind(nnkIdent)
        doAssert v.strVal in ["cache_default", "cache_always", "cache_global_bypass_L1"],
          "declareCopyAtoms: invalid cache `" & v.strVal & "` on atom `" & $name & "`"
        # cache_global bypasses L1 and is only legal at 16-byte chunk width
        if v.strVal == "cache_global_bypass_L1":
          for (k2, v2) in params.props:
            if k2 == "vecBytes":
              doAssert v2.intVal == 16,
                "declareCopyAtoms: cache_global_bypass_L1 requires vecBytes 16 on atom `" &
                $name & "`"
      of "commitAtom", "waitAtom":
        v.expectKind(nnkStrLit)
      of "instr":
        v.expectKind(nnkStrLit)
        doAssert v.strVal == "" or v.strVal.startsWith("cp.async."),
          "declareCopyAtoms: invalid instruction `" & v.strVal & "` on atom `" & $name &
          "` (expected \"\" or a cp.async.* spelling)"
      else:
        doAssert false, "declareCopyAtoms: unhandled property `" & keyStr & "`"
      params.props.add (keyStr, v)
    for key in RequiredCopyAtomKeys:
      doAssert key in seen,
        "declareCopyAtoms: atom `" & $name & "` is missing property `" & key & "`"
    defs.add params

proc genCopyAtomDecls(defs: seq[CopyAtomParams]): NimNode =
  ## Generates the declarations
  ##   type CopyAtom* = enum NAME1, NAME2, …
  ##   const NAME1_kind* = ckCopy     (one exported const per atom per declared property)
  ##
  ## Defaults for undeclared properties
  ##
  ## | property    | default          |
  ## | ----------- | ---------------- |
  ## | srcSpace    | spAny            |
  ## | dstSpace    | spAny            |
  ## | vecBytes    | 16               |
  ## | minAlign    | 1                |
  ## | zeroFill    | false            |
  ## | cache       | cache_default    |
  ## | transpose   | false            |
  ## | minCudaArch | 0                |
  ## | waitDepth   | 0                |
  ## | commitAtom  | UNIVERSAL_COMMIT |
  ## | waitAtom    | UNIVERSAL_WAIT   |
  result = newStmtList()
  var fields: seq[NimNode]
  for d in defs:
    fields.add d.name
  result.add newEnum(name = ident"CopyAtom", fields = fields, public = true, pure = false)
  for d in defs:
    let base = $d.name
    var have: seq[string]
    for (key, val) in d.props:
      result.add newConstStmt(
        nnkPostfix.newTree(ident"*", ident(base & "_" & key)), val)
      have.add key
    const defaults = [
      ("srcSpace", "spAny"), ("dstSpace", "spAny"), ("vecBytes", "16"),
      ("minAlign", "1"), ("zeroFill", "false"), ("cache", "cache_default"),
      ("transpose", "false"), ("minCudaArch", "0"), ("waitDepth", "0"),
      ("commitAtom", "\"UNIVERSAL_COMMIT\""), ("waitAtom", "\"UNIVERSAL_WAIT\"")]
    for (key, defVal) in defaults:
      if key in have: continue
      result.add newConstStmt(
        nnkPostfix.newTree(ident"*", ident(base & "_" & key)),
        parseStmt(defVal)[0])

macro declareCopyAtoms*(body: untyped): untyped =
  ## Parses the copy-atom registry block and expands to the CopyAtom enum
  ## plus the per-atom named consts.
  body.expectKind(nnkStmtList)
  var defs: seq[CopyAtomParams]
  defs.parseCopyAtomDecls(body)
  # Point every copy atom's commitAtom/waitAtom at the slot atoms matching
  # its instruction flavor
  # - an async spelling (cp.async family) degrades to the sm80-family group slots
  # - the blocking tier degrades to the universal discards
  var names: seq[string]
  for d in defs: names.add $d.name
  for d in mitems(defs):
    var kind, instr, haveCommit, haveWait: string
    for (k, v) in d.props:
      if k == "kind": kind = v.strVal
      elif k == "instr": instr = v.strVal
      elif k == "commitAtom": haveCommit = v.strVal
      elif k == "waitAtom": haveWait = v.strVal
    if kind != "ckCopy": continue
    let commitSlot = (if haveCommit != "": haveCommit
                      elif instr == "": "UNIVERSAL_COMMIT"
                      else: "SM80_CP_ASYNC_COMMIT")
    let waitSlot = (if haveWait != "": haveWait
                    elif instr == "": "UNIVERSAL_WAIT"
                    else: "SM80_CP_ASYNC_WAIT")
    for slot in [commitSlot, waitSlot]:
      doAssert slot in names,
        "declareCopyAtoms: copy atom `" & $d.name & "` points at slot `" &
        slot & "` which is not a declared atom"
    if haveCommit == "": d.props.add ("commitAtom", newStrLitNode(commitSlot))
    if haveWait == "": d.props.add ("waitAtom", newStrLitNode(waitSlot))
  result = defs.genCopyAtomDecls()
