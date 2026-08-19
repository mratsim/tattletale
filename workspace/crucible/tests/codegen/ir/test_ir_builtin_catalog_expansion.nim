## Builtin catalog: in-macro alias expansion + identifier-collision sweep.
##
## The alias set is the per-idiom coordinate spellings: CUDA `blockIdx`,
## OpenCL `get_global_id`, GLSL `gl_GlobalInvocationID`, WGSL `global_id`,
## etc. They are implemented as zero-arg and static-dim templates that
## expand to the canonical MSL names during sem, so the IR only ever
## contains canonical names. This test pins the expansion *inside* a typed
## GPU macro body (the `opencl:` macro specifically).
## The runtime-dim loud-error probe fires inside the macro body:
##   - dot context:     `blockIdx.x`      -> `threadgroup_position_in_grid.x`
##   - positional:      `blockIdx[1]`     -> `threadgroup_position_in_grid.y`
##   - static-dim:      `get_global_id(d)` folds to the `d`-th component
##   - runtime dim `get_global_id(d)` is a loud Nim static error
##
## The collision sweep asserts no two catalog names are identifier-equal under
## Nim's case/underscore-insensitive matching. The only intentional pair is
## `workgroup_barrier` ≡ `workgroupBarrier`: one defaulted-param declaration
## serves both the OpenCL flags arity and the WGSL zero-arg spelling.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_builtin_catalog_expansion.nim

import std/[strutils, tables, sequtils]
import workspace/crucible

# ═══════════════════════════════════════════════════════════════════════
# 1. In-macro template expansion (verified inside `opencl:`)
# ═══════════════════════════════════════════════════════════════════════
const expansionKernel = opencl:
  proc expansionKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    C[0] = blockIdx.x              # dot context
    C[1] = blockIdx[1]             # positional tuple-bracket access
    C[2] = get_global_id(0'u32)    # static-dim: folds to component 0
    C[3] = get_global_id(1'u32)    # static-dim: folds to component 1
    C[4] = get_global_id(2'u32)    # static-dim: folds to component 2
    C[5] = thread_index_in_threadgroup
    syncthreads()                  # CUDA-idiom barrier alias
    barrier(CLK_LOCAL_MEM_FENCE)   # OpenCL-idiom barrier alias

block:
  # Aliases must expand to canonical names, and the OpenCL printer must map
  # each canonical back to its native `get_*(d)` call. A canonical name
  # leaking into the output would mean the alias never expanded
  # (or the printer case missed it).
  doAssert "thread_position_in_grid" notin expansionKernel
  doAssert "threadgroup_position_in_grid" notin expansionKernel
  doAssert "thread_position_in_threadgroup" notin expansionKernel
  doAssert "thread_index_in_threadgroup" notin expansionKernel
  # dot context: blockIdx.x -> threadgroup_position_in_grid.x -> get_group_id(0)
  doAssert "get_group_id(0)" in expansionKernel
  # positional: blockIdx[1] -> threadgroup_position_in_grid.y -> get_group_id(1)
  doAssert "get_group_id(1)" in expansionKernel
  # static-dim get_global_id(d) -> thread_position_in_grid.d -> get_global_id(d)
  doAssert "get_global_id(0)" in expansionKernel
  doAssert "get_global_id(1)" in expansionKernel
  doAssert "get_global_id(2)" in expansionKernel
  # flat local index: x-major linearization of get_local_id / get_local_size
  doAssert "(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))" in expansionKernel
  # both barrier aliases fold into the canonical call -> native spelling
  doAssert expansionKernel.count("barrier(CLK_LOCAL_MEM_FENCE)") == 2
  echo "  OK — in-macro alias expansion (dot / positional / static-dim / barrier aliases)"

block:
  # Runtime dim: a non-static argument cannot bind the `static uint32` template
  # param, so `get_global_id(d)` with a runtime `d` is a loud Nim error.
  # The rejection is verified here inside the `opencl:` macro body.
  static:
    doAssert not compiles(block:
      const bad = opencl:
        proc bad(C: ptr UncheckedArray[uint32]) {.global.} =
          let d = 0'u32
          C[0] = get_global_id(d)
    )
  echo "  OK — runtime-dim get_global_id(d) rejected loudly inside the macro"

# ═══════════════════════════════════════════════════════════════════════
# 2. Identifier-collision sweep (identifier-equal names under Nim matching)
# ═══════════════════════════════════════════════════════════════════════
const CatalogSource = staticRead("../../../src/codegen/builtins/builtins_catalog.nim")

proc declaredCatalogNames(src: string): seq[string] =
  ## Extracts the exported declaration names from the catalog source:
  ## `let` / `proc` / `template` / `const` declarations and the `uvec3` type.
  for line in src.splitLines():
    let s = line.strip()
    var name = ""
    if s.startsWith("template ") or s.startsWith("proc ") or
       s.startsWith("let ") or s.startsWith("const "):
      let rest = s[s.find(' ') + 1 .. ^1]
      for c in rest:
        if c in {'*', '(', '{', ':', '[', ' ', '='}:
          break
        name.add c
    elif s.contains("* = tuple"):
      # `uvec3* = tuple[x, y, z: uint32]`: the type declaration
      for c in s:
        if c == '*':
          break
        name.add c
    if name.len > 0:
      result.add name

block:
  var names = declaredCatalogNames(CatalogSource)
  doAssert names.len > 20, "catalog name extraction must find the full declaration set, got " & $names.len
  # `barrier` is deliberately overloaded (0-arg Vulkan, flags OpenCL).
  # Two declarations share one identifier: not a collision.
  names = names.deduplicate()

  # The intentional style-insensitive twin: WGSL `workgroupBarrier` is
  # identifier-equal to `workgroup_barrier`. One declaration serves both
  # arities, so the twin is the only spelling that collides with a declared
  # name.
  names.add "workgroupBarrier"

  proc norm(name: string): string =
    ## Nim default identifier matching: case- and underscore-insensitive.
    name.replace("_", "").toLowerAscii()

  var byNorm = initTable[string, seq[string]]()
  for name in names:
    byNorm.mgetOrPut(norm(name), @[]).add name

  for normalized, spellings in byNorm.pairs:
    if spellings.len > 1:
      doAssert spellings.len == 2 and "workgroup_barrier" in spellings and
        "workgroupBarrier" in spellings,
        "unexpected identifier collision: " & $spellings
  echo "  OK — collision sweep: no unintended identifier-equal catalog names (only workgroup_barrier ≡ workgroupBarrier)"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All builtin catalog expansion + collision tests passed."
