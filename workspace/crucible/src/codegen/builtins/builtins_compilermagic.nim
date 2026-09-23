# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


type CompileTarget* = enum
  ctNone
  ctCuda   
  ctMetal  
  ctOpenCL 
  ctVulkan 
  ctWebGPU 

var crucibleCompileTarget* {.compileTime.}: CompileTarget = ctNone
  ## Active DSL target, the tag a `cuda:`/`metal:`/`opencl:`/`vulkan:`/
  ## `webgpu:` wrapper records for its body's compile-time resolution
  ## and its codegen macro restores to the pre-block value after.
var crucibleHostBackend* {.compileTime.}: CompileTarget = ctNone
  ## Host default, set only by `crucibleSetBackend`, survives DSL blocks.

macro crucibleSetBackend*(target: static CompileTarget) =
  ## Sets the host default backend for the rest of the module, re-callable.
  ##
  ## Usage:
  ## `crucibleSetBackend(ctMetal)` at module scope,
  ## before the first ccGetBackend()-derived default resolves.
  ##
  ## Postcondition:
  ## - ccGetBackend() returns `target` in host code outside any DSL block,
  ##   whatever blocks ran before or run after
  crucibleHostBackend = target

proc ccGetBackend*(): CompileTarget {.compileTime.} =
  ## Returns the backend of the enclosing DSL block.
  ## Outside any block, the host default.
  ## Usage: `when ccGetBackend() == ctMetal: ...` selects the Metal branch.
  ##
  ## Valid call sites:
  ## - inside a `cuda:` / `metal:` / `opencl:` / `vulkan:` / `webgpu:` block
  ## - in templates and generic procs instantiated from one
  ## - in host code after an explicit crucibleSetBackend call at module scope
  ##
  ## Elsewhere the call fails at compile time.
  ## A call in a runtime statement is rejected by Nim.
  let backend =
    if crucibleCompileTarget != ctNone: crucibleCompileTarget
    else: crucibleHostBackend
  doAssert backend != ctNone,
    "ccGetBackend: not inside a cuda:/metal:/opencl:/vulkan:/webgpu: block " &
    "and no crucibleSetBackend call at module scope"
  backend
