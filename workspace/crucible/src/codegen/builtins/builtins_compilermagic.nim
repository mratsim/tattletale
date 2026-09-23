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
  ## Compile-time backend tag consulted by ccGetBackend.
  ##
  ## - ctNone until a DSL wrapper records its target
  ##   or a crucibleSetBackend call at module scope
  ## - a DSL wrapper overrides the tag per block
  ## - the tag is never reset to ctNone after a block

macro crucibleSetBackend*(target: static CompileTarget) =
  ## Sets crucibleCompileTarget to `target` for the rest of the module.
  ##
  ## Usage:
  ## `crucibleSetBackend(ctMetal)` at module scope,
  ## before the first ccGetBackend()-derived default resolves.
  ##
  ## Postcondition:
  ## - ccGetBackend() returns `target` in host code outside any DSL block
  ##
  ## DSL wrappers override the tag per block.
  crucibleCompileTarget = target

proc ccGetBackend*(): CompileTarget {.compileTime.} =
  ## Returns the backend the enclosing DSL block compiles for.
  ## Usage: `when ccGetBackend() == ctMetal: ...` selects the Metal branch.
  ##
  ## Valid call sites:
  ## - inside a `cuda:` / `metal:` / `opencl:` / `vulkan:` / `webgpu:` block
  ## - in templates and generic procs instantiated from one
  ## - after an explicit crucibleSetBackend call at module scope
  ##
  ## Elsewhere the call fails at compile time.
  ## A call in a runtime statement is rejected by Nim.
  doAssert crucibleCompileTarget != ctNone,
    "ccGetBackend: not inside a cuda:/metal:/opencl:/vulkan:/webgpu: block " &
    "and no crucibleSetBackend call at module scope"
  crucibleCompileTarget
