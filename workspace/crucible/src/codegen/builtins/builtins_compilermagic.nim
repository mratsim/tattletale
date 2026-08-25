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
  # ctNone until a DSL wrapper records its target.
  # ccGetBackend rejects calls outside any DSL block.

proc ccGetBackend*(): CompileTarget {.compileTime.} =
  ## Returns the backend the enclosing DSL block compiles for.
  ## Usage: `when ccGetBackend() == ctMetal: ...` selects the Metal branch.
  ## Call it only inside a `cuda:` / `metal:` / `opencl:` / `vulkan:` / `webgpu:` block,
  ## or in templates and generic procs instantiated from one.
  ## Outside a DSL block the call fails with a compile-time error.
  ## A call in a runtime statement is rejected by Nim.
  doAssert crucibleCompileTarget != ctNone,
    "ccGetBackend: not inside a cuda:/metal:/opencl:/vulkan:/webgpu: block"
  crucibleCompileTarget
