# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU kernel compiler: CIRv2 lowering + Nim or C emission for `launchCpuFn`.
##
##   kernel wrapper → lowerCpuKernel → emitCpuCallees + emitCpuKernelBody
##
## Launch macro (`runtime/launch/launch_cpu_fn`): drives the pipeline
## at compile time:
##
## - it lowers the kernel wrapper, splices the callee procs, and inlines the kernel body into the launch loop nest it generates
## - there is no `cpu:` standalone macro, a kernel body without a launch
##   loop nest has no meaning on the CPU
## - the compile target is `ctCpu`, the CPU pass stage registers nothing for v0,
##   CIRv2 identifiers are Nim-derived, the emitters own target specifics
## - refusals and verifier violations surface as compile-time errors naming the construct

import std/macros

import ../builtins/builtins_compilermagic
import ./frontend/nim_to_crucible
import ./ir/cirv2_lang
import ./passes/pass_registry
import ./passes/passes_normalizations
import ./passes/passes_lowering
import ./passes/passes_validations
import ./passes/passes_lowering_cpu
import ./backends/nim_lang
import ./backends/c_lang

export nim_lang.CpuCoordTable, nim_lang.cpuCoordTable
export nim_lang.emitCpuCallees, nim_lang.emitCpuKernelBody
export c_lang.CpuCoordTableC, c_lang.cpuCoordTableC
export c_lang.emitCpuCalleesC, c_lang.emitCpuWorkgroupLaunchC
export c_lang.emitCpuTypeC, c_lang.kernelRuntimeParams

proc lowerCpuKernel*(wrapper: NimNode): CirModule =
  ## Lowers a `{.global.}` kernel wrapper to CIRv2 and runs the pass
  ## registry in stage order, CPU target.
  crucibleCompileTarget = ctCpu
  let (m, lerr) = lowerCirv2(wrapper)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  registerSharedLoweringPasses(reg)
  reg.registerCpuLoweringPasses()
  registerValidationPostPasses(reg)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  m
