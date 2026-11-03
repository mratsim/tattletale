# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU kernel compiler: CIRv2 lowering + C emission for `launchCpuFn`.
##
##   kernel wrapper → lowerCpuKernel → emitCpuCalleesC + emitCpuWorkgroupLaunchC
##
## Launch macro (`runtime/launch/launch_cpu_fn`): drives the pipeline
## at compile time:
##
## - it lowers the kernel wrapper, prints the callee procs and the kernel body
##   into the generated launch unit. The launch macro assembles the two
## - there is no `cpu:` standalone macro, a kernel body without a launch
##   has no meaning on the CPU
## - the compile target is `ctCpu`, the CPU pass stage registers the C
##   reserved-word rename and the array-assign decomposition
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
import ./backends/lang_c

export lang_c.CpuCoordTableC, lang_c.cpuCoordTableC
export lang_c.emitCpuCalleesC, lang_c.emitCpuWorkgroupLaunchC
export lang_c.emitCpuSerialTailC, lang_c.emitCpuDataFillC
export lang_c.emitCpuTypeC, lang_c.kernelRuntimeParams

proc lowerCpuKernel*(wrapper: NimNode): CirModule =
  let savedTarget = crucibleCompileTarget
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
  crucibleCompileTarget = savedTarget
  m
