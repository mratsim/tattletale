# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU kernel compiler: CIRv2 lowering + C emission for `launchCpuFn`.
##
##   kernel wrapper → lowerCpuKernel → emitCpuCalleesC + emitCpuWorkgroupLaunchC

import std/macros

import ../builtins/builtins_compilermagic
import ./frontend/nim_to_crucible
import ./ir/cirv2_lang
import ./passes/pass_datatypes
import ./passes/passes_normalizations
import ./passes/passes_legalization
import ./passes/passes_lowering
import ./passes/passes_validations
import ./passes/passes_lowering_cpu
import ./backends/lang_c

export lang_c.CpuCoordTableC, lang_c.cpuCoordTableC
export lang_c.emitCpuCalleesC, lang_c.emitCpuWorkgroupLaunchC
export lang_c.emitCpuSerialTailC, lang_c.emitCpuDataFillC
export lang_c.kernelRuntimeParams

proc lowerCpuKernel*(wrapper: NimNode): CirModule =
  let savedTarget = crucibleCompileTarget
  crucibleCompileTarget = ctCpu
  let m = lowerCirv2(wrapper)
  var reg = PassRegistry()
  registerRewriteStmtsExprs(reg)
  registerResolveBoolDomain(reg)
  registerValueNormalizations(reg)
  registerReshapingPasses(reg)
  registerSharedLoweringPasses(reg)
  reg.registerCpuLoweringPasses()
  registerValidationPostPasses(reg)
  reg.runPasses(m)
  crucibleCompileTarget = savedTarget
  m
