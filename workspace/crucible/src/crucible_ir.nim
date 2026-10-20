# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test surface of the crucible IR: one import for IR-level suites.
##
## The IR is private, this module re-exports it for IR tests,

import
  ./codegen/ir/gpu_ast_datatypes,
  ./codegen/ir/gpu_type_constructors,
  ./codegen/ir/gpu_ast_type_resolvers,
  ./codegen/ir/nim_to_gpu,
  ./codegen/ir/nim_to_gpu_type_resolvers,
  ./codegen/passes/pass_datatypes,
  ./codegen/passes/pass_registry,
  ./codegen/passes/passes_preprocessing,
  ./codegen/passes/passes_legalizations,
  ./codegen/passes/passes_validations,
  ./codegen/passes/passes_optimizations,
  ./codegen/passes/passes_normalizations,
  ./codegen/passes/passes_lowering,
  ./codegen/passes/passes_metal,
  ./codegen/passes/passes_cuda,
  ./codegen/passes/passes_vulkan,
  ./codegen/passes/passes_opencl,
  ./codegen/passes/passes_wgsl,
  ./codegen/gpu_compiler,
  ./codegen/builtins/builtins_pragmas,
  ./codegen/builtins/builtins_functions

export gpu_compiler, gpu_ast_datatypes, gpu_type_constructors, gpu_ast_type_resolvers,
       nim_to_gpu, nim_to_gpu_type_resolvers,
       pass_datatypes, pass_registry,
       passes_preprocessing, passes_legalizations, passes_validations,
       passes_optimizations, passes_normalizations, passes_lowering,
       passes_metal, passes_cuda, passes_vulkan, passes_opencl, passes_wgsl,
       builtins_pragmas, builtins_functions
