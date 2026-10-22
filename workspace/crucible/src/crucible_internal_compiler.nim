# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test surface of the crucible internals

import
  ./codegen/ir/gpu_ast_datatypes,
  ./codegen/ir/gpu_type_constructors,
  ./codegen/ir/gpu_ast_type_resolvers,
  ./codegen/frontend/nim_to_gpu,
  ./codegen/frontend/nim_to_gpu_type_resolvers,
  ./codegen/passes/pass_datatypes,
  ./codegen/passes/pass_registry,
  ./codegen/passes/passes_preprocessing,
  ./codegen/passes/passes_legalizations,
  ./codegen/passes/passes_validations,
  ./codegen/passes/passes_optimizations,
  ./codegen/passes/passes_normalizations,
  ./codegen/passes/passes_lowering,
  ./codegen/passes/passes_lowering_msl,
  ./codegen/passes/passes_lowering_cuda,
  ./codegen/passes/passes_lowering_glsl,
  ./codegen/passes/passes_lowering_opencl,
  ./codegen/passes/passes_lowering_wgsl,
  ./codegen/gpu_compiler,
  ./builtins/builtins_pragmas,
  ./builtins/builtins_functions

export gpu_compiler, gpu_ast_datatypes, gpu_type_constructors, gpu_ast_type_resolvers,
       nim_to_gpu, nim_to_gpu_type_resolvers,
       pass_datatypes, pass_registry,
       passes_preprocessing, passes_legalizations, passes_validations,
       passes_optimizations, passes_normalizations, passes_lowering,
       passes_lowering_msl, passes_lowering_cuda, passes_lowering_glsl,
       passes_lowering_opencl, passes_lowering_wgsl,
       builtins_pragmas, builtins_functions
