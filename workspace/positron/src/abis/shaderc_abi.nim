## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Minimal Nim bindings for libshaderc_shared.so — SPIR-V compilation from GLSL.

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

type
  shaderc_compiler_t* = pointer
  shaderc_compile_options_t* = pointer
  shaderc_compilation_result_t* = pointer

  shaderc_shader_kind* = enum
    shaderc_glsl_infer_from_source = 0
    shaderc_glsl_default_vertex_shader = 1
    shaderc_glsl_default_fragment_shader = 2
    shaderc_glsl_default_compute_shader = 3

  shaderc_target_env* = enum
    shaderc_target_env_vulkan = 0
    shaderc_target_env_opengl = 1
    shaderc_target_env_opengl_compat = 2
    shaderc_target_env_webgpu = 3
    shaderc_target_env_glslang = 4

  shaderc_spirv_version* = enum
    shaderc_spirv_version_1_0 = 0x00010000'u32
    shaderc_spirv_version_1_1 = 0x00010100'u32
    shaderc_spirv_version_1_2 = 0x00010200'u32
    shaderc_spirv_version_1_3 = 0x00010300'u32
    shaderc_spirv_version_1_4 = 0x00010400'u32
    shaderc_spirv_version_1_5 = 0x00010500'u32
    shaderc_spirv_version_1_6 = 0x00010600'u32

  shaderc_source_language* = enum
    shaderc_source_language_glsl = 0
    shaderc_source_language_hlsl = 1
# ═══════════════════════════════════════════════════════════════════════
# Library
# ═══════════════════════════════════════════════════════════════════════

const ShadercLib* =
  when defined(windows): "shaderc_shared.dll"
  elif defined(macosx):  "libshaderc_shared.dylib"
  else:                  "libshaderc_shared.so"

# ═══════════════════════════════════════════════════════════════════════
# Compiler API
# ═══════════════════════════════════════════════════════════════════════

proc shaderc_compiler_initialize*(): shaderc_compiler_t
  {.importc: "shaderc_compiler_initialize", dynlib: ShadercLib.}

proc shaderc_compiler_release*(compiler: shaderc_compiler_t): void
  {.importc: "shaderc_compiler_release", dynlib: ShadercLib.}

proc shaderc_compile_into_spv*(compiler: shaderc_compiler_t,
                               source: cstring,
                               source_size: csize_t,
                               shader_kind: shaderc_shader_kind,
                               input_file_name: cstring,
                               entry_point_name: cstring,
                               additional_options: shaderc_compile_options_t): shaderc_compilation_result_t
  {.importc: "shaderc_compile_into_spv", dynlib: ShadercLib.}

# ═══════════════════════════════════════════════════════════════════════
# Result API
# ═══════════════════════════════════════════════════════════════════════

proc shaderc_result_get_length*(result: shaderc_compilation_result_t): csize_t
  {.importc: "shaderc_result_get_length", dynlib: ShadercLib.}

proc shaderc_result_get_bytes*(result: shaderc_compilation_result_t): pointer
  {.importc: "shaderc_result_get_bytes", dynlib: ShadercLib.}

proc shaderc_result_release*(result: shaderc_compilation_result_t): void
  {.importc: "shaderc_result_release", dynlib: ShadercLib.}

proc shaderc_result_get_compilation_status*(result: shaderc_compilation_result_t): int32
  {.importc: "shaderc_result_get_compilation_status", dynlib: ShadercLib.}

proc shaderc_result_get_error_message*(result: shaderc_compilation_result_t): cstring
  {.importc: "shaderc_result_get_error_message", dynlib: ShadercLib.}

# ═══════════════════════════════════════════════════════════════════════
# Compile options API
# ═══════════════════════════════════════════════════════════════════════

proc shaderc_compile_options_initialize*(): shaderc_compile_options_t
  {.importc: "shaderc_compile_options_initialize", dynlib: ShadercLib.}

proc shaderc_compile_options_release*(options: shaderc_compile_options_t): void
  {.importc: "shaderc_compile_options_release", dynlib: ShadercLib.}

proc shaderc_compile_options_set_target_env*(options: shaderc_compile_options_t,
                                               target: shaderc_target_env,
                                               version: uint32): void
  {.importc: "shaderc_compile_options_set_target_env", dynlib: ShadercLib.}

proc shaderc_compile_options_set_target_spirv*(options: shaderc_compile_options_t,
                                                 version: shaderc_spirv_version): void
  {.importc: "shaderc_compile_options_set_target_spirv", dynlib: ShadercLib.}

proc shaderc_compile_options_set_source_language*(options: shaderc_compile_options_t,
                                                    lang: shaderc_source_language): void
  {.importc: "shaderc_compile_options_set_source_language", dynlib: ShadercLib.}
