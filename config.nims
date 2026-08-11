# Absolute imports from package root
# --------------------------------------------------
--path:"."

# Task-level dependencies
# --------------------------------------------------
# taskRequires "download_test_tokenizers", "chronos >= 4.2.0"

# Imports
# --------------------------------------------------
import std/os, std/strutils, std/strformat

# Project root
# --------------------------------------------------
#
# We want to be able to execute tasks even when we `cd` into subfolders

const ProjectRoot = currentSourcePath().parentDir()

# Dependencies
# --------------------------------------------------
# Gathered from each subpackage's .nimble file.
# `deps` = runtime dependencies (needed for compilation/testing)
# `deps_dev` = dev-only dependencies (nim install_libtorch, test tokenizers downloads)

const deps = [
  "nimpy >= 0.2.1",        # libtorch: Python interop
  "jsony",                  # safetensors, toktoktok: JSON parsing
  "stew",                   # safetensors: bit manipulation
  "packedjson@#head",       # transformers: JSON config loading (needs shallowCopy fix)
  "https://github.com/yglukhov/iface",  # transformers: interface support
]

const deps_dev: seq[string] = @[
  "zip",       # dev: nim install_libtorch (download/extract libtorch)
  "chronos",   # dev: download_test_tokenizers (HTTP async)
] & newSeq[string]()

task install_deps, "Install runtime dependencies":
  exec "nimble install " & deps.join(" ")

task install_deps_dev, "Install dev-only dependencies (zip, chronos)":
  exec "nimble install " & deps_dev.join(" ")

# Build libpositron_cuda.a (CUDA kernel static library)
# ---------------------------------------------------

task make_libpositron_cuda, "Build Positron Cuda kernels in static library":
  # Compiles make_libpositron_cuda.cu directly with nvcc.
  # The .cu file #include's all kernel source files as a single translation unit.
  # Caller must have nvcc on PATH (e.g. export PATH="$VENV/lib/python3.14/site-packages/nvidia/cu13/bin:$PATH")
  exec("mkdir -p build/")
  exec "nvcc -lib -O3 --use_fast_math --std=c++17 -o build/libpositron_cuda.a workspace/positron/make_libpositron_cuda.cu"

# Utils
# --------------------------------------------------

proc runCmd(cmd: string) =
  echo "\n=============================================================================================="
  echo "Running '", cmd, "'"
  echo "=============================================================================================="
  exec cmd

func testerCmd(path: string; extraFlags = ""; compiler = "nim c"): string =
  let filename = path.extractFilename()
  return
    compiler & " -r" &
    (if extraFlags.len > 0: " " & extraFlags else: "") &
    " --debugger:native " &
    " --hints:off --warnings:off " &
    &" --outdir:build/tests/{filename} --nimcache:nimcache/tests/{filename} " &
    path

func downloaderCmd(path: string): string =
  let filename = path.extractFilename()
  return
    "nim c -r -d:ssl" &
    " --verbosity:0 --hints:off --warnings:off " &
    &" --outdir:build/downloaders/{filename} --nimcache:nimcache/downloaders/{filename} " &
    path

# Vendoring
# --------------------------------------------------

task install_libtorch, "Download and install libtorch":
  const libInstaller = "workspace/libtorch/vendor/libtorch_installer.nim"
  let cmd = downloaderCmd(libInstaller)
  withDir(ProjectRoot):
    runCmd(cmd)

task download_test_tokenizers, "Download gpt-2 and llama3 tokenizers for testing":
  const tokDownloader = "workspace/toktoktok/tests/download_tokenizers.nim"
  let cmd = downloaderCmd(tokDownloader)
  withDir(ProjectRoot):
    runCmd(cmd)


# Python extension tasks
# --------------------------------------------------

func pytoktoktokBuildCmd(): string =
  return
    "nim c --app:lib" &
    # " -d:release --stackTrace:on --lineTrace:on " &
    " --debugger:native " &
    " --verbosity:0 --hints:off --warnings:off" &
    " --outdir:workspace/toktoktok/tests" &
    " --nimcache:nimcache/pytoktoktok" &
    " -o:workspace/toktoktok/tests/pytoktoktok.so" &
    " workspace/toktoktok/tests/pytoktoktok.nim"

task make_pytoktoktok, "Build pytoktoktok.so for Python import":
  let cmd = pytoktoktokBuildCmd()
  withDir(ProjectRoot):
    runCmd(cmd)

func pytttransformersBuildCmd(): string =
  return
    "nim c --app:lib" &
    " --debugger:native " &
    " --verbosity:0 --hints:off --warnings:off" &
    " --outdir:workspace/transformers/tests" &
    " --nimcache:nimcache/pytttransformers" &
    " -o:workspace/transformers/tests/pytttransformers.so" &
    " workspace/transformers/tests/pytttransformers.nim"

task make_pytttransformers, "Build pytttransformers.so for Python import":
  let cmd = pytttransformersBuildCmd()
  withDir(ProjectRoot):
    runCmd(cmd)

# Test tasks
# --------------------------------------------------
# Compile with: nim cpp --outdir:build/tests --nimcache:nimcache/tests --hints:off --warnings:off

iterator getTestCommands(path: string; extraFlags = ""; compiler = "nim c"): string =
  ## Convention: tests start with test_ or t_
  for filepath in listFiles(path):
    let filename = filepath.extractFilename()
    if filename.endsWith(".nim") and (
      filename.startsWith("test_") or filename.startsWith("t_")
    ):
      yield testerCmd(filepath, extraFlags = extraFlags, compiler = compiler)

task test_libtorch, "Test workspace/libtorch":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/libtorch/tests/raw_torch_tensors"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/libtorch/tests/tensors"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/libtorch/tests/python_integration"):
      runCmd(cmd)

task test_safetensors, "Test workspace/safetensors":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/safetensors/tests"):
      runCmd(cmd)

task test_transformers, "Test workspace/transformers":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/transformers/tests"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/transformers/tests/q_bf16"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/transformers/tests/q_exl3"):
      runCmd(cmd)

task test_toktoktok, "Test workspace/toktoktok":
  withDir(ProjectRoot):
    const fixturesDir = "workspace/toktoktok/tests/tokenizers"
    const gpt2Fixture = fixturesDir / "gpt2-tokenizer.json"
    const llama3Fixture = fixturesDir / "llama3-tokenizer.json"
    if not dirExists(fixturesDir) or not fileExists(gpt2Fixture) or not fileExists(llama3Fixture):
      echo "Downloading tokenizer fixtures..."
      download_test_tokenizersTask()

    # Ensure we regenerate the dynlib
    make_pytoktoktokTask()
    for cmd in getTestCommands("workspace/toktoktok/tests"):
      runCmd(cmd)

task test_ceramic, "Test workspace/ceramic":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/ceramic/tests"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/ceramic/tests/gemm"):
      runCmd(cmd)

task test_crucible_nvrtc, "Test workspace/crucible NVRTC codegen":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/crucible/tests/codegen/nvrtc"):
      runCmd(cmd)

# Per-file ENV variables configuration for PCRE2

task test_crucible_opencl, "Test workspace/crucible OpenCL codegen":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/crucible/tests/codegen/opencl"):
      runCmd(cmd)

task test_crucible_vulkan, "Test workspace/crucible Vulkan codegen":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/crucible/tests/codegen/vulkan"):
      runCmd(cmd)

task test_crucible_webgpu, "Test workspace/crucible WebGPU codegen":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/crucible/tests/codegen/webgpu"):
      runCmd(cmd)

# ---------------------------------------------------

const Pcre2Dir = ProjectRoot/"workspace/pcre2"

const CONFIG_H =
  # Include local pcre2.h
  " -I" & Pcre2Dir/"vendor" &
  " -I" & Pcre2Dir/"vendor/pcre2/src" &
  # Platform OS/Compile specific
  " -DHAVE_ASSERT_H=true" &
  (when defined(windows):
    " -DHAVE_WINDOWS_H=true"
  else:
    " -DHAVE_UNISTD_H=true") &
  " -DHAVE_ATTRIBUTE_UNINITIALIZED=true" &
  " -DHAVE_BUILTIN_MUL_OVERFLOW=true" &
  " -DHAVE_BUILTIN_UNREACHABLE=true" &
  # PCRE2 specific
  " -DSUPPORT_PCRE2_8=true" &
  " -DSUPPORT_PCRE2_16=false" &
  " -DSUPPORT_PCRE2_32=false" &
  " -DSUPPORT_UNICODE=true" &
  " -DSUPPORT_JIT=true" &
  # config-cmake.h.in
  " -DPCRE2_EXPORT=\"\"" &
  " -DLINK_SIZE=2" &
  " -DHEAP_LIMIT=20000000" &
  " -DMATCH_LIMIT=10000000" &
  " -DMATCH_LIMIT_DEPTH=\"MATCH_LIMIT\"" &
  " -DMAX_VARLOOKBEHIND=255" &
  " -DNEWLINE_DEFAULT=2" &
  " -DPARENS_NEST_LIMIT=250" &
  " -DPCRE2GREP_BUFSIZE=20480" &
  " -DPCRE2GREP_MAX_BUFSIZE=1048576" &
  " -DMAX_NAME_SIZE=128" &
  " -DMAX_NAME_COUNT=10000" &
  # Devops
  " -UHAVE_CONFIG_H" &
  " -DPCRE2_CODE_UNIT_WIDTH=8" &
  " -DPCRE2_STATIC"

put("pcre2_chartables.always", CONFIG_H)
put("pcre2_auto_possess.always", CONFIG_H)
put("pcre2_chkdint.always", CONFIG_H)
put("pcre2_compile.always", CONFIG_H)
put("pcre2_compile_cgroup.always", CONFIG_H)
put("pcre2_compile_class.always", CONFIG_H)
put("pcre2_config.always", CONFIG_H)
put("pcre2_context.always", CONFIG_H)
put("pcre2_convert.always", CONFIG_H)
put("pcre2_dfa_match.always", CONFIG_H)
put("pcre2_error.always", CONFIG_H)
put("pcre2_extuni.always", CONFIG_H)
put("pcre2_find_bracket.always", CONFIG_H)
put("pcre2_jit_compile.always", CONFIG_H)
put("pcre2_maketables.always", CONFIG_H)
put("pcre2_match.always", CONFIG_H)
put("pcre2_match_data.always", CONFIG_H)
put("pcre2_match_next.always", CONFIG_H)
put("pcre2_newline.always", CONFIG_H)
put("pcre2_ord2utf.always", CONFIG_H)
put("pcre2_pattern_info.always", CONFIG_H)
put("pcre2_script_run.always", CONFIG_H)
put("pcre2_serialize.always", CONFIG_H)
put("pcre2_string_utils.always", CONFIG_H)
put("pcre2_study.always", CONFIG_H)
put("pcre2_substitute.always", CONFIG_H)
put("pcre2_substring.always", CONFIG_H)
put("pcre2_tables.always", CONFIG_H)
put("pcre2_ucd.always", CONFIG_H)
put("pcre2_valid_utf.always", CONFIG_H)
put("pcre2_xclass.always", CONFIG_H)
