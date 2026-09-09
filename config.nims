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
    " -d:release --stackTrace:on --lineTrace:on --lineDir:on " &
    " --debugger:native " &
    " --hints:off --warnings:off " &
    &" --outdir:build/tests/{filename} --nimcache:nimcache/tests/{filename} " &
    path


func downloaderCmd(path: string): string =
  let filename = path.extractFilename()
  return
    "nim c -r -d:ssl -d:release --stackTrace:on --lineTrace:on --lineDir:on" &
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
    " -d:release --stackTrace:on --lineTrace:on --lineDir:on " &
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
    "nim cpp --app:lib" &
    " -d:release --stackTrace:on --lineTrace:on --lineDir:on " &
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
# Build with -d:release --stackTrace:on --lineTrace:on --lineDir:on:
# debug builds cannot parse GB-scale jsony fixtures.
# Compile with: nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on
#   --outdir:build/tests --nimcache:nimcache/tests --hints:off --warnings:off

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
    for cmd in getTestCommands("workspace/libtorch/tests/raw_torch_tensors", compiler = "nim cpp"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/libtorch/tests/tensors", compiler = "nim cpp"):
      runCmd(cmd)
    for cmd in getTestCommands("workspace/libtorch/tests/python_integration", compiler = "nim cpp"):
      runCmd(cmd)

task test_safetensors, "Test workspace/safetensors":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/safetensors/tests", compiler = "nim cpp"):
      runCmd(cmd)

# Granular transformer suite tasks
# ===================================================
# Per-suite and per-family tasks so an agent picks exactly the suites
# a change touched. The device-flip convention (harness/device.nim)
# rides the TTT_TEST_ON environment variable: its value becomes
# the compile-time define of every transformer suite command.
# Therefore `TTT_TEST_ON=cpu nim test_tf_bf16_qwen3_03_chain`
# flips the device of one suite.

proc tttDeviceDefine(): string =
  ## TTT_TEST_ON passthrough: an empty value adds nothing, a named
  ## device becomes the define, junk names fail loudly.
  let v = getEnv("TTT_TEST_ON")
  if v.len == 0:
    return ""
  case v
  of "auto", "metal", "cpu", "cuda":
    return " -d:TTT_TEST_ON=" & v
  else:
    echo "TTT_TEST_ON must name auto, metal, cpu or cuda, got: " & v
    quit(1)

proc transformerSuiteCmd(folder, filename: string): string =
  ## Build-and-run command of one transformer suite file, carrying
  ## the C++20 compile flag the transformer sources require.
  testerCmd("workspace/transformers/tests/" & folder & "/" & filename,
    extraFlags = tttDeviceDefine() & " --passC:\"-std=c++20\"",
    compiler = "nim cpp")

proc runTransformerSuite(folder, filename: string) =
  withDir(ProjectRoot):
    runCmd(transformerSuiteCmd(folder, filename))

proc familyName(): string =
  ## Family selector argument: `nim test_tf_family name=chain`
  ## on the command line, with TTT_TEST_FAMILY in the environment
  ## as the fallback.
  result = getEnv("TTT_TEST_FAMILY")
  for i in 2 .. paramCount():
    let p = paramStr(i)
    if p.startsWith("name="):
      result = p[5 .. ^1]

proc runFamily(suites: seq[tuple[folder, filename: string]]) =
  for s in suites:
    runTransformerSuite(s.folder, s.filename)

task test_tf_bf16_qwen3_03_chain, "Suite: Qwen3-0.6B 8+1 chain checkpoints":
  runTransformerSuite("q_bf16", "t_bf16_qwen3_03_chain.nim")
task test_tf_bf16_qwen3_05_ids_to_logits_inference, "Suite: Qwen3-0.6B ids to logits inference":
  runTransformerSuite("q_bf16", "t_bf16_qwen3_05_ids_to_logits_inference.nim")
task test_tf_bf16_qwen3_07_greedy_decoding, "Suite: Qwen3-0.6B greedy decoding":
  runTransformerSuite("q_bf16", "t_bf16_qwen3_07_greedy_decoding.nim")
task test_tf_bf16_qwen35dense_03_chain, "Suite: Qwen3.5-0.8B 8+1 chain checkpoints":
  runTransformerSuite("q_bf16", "t_bf16_qwen35dense_03_chain.nim")
task test_tf_bf16_qwen35dense_05_ids_to_logits_inference, "Suite: Qwen3.5-0.8B ids to logits inference":
  runTransformerSuite("q_bf16", "t_bf16_qwen35dense_05_ids_to_logits_inference.nim")
task test_tf_bf16_qwen35dense_07_greedy_decoding, "Suite: Qwen3.5-0.8B greedy decoding":
  runTransformerSuite("q_bf16", "t_bf16_qwen35dense_07_greedy_decoding.nim")
task test_tf_bf16_qwen35dense_single_file_checkpoint, "Suite: Qwen3.5-0.8B single-file checkpoint load":
  runTransformerSuite("q_bf16", "t_bf16_qwen35dense_single_file_checkpoint.nim")
task test_tf_bf16_qwen36moe_01_moe, "Suite: Qwen3.6-35B-A3B MoE expert math":
  runTransformerSuite("q_bf16", "t_bf16_qwen36moe_01_moe.nim")
task test_tf_bf16_qwen36moe_02_attn, "Suite: Qwen3.6-35B-A3B attention":
  runTransformerSuite("q_bf16", "t_bf16_qwen36moe_02_attn.nim")
task test_tf_bf16_qwen36moe_02_gdn, "Suite: Qwen3.6-35B-A3B Gated DeltaNet":
  runTransformerSuite("q_bf16", "t_bf16_qwen36moe_02_gdn.nim")
task test_tf_bf16_qwen36moe_03_layers, "Suite: Qwen3.6-35B-A3B decoder layers":
  runTransformerSuite("q_bf16", "t_bf16_qwen36moe_03_layers.nim")
task test_tf_bf16_qwen36moe_05_ids_to_logits_inference, "Suite: Qwen3.6-35B-A3B ids to logits inference":
  runTransformerSuite("q_bf16", "t_bf16_qwen36moe_05_ids_to_logits_inference.nim")
task test_tf_bf16_qwen36moe_07_greedy, "Suite: Qwen3.6-35B-A3B greedy decoding":
  runTransformerSuite("q_bf16", "t_bf16_qwen36moe_07_greedy.nim")
task test_tf_exl3_codec, "Suite: EXL3 trellis decode vs production kernel hash":
  runTransformerSuite("q_exl3", "t_exl3_codec.nim")
task test_tf_exl3_hadamard, "Suite: EXL3 hadamard vs production kernel":
  runTransformerSuite("q_exl3", "t_exl3_hadamard.nim")
task test_tf_exl3_qwen3_03_layers, "Suite: Qwen3-0.6B-EXL3 layer internals":
  runTransformerSuite("q_exl3", "t_exl3_qwen3_03_layers.nim")
task test_tf_exl3_qwen3_03b_layer2, "Suite: Qwen3-0.6B-EXL3 layer-02 stage trace":
  runTransformerSuite("q_exl3", "t_exl3_qwen3_03b_layer2.nim")
task test_tf_exl3_qwen3_05_ids_to_logits, "Suite: Qwen3-0.6B-EXL3 ids to logits inference":
  runTransformerSuite("q_exl3", "t_exl3_qwen3_05_ids_to_logits.nim")
task test_tf_exl3_qwen3_07_greedy, "Suite: Qwen3-0.6B-EXL3 greedy decoding":
  runTransformerSuite("q_exl3", "t_exl3_qwen3_07_greedy.nim")
task test_tf_bf16_unit_rope, "Suite: rope unit donors":
  runTransformerSuite("q_bf16", "t_bf16_unit_rope.nim")
task test_tf_bf16_unit_attn, "Suite: attention unit donors":
  runTransformerSuite("q_bf16", "t_bf16_unit_attn.nim")
task test_tf_kvcache_kvcache, "Suite: kvcache core (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_kvcache.nim")
task test_tf_kvcache_page_pool, "Suite: page pool lifecycle (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_page_pool.nim")
task test_tf_kvcache_orchestrator, "Suite: orchestrator (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_orchestrator.nim")
task test_tf_kvcache_radix_invariants, "Suite: radix trie invariants (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_radix_invariants.nim")
task test_tf_kvcache_fork_stability, "Suite: fork stability (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_fork_stability.nim")
task test_tf_kvcache_kvcache_lpm, "Suite: longest prefix match (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_kvcache_lpm.nim")
task test_tf_kvcache_codera020_batch_guard, "Suite: codera020 batch guard (cpu-only, model-free)":
  runTransformerSuite("kvcache", "test_codera020_batch_guard.nim")

task test_tf_harness_invariants, "Suite: harness analytic invariants":
  runTransformerSuite("harness", "t_harness_invariants.nim")
task test_tf_harness_selftest, "Suite: harness selftest (fault corpus)":
  runTransformerSuite("harness", "t_harness_selftest.nim")
task test_tf_sampler, "Suite: samplers":
  runTransformerSuite("samplers", "t_sampler.nim")
task test_tf_block_sparse_batch_property, "Suite: synthetic block-sparse batch property":
  runTransformerSuite("synthetic", "t_block_sparse_batch_property.nim")
task test_tf_deserialization_lmhead, "Suite: synthetic deserialization and lm head":
  runTransformerSuite("synthetic", "t_deserialization_lmhead.nim")

task test_tf_family, "Run one suite family (name=chain|ids|greedy|unit|moe|checkpoint|harness|sampler|synthetic|kvcache)":
  case familyName()
  of "chain":
    runFamily(@[
      ("q_bf16", "t_bf16_qwen3_03_chain.nim"),
      ("q_bf16", "t_bf16_qwen35dense_03_chain.nim")])
  of "ids":
    runFamily(@[
      ("q_bf16", "t_bf16_qwen3_05_ids_to_logits_inference.nim"),
      ("q_bf16", "t_bf16_qwen35dense_05_ids_to_logits_inference.nim"),
      ("q_bf16", "t_bf16_qwen36moe_05_ids_to_logits_inference.nim")])
  of "greedy":
    runFamily(@[
      ("q_bf16", "t_bf16_qwen3_07_greedy_decoding.nim"),
      ("q_bf16", "t_bf16_qwen35dense_07_greedy_decoding.nim"),
      ("q_bf16", "t_bf16_qwen36moe_07_greedy.nim")])
  of "unit":
    runFamily(@[
      ("q_bf16", "t_bf16_unit_rope.nim"),
      ("q_bf16", "t_bf16_unit_attn.nim")])
  of "moe":
    runFamily(@[
      ("q_bf16", "t_bf16_qwen36moe_01_moe.nim"),
      ("q_bf16", "t_bf16_qwen36moe_02_attn.nim"),
      ("q_bf16", "t_bf16_qwen36moe_02_gdn.nim"),
      ("q_bf16", "t_bf16_qwen36moe_03_layers.nim"),
      ("q_bf16", "t_bf16_qwen36moe_05_ids_to_logits_inference.nim"),
      ("q_bf16", "t_bf16_qwen36moe_07_greedy.nim")])
  of "checkpoint":
    runFamily(@[("q_bf16", "t_bf16_qwen35dense_single_file_checkpoint.nim")])
  of "harness":
    runFamily(@[
      ("harness", "t_harness_invariants.nim"),
      ("harness", "t_harness_selftest.nim")])
  of "sampler":
    runFamily(@[("samplers", "t_sampler.nim")])
  of "synthetic":
    runFamily(@[
      ("synthetic", "t_block_sparse_batch_property.nim"),
      ("synthetic", "t_deserialization_lmhead.nim")])
  of "kvcache":
    runFamily(@[
      ("kvcache", "test_kvcache.nim"),
      ("kvcache", "test_page_pool.nim"),
      ("kvcache", "test_orchestrator.nim"),
      ("kvcache", "test_radix_invariants.nim"),
      ("kvcache", "test_fork_stability.nim"),
      ("kvcache", "test_kvcache_lpm.nim"),
      ("kvcache", "test_codera020_batch_guard.nim")])
  else:
    echo "unknown family: name the family chain, ids, greedy, unit, moe, checkpoint, harness, sampler or synthetic"
    quit(1)

task test_transformers, "Test workspace/transformers (the full set, final verification)":
  withDir(ProjectRoot):
    runFamily(@[
      ("q_bf16", "t_bf16_unit_rope.nim"),
      ("q_bf16", "t_bf16_unit_attn.nim"),
      ("q_bf16", "t_bf16_qwen3_03_chain.nim"),
      ("q_bf16", "t_bf16_qwen3_05_ids_to_logits_inference.nim"),
      ("q_bf16", "t_bf16_qwen3_07_greedy_decoding.nim"),
      ("q_bf16", "t_bf16_qwen35dense_03_chain.nim"),
      ("q_bf16", "t_bf16_qwen35dense_05_ids_to_logits_inference.nim"),
      ("q_bf16", "t_bf16_qwen35dense_07_greedy_decoding.nim"),
      ("q_bf16", "t_bf16_qwen35dense_single_file_checkpoint.nim"),
      ("q_bf16", "t_bf16_qwen36moe_01_moe.nim"),
      ("q_bf16", "t_bf16_qwen36moe_02_attn.nim"),
      ("q_bf16", "t_bf16_qwen36moe_02_gdn.nim"),
      ("q_bf16", "t_bf16_qwen36moe_03_layers.nim"),
      ("q_bf16", "t_bf16_qwen36moe_05_ids_to_logits_inference.nim"),
      ("q_bf16", "t_bf16_qwen36moe_07_greedy.nim"),
      ("harness", "t_harness_invariants.nim"),
      ("harness", "t_harness_selftest.nim"),
      ("samplers", "t_sampler.nim"),
      ("synthetic", "t_block_sparse_batch_property.nim"),
      ("synthetic", "t_deserialization_lmhead.nim"),
      ("kvcache", "test_kvcache.nim"),
      ("kvcache", "test_page_pool.nim"),
      ("kvcache", "test_orchestrator.nim"),
      ("kvcache", "test_radix_invariants.nim"),
      ("kvcache", "test_fork_stability.nim"),
      ("kvcache", "test_kvcache_lpm.nim"),
      ("kvcache", "test_codera020_batch_guard.nim")])

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
    for cmd in getTestCommands("workspace/ceramic/tests/atoms_mma"):
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

task test_crucible_metal, "Test workspace/crucible Metal codegen":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/crucible/tests/codegen/metal"):
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

# Per-file compile options for the vendored zstd
# ══════════════════════════════════════════════════

const ZstdDir = ProjectRoot/"workspace/zstd"

# Under nim cpp the -std=c++20 passC of the transformer suites reaches
# every compiled file and the C driver rejects that flag for plain .c
# inputs. The vendored zstd compiles clean as C++ at v1.5.7, a probe
# result recorded in workspace/zstd/vendor/README.md, so every source
# is told its true language before the C++ standard reaches it.
# Sources live in the submodule lib/ subtree (pcre2 vendoring shape).
# The put keys are file stems. The vendored tree therefore stays
# free of basename collisions, verified at vendor time for v1.5.7.

const ZstdSources = [
  "common/debug",
  "common/entropy_common",
  "common/error_private",
  "common/fse_decompress",
  "common/pool",
  "common/threading",
  "common/xxhash",
  "common/zstd_common",
  "compress/fse_compress",
  "compress/hist",
  "compress/huf_compress",
  "compress/zstd_compress",
  "compress/zstd_compress_literals",
  "compress/zstd_compress_sequences",
  "compress/zstd_compress_superblock",
  "compress/zstd_double_fast",
  "compress/zstd_fast",
  "compress/zstd_lazy",
  "compress/zstd_ldm",
  "compress/zstd_opt",
  "compress/zstd_preSplit",
  "compress/zstdmt_compress",
  "decompress/huf_decompress",
  "decompress/zstd_ddict",
  "decompress/zstd_decompress",
  "decompress/zstd_decompress_block",
  "dictBuilder/cover",
  "dictBuilder/divsufsort",
  "dictBuilder/fastcover",
  "dictBuilder/zdict",
]

for zstdSource in ZstdSources:
  put(zstdSource.rsplit("/", 1)[1] & ".always", "-x c++")
