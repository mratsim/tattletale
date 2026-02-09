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

const ProjectRoot = currentSourcePath.rsplit(DirSep, 1)[0]

# Utils
# --------------------------------------------------

proc runCmd(cmd: string) =
  echo "\n=============================================================================================="
  echo "Running '", cmd, "'"
  echo "=============================================================================================="
  exec cmd

func testerCmd(path: string): string =
  let filename = path.extractFilename()
  return
    "nim cpp -r" &
    " --verbosity:0 --hints:off --warnings:off " &
    &" --outdir:build/tests/{filename} --nimcache:nimcache/tests/{filename} " &
    path

func downloaderCmd(path: string): string =
  let filename = path.extractFilename()
  return
    "nim c -r" &
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

# Test tasks
# --------------------------------------------------
# Compile with: nim cpp --outdir:build/tests --nimcache:nimcache/tests --hints:off --warnings:off

iterator getTestCommands(path: string): string =
  ## Convention: tests start with test_ or t_
  for filepath in listFiles(path):
    let filename = filepath.extractFilename()
    if filename.endsWith(".nim") and (
      filename.startsWith("test_") or filename.startsWith("t_")
    ):
      yield testerCmd(filepath)

task test_libtorch, "Test workspace/libtorch":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/libtorch/tests"):
      runCmd(cmd)

task test_safetensors, "Test workspace/safetensors":
  withDir(ProjectRoot):
    for cmd in getTestCommands("workspace/safetensors/tests"):
      runCmd(cmd)

task test_toktoktok, "Test workspace/toktoktok":
  withDir(ProjectRoot):
    const fixturesDir = "workspace/toktoktok/tests/tokenizers"
    const gpt2Fixture = fixturesDir / "gpt2-tokenizer.json"
    const llama3Fixture = fixturesDir / "llama3-tokenizer.json"
    if not dirExists(fixturesDir) or not fileExists(gpt2Fixture) or not fileExists(llama3Fixture):
      echo "Downloading tokenizer fixtures..."
      download_test_tokenizersTask()
    for cmd in getTestCommands("workspace/toktoktok/tests"):
      runCmd(cmd)

# Python extension tasks
# --------------------------------------------------

func pytoktoktokBuildCmd(): string =
  return
    "nim c --app:lib" &
    " --verbosity:0 --hints:off --warnings:off" &
    " --outdir:workspace/toktoktok/tests" &
    " --nimcache:nimcache/pytoktoktok" &
    " -o:workspace/toktoktok/tests/pytoktoktok.so" &
    " workspace/toktoktok/tests/pytoktoktok.nim"

task make_pytoktoktok, "Build pytoktoktok.so for Python import":
  let cmd = pytoktoktokBuildCmd()
  withDir(ProjectRoot):
    runCmd(cmd)

# Per-file ENV variables configuration for PCRE2
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