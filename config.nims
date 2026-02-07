# Absolute imports from package root
# --------------------------------------------------
--path:"."

# Task-level dependencies
# --------------------------------------------------
# taskRequires "download_test_tokenizers", "chronos >= 4.2.0"

# Imports
# --------------------------------------------------
import std/os, std/strutils, std/strformat

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
  runCmd(cmd)

task download_test_tokenizers, "Download gpt-2 and llama3 tokenizers for testing":
  const tokDownloader = "workspace/toktoktok/tests/download_tokenizers.nim"
  let cmd = downloaderCmd(tokDownloader)
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
  for cmd in getTestCommands("workspace/libtorch/tests"):
    runCmd(cmd)

task test_safetensors, "Test workspace/safetensors":
  for cmd in getTestCommands("workspace/safetensors/tests"):
    runCmd(cmd)

task test_toktoktok, "Test workspace/toktoktok":
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
  runCmd(cmd)
