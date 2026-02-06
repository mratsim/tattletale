# Absolute imports from package root
# --------------------------------------------------
--path:"."

# Imports
# --------------------------------------------------
import std/os, std/strutils

# Vendoring
# --------------------------------------------------

task install_libtorch, "Download and install libtorch":
  const libInstaller = "workspace/libtorch/vendor/libtorch_installer.nim"
  exec("nim cpp -r --skipParentCfg:on " & libInstaller)

# Test tasks
# --------------------------------------------------
# Compile with: nim cpp --outdir:build/tests --nimcache:nimcache/tests --hints:off --warnings:off

func testCmd(path: string): string =
  "nim cpp -r" &
  " --verbosity:0 --hints:off --warnings:off " &
  " --outdir:build/tests --nimcache:nimcache/tests " &
  path

iterator getTestCommands(path: string): string =
  ## Convention: tests start with test_ or t_
  for filepath in listFiles(path):
    let filename = filepath.extractFilename()
    if filename.endsWith(".nim") and (
      filename.startsWith("test_") or filename.startsWith("t_")
    ):
      yield testCmd(filepath)

proc runCmd(cmd: string) =
  echo "\n=============================================================================================="
  echo "Running '", cmd, "'"
  echo "=============================================================================================="
  exec cmd

task test_libtorch, "Test workspace/libtorch":
  for cmd in getTestCommands("workspace/libtorch/tests"):
    runCmd(cmd)

task test_safetensors, "Test workspace/safetensors":
  for cmd in getTestCommands("workspace/safetensors/tests"):
    runCmd(cmd)