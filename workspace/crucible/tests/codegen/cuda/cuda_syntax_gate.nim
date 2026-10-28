## CUDA-C++ syntax check utilities:
## - Apple clang type-checks one emitted CUDA-C++ source as plain C++
##   with the NVRTC stub header force-included
##   (`clang++ -x c++ -fsyntax-only -std=c++17 -include <stub> <file>`)
## - the stub (`src/codegen/backends/cuda_nvrtc_stub.h`) defines away
##   the execution-space qualifiers and declares the builtin surface
##
## Shared by the fixtures under `tests/codegen/cuda/cirv2_gate/`,
## imported as `import workspace/crucible/tests/codegen/cuda/cuda_syntax_gate`.

import std/[os, osproc, strutils]

proc cudaStubHeader*(): string =
  ## Returns the absolute NVRTC stub header path. The path is independent
  ## of the working directory and asserted to exist.
  # .../tests/codegen/cuda/cuda_syntax_gate.nim -> .../workspace/crucible
  let crucibleRoot = parentDir(parentDir(parentDir(parentDir(currentSourcePath()))))
  result = crucibleRoot / "src" / "codegen" / "backends" / "cuda_nvrtc_stub.h"
  doAssert fileExists(result),
    "CUDA NVRTC stub header missing: " & result

proc gateCudaSyntax*(name, source: string) =
  ## Type-checks one emitted CUDA-C++ source with Apple clang:
  ## `clang++ -x c++ -fsyntax-only -std=c++17 -include <stub> <file>`.
  ##
  ## Returns: nothing, rejects through `doAssert`.
  ##
  ## Contract:
  ## - artifacts go to `build/tests/cuda_syntax/<name>.cu`
  ## - a non-zero exit fails the test with the compiler output
  let clang = findExe("clang++")
  doAssert clang.len > 0,
    "clang++ not found on PATH, the CUDA syntax fixtures need Apple clang"
  let dir = "build" / "tests" / "cuda_syntax"
  createDir(dir)
  let path = dir / name & ".cu"
  writeFile(path, source)
  let (outp, code) = execCmdEx(
    clang & " -x c++ -fsyntax-only -std=c++17 -include " &
    cudaStubHeader() & " " & path)
  doAssert code == 0,
    "clang++ syntax check failed (exit " & $code & ") for " & name & ":\n" & outp
  echo name, ": clang++ syntax ok"

proc gateCudaSyntaxReject*(name, source: string) =
  ## Rejection check over the same clang invocation: the source must be
  ## ill-formed for the check to pass.
  ##
  ## Returns: nothing, rejects through `doAssert`.
  ##
  ## Contract: a zero exit fails the test with the compiler output.
  let clang = findExe("clang++")
  doAssert clang.len > 0,
    "clang++ not found on PATH, the CUDA syntax fixtures need Apple clang"
  let dir = "build" / "tests" / "cuda_syntax"
  createDir(dir)
  let path = dir / name & ".cu"
  writeFile(path, source)
  let (outp, code) = execCmdEx(
    clang & " -x c++ -fsyntax-only -std=c++17 -include " &
    cudaStubHeader() & " " & path)
  doAssert code != 0,
    "clang++ accepted " & name & ", expected a rejection:\n" & outp
  echo name, ": clang++ rejected as expected"
