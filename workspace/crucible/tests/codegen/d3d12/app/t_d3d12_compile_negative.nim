## D3D12 DXIL compile negative paths: missing dxc and unparseable HLSL fail loudly.
##
## Test that compileHlslToDxilViaDxc exits 1 with a legible message on each negative path.
## The parent re-execs itself per leg, failLoud's quit(1) is observable.
## The missing-d3d12.dll leg is Windows-only, t_d3d12_smoke.nim covers it.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/d3d12/app/t_d3d12_compile_negative.nim

import
  workspace/crucible/src/runtime/backends/d3d12_compile,
  std/[os, osproc, strutils]

proc childMode(mode: string) =
  if mode == "no-dxc":
    putEnv("PATH", "")
    discard compileHlslToDxilViaDxc("void main() {}")
  else:
    discard compileHlslToDxilViaDxc("this is not valid HLSL at all")
  echo "[negative] UNREACHABLE: mode " & mode & " did not fail"

if paramCount() >= 1 and paramStr(1).startsWith("--child-"):
  childMode(paramStr(1)[8 .. ^1])
else:
  let self = getAppFilename()
  for mode in ["no-dxc", "bad-hlsl"]:
    let (outp, code) = execCmdEx(self & " --child-" & mode)
    doAssert code == 1, mode & ": expected exit 1, got " & $code & "\n" & outp
    doAssert "exited with error" in outp, mode & ": no legible failure message\n" & outp
  echo "[negative] both negative paths exit 1 with a legible message"
