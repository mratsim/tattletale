## cutile/compiler.nim
## Layer 2: TileIR Compiler
##
## Two compilation strategies:
##   A. Direct cuModuleLoadData — CUDA 13.1+ driver JIT-compiles TileIR
##   B. tileiras binary        — external process fallback
##
## Strategy B is used when the driver's PTX JIT compiler library doesn't
## include the TileIR backend (pre-13.1 driver, or older driver build).

import
  std/[os, strutils, tables, strformat, osproc, syncio, options],
  bytecode,
  cuda_driver

# ############################################################
# tileiras binary search
# ############################################################

const TileirasSearchPaths* = [
  "/usr/local/cuda/bin/tileiras",
  "/opt/cuda/bin/tileiras",
  "/usr/bin/tileiras",
]

proc findTileirasBinary*(): string =
  let path = getEnv("CUTILE_TILEIRAS", "")
  if path != "" and fileExists(path): return path
  let searchPath = getEnv("PATH", "")
  if searchPath != "":
    for dir in searchPath.split(':'):
      let p = dir / "tileiras"
      if fileExists(p): return p
  for p in TileirasSearchPaths:
    if fileExists(p): return p
  return ""

# ############################################################
# Strategy A: cuModuleLoadData (driver JIT-compiles TileIR)
# ############################################################

proc tryLoadTileIRDirect*(bc: openArray[byte]): Option[CUmodule] =
  ## Try loading TileIR bytecode directly via cuModuleLoadData.
  if bc.len == 0: return none[CUmodule]()
  if bc.len < 12 or bc[0] != 0x7F'u8: return none[CUmodule]()
  var m: CUmodule = cast[CUmodule](nil)
  let res = cuModuleLoadData(m, bc[0].unsafeAddr)
  if res == CUDA_SUCCESS: return some(m)
  return none[CUmodule]()

type
  CompileError* = object of Exception


# ############################################################
# Strategy B: tileiras binary → cubin → cuModuleLoad
# ############################################################

proc compileBytecodeToCubinViaBinary*(
    bcPath: string,
    cubinPath: string,
    gpuArch: string,
    optLevel: int = 3,
    tileirasPath: string
  ) =
  let cmd = fmt"{tileirasPath} --gpu-name {gpuArch} --opt-level {optLevel} -o {cubinPath} {bcPath}"
  echo "  [tileiras] ", cmd
  let output = execCmdEx(cmd)
  if output.exitCode != 0:
    raise newException(CompileError,
      "tileiras failed (exit " & $output.exitCode & "):\n" & output.output)
  if not fileExists(cubinPath):
    raise newException(CompileError,
      "tileiras did not produce: " & cubinPath)

# ############################################################
# Cached compilation
# ############################################################

proc compileBytecodeCached*(
    m: BytecodeModule,
    gpuArch: string = "sm_120",
    cacheDir: string = "/tmp/cutile_cache"
  ): CUmodule =
  ## Compile bytecode to a loadable CUmodule.
  ##
  ## 1. Check cubin cache
  ## 2. Try cuModuleLoadData (CUDA 13.1+ driver JIT-compiles TileIR)
  ## 3. Fall back to tileiras binary

  if not dirExists(cacheDir): createDir(cacheDir)

  let bc = toBytecode(m)
  let bcHash = (block:
    var h = 0'u64
    for b in bc: h = h * 31 + b.uint64
    h)
  let cacheKey = fmt"{bcHash:x}_{gpuArch}"
  let cubinPath = cacheDir / (cacheKey & ".cubin")

  # ── Cache hit ──
  if fileExists(cubinPath):
    echo "  [cutile] Cubin cache hit: ", cubinPath
    return loadModuleFromFile(cubinPath)

  # ── Strategy A: cuModuleLoadData ──
  echo "  [cutile] Trying cuModuleLoadData (TileIR JIT)..."
  let modA = tryLoadTileIRDirect(bc)
  if modA.isSome:
    echo "  [cutile] ✓ TileIR bytecode JIT-compiled by CUDA driver!"
    return modA.get

  # ── Strategy B: tileiras binary ──
  let tileirasBin = findTileirasBinary()
  if tileirasBin == "":
    raise newException(CompileError,
      "TileIR JIT unavailable and tileiras not found.\n" &
      "  Install CUDA 13.1+ Toolkit (includes tileiras in bin/) or\n" &
      "  set environment: export CUTILE_TILEIRAS=/path/to/tileiras")

  let bcPath = cacheDir / (cacheKey & ".bc")
  var s = newStringOfCap(bc.len)
  for b in bc: s.add(chr(b))
  writeFile(bcPath, s)

  compileBytecodeToCubinViaBinary(bcPath, cubinPath, gpuArch,
                                   tileirasPath = tileirasBin)
  echo "  [cutile] ✓ Compiled via tileiras: ", cubinPath

  try: removeFile(bcPath)
  except: discard

  return loadModuleFromFile(cubinPath)
