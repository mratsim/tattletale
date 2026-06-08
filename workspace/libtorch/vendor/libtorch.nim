# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/os,
  workspace/libtorch/vendor/libtorch_config

{.used.}

# #######################################################################
#
#                          C++ Interop
#
# #######################################################################

# Libraries
# -----------------------------------------------------------------------

# Source root (vendor directory containing libtorch/)
const LibTorchSourceRoot = currentSourcePath.parentDir()

when TTT_LIBTORCH_SOURCE == "vendor":
  const LibTorchPath* = LibTorchSourceRoot / "libtorch"
  const LibrariesPath* = LibTorchPath / "lib"
  const HeadersPath* = LibTorchPath / "include"

  static:
    doAssert dirExists(HeadersPath), block:
      "Tattletale is currently configured with -d:TTT_LIBTORCH_SOURCE=" & TTT_LIBTORCH_SOURCE & "\n" &
      "PyTorch headers do not exist at '" & HeadersPath & "\n" &
      "Please run the libtorch downloader via `nim install_libtorch`\n" &
      "or switch to -d:TTT_LIBTORCH_SOURCE=venv"

elif TTT_LIBTORCH_SOURCE == "venv":
  # .venv is at project root: go up from vendor/ → libtorch/ → workspace/ → tattletale/
  const ProjectRoot = LibTorchSourceRoot.parentDir().parentDir().parentDir()
  const VenvSitePackages = ProjectRoot / ".venv" / "lib" / TTT_LIBTORCH_VENV_PYTHON_LIB / "site-packages"
  const LibTorchPath* = VenvSitePackages / "torch"
  const LibrariesPath* = LibTorchPath / "lib"
  const HeadersPath* = LibTorchPath / "include"

  static:
    doAssert dirExists(HeadersPath), block:
      "Tattletale is currently configured with -dTTT_LIBTORCH_SOURCE=" & TTT_LIBTORCH_SOURCE & "\n" &
      "PyTorch headers do not exist at '" & HeadersPath & "\n" &
      "Please double-check your Python version or venv installation\n" &
      "or switch to -dTTT_LIBTORCH_SOURCE=vendor\n" &
      "and run the libtorch downloader via `nim install_libtorch`"

elif TTT_LIBTORCH_SOURCE == "system":
  {.error: "system libtorch mode is not yet implemented".}

else:
  {.error: "Unknown TTT_LIBTORCH_SOURCE: " & TTT_LIBTORCH_SOURCE &
            " (must be 'vendor', 'venv', or 'system')".}

# Torch header sub-path is the same for vendor and venv
const TorchHeadersPath* = HeadersPath / "torch" / "csrc" / "api" / "include"
const TorchHeader* = TorchHeadersPath / "torch" / "torch.h"

# TODO: proper build system on "nimble install" (put libraries in .nimble/bin?)
# if the libPath is not in LD_LIBRARY_PATH
# The libraries won't be loaded at runtime

when defined(windows): # Static linking
  when defined(windows):
    const libSuffix = ".lib"
    const libPrefix = ""
  elif defined(maxosx):
    const libSuffix = ".a" # MacOS
    const libPrefix = "lib"
  else:
    const libSuffix = ".a" # BSD / Linux
  {.link: librariesPath / (libPrefix & "c10" & libSuffix).}
  {.link: librariesPath / (libPrefix & "torch_cpu" & libSuffix).}

  when UseCuda:
    {.link: LibrariesPath / libPrefix & "torch_cuda" & libSuffix.}
else: # Dynamic linking
  # Standard GCC compatible linker
  {.passL: "-L" & LibrariesPath & " -lc10 -ltorch_cpu ".}

  when UseCuda:
    {.passL: " -ltorch_cuda ".}

  when TTT_LIBTORCH_SOURCE == "vendor" or TTT_LIBTORCH_SOURCE == "venv":
    # Link to library in vendor (not for deployment!)
    when defined(macosx):
      {.passL:"-rpath " & LibrariesPath.}
    elif defined(posix):
      {.passL:"-Wl,-rpath," & LibrariesPath.}
  # For "venv": runtime library discovery is handled by the venv environment
  # or LD_LIBRARY_PATH. The .venv's libtorch_python.so already has proper rpath.

  # For "system": system linker paths should handle it

  # Look next to the final binary
  # when defined(macosx):
  #   {.passL:"-rpath @loader_path".}
  # elif defined(posix):
  #   {.passL:"-Wl,-rpath,\\$ORIGIN".}

{.push cdecl.}

# Headers
# -----------------------------------------------------------------------

{.passC: "-I" & HeadersPath.}
{.passC: "-I" & TorchHeadersPath.}

{.push header: TorchHeader.}

when not defined(windows): # couldn't find an equivalent for vcc
  {.passC: "-Wfatal-errors".} # The default "-fmax-errors=3" is unreadable
