# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Compile-time discovery of the system zstd library.
# Exports the link flags and the dynlib soname per host OS.
# A missing library is a compile-time error.
#
# - macOS: probe /opt/homebrew/opt/zstd/lib and link
#   with an rpath, so the binary finds the dylib at run
#   time
# - Linux: the plain soname through the loader paths.
#   libzstd is a base distribution package
# - Windows: no system zstd to discover. The vendored
#   build is the default there

import std/os

when defined(macosx):
  const ZstdLibDir* = "/opt/homebrew/opt/zstd/lib"

  static:
    doAssert fileExists(ZstdLibDir / "libzstd.dylib"), block:
      "Tattletale is compiled with -d:TTT_USE_SYSTEM_ZSTD=true\n" &
      "no zstd dylib found in '" & ZstdLibDir & "'\n" &
      "fix: brew install zstd\n" &
      "or build the vendored source: -d:TTT_USE_SYSTEM_ZSTD=false\n" &
      "(requires the submodule, run:\n" &
      "  git submodule update --init workspace/zstd/vendor/zstd)"

  const ZstdLinkFlags* = "-L" & ZstdLibDir & " -lzstd -Wl,-rpath," & ZstdLibDir
  const ZstdDynlib* = "libzstd.1.dylib"

elif defined(linux):
  const ZstdLinkFlags* = "-lzstd"
  const ZstdDynlib* = "libzstd.so.1"

else:
  {.error: "System zstd has no discovery for this OS. " &
           "Build the vendored path: -d:TTT_USE_SYSTEM_ZSTD=false " &
           "after `git submodule update --init " &
           "workspace/zstd/vendor/zstd`".}
