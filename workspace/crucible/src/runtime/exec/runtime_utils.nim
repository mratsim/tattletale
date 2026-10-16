## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os

proc sanitizePath*(name: string): string =
  ## Strips directory components, raises on path traversal.
  result = extractFilename(name)
  if result != name or result in [".", ".."]:
    raiseAssert "Path traversal detected — name contains directory separators: \"" & name & "\""

proc getDebugDir(): string =
  ## getTempDir() / "tattletale" / "debug", all debug-only writes.
  result = getTempDir() / "tattletale" / "debug"
  createDir(result)

proc getDebugPath*(name: string): string =
  ## getDebugDir() / sanitizePath(name), parent dirs created.
  let safe = sanitizePath(name)
  result = getDebugDir() / safe
  createDir(result.parentDir())

proc getKernelDir*(segments: varargs[string, sanitizePath]): string =
  ## getTempDir() / "tattletale" / sanitizePath of each arg.
  ## with parent dirs created.
  result = getTempDir() / "tattletale"
  for s in segments:
    result = result / s
  createDir(result)
