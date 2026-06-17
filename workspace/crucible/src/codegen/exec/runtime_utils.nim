## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os

proc sanitizePath*(name: string): string =
  ## Strip all directory components, returning only the final filename component.
  ## Raises if the result differs from the input (path traversal attempt detected).
  result = extractFilename(name)
  if result != name or result in [".", ".."]:
    raiseAssert "Path traversal detected — name contains directory separators: \"" & name & "\""

proc getDebugDir*(): string =
  ## Returns getTempDir() / "tattletale" / "debug" — all debug-only writes go here.
  result = getTempDir() / "tattletale" / "debug"
  createDir(result)

proc getDebugPath*(name: string): string =
  ## Returns getDebugDir() / sanitizePath(name) with parent dirs created.
  ## Convenience for debug-only file writes.
  let safe = sanitizePath(name)
  result = getDebugDir() / safe
  createDir(result.parentDir())

proc getKernelDir*(segments: varargs[string, sanitizePath]): string =
  ## Returns getTempDir() / "tattletale" / sanitizePath(s0) / sanitizePath(s1) / ...
  ## with parent dirs created.
  result = getTempDir() / "tattletale"
  for s in segments:
    result = result / s
  createDir(result)
