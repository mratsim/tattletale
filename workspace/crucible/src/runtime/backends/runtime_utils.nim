#Tattletale
#Copyright (c) 2026 Mamy André-Ratsimbazafy
#Licensed and distributed under either of
#  * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#  * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
#at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os

proc sanitizePath*(name: string): string =
  ## Strips directory components, raises on path traversal.
  result = extractFilename(name)
  if result != name or result in [".", ".."]:
    raiseAssert "Path traversal detected — name contains directory separators: \"" & name & "\""

proc getKernelDir*(segments: varargs[string, sanitizePath]): string =
  result = getTempDir() / "tattletale"
  for s in segments:
    result = result / s
  createDir(result)

template ceil_div*(a, b: SomeInteger): SomeInteger =
  (a + b - 1) div b

template round_multiple_up*(a, b: SomeInteger): SomeInteger =
  ceil_div(a, b) * b
