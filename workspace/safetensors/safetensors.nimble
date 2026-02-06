# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

packageName = "safetensors"
version = "0.1.0"
author = "Mamy Ratsimbazafy"
description = "Safetensors implementation for Nim"
license = "MIT or Apache License 2.0"
installDirs = @["src"]

# Dependencies
# ---------------------------------------------------------

requires "nim >= 2.2.0"
requires "jsony"
requires "stew"