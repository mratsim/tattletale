# Tattletale - raw-libtorch
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

packageName = "raw_libtorch"
version = "0.1.0"
author = "Mamy Ratsimbazafy"
description = "Raw libtorch bindings for Nim"
license = "MIT or Apache License 2.0"
installDirs = @["vendor"]
backend = "cpp"

# Dependencies
# ---------------------------------------------------------

requires "nim >= 2.2.0"
requires "zip"

# Tasks
# ---------------------------------------------------------

import os

task install_libtorch, "Download and install libtorch":
  const libInstaller = "vendor" / "libtorch_installer.nim"
  selfExec("cpp -r --skipParentCfg:on " & libInstaller)

task test, "Execute TorchTensor tests ":
  withDir "tests":
    for fstr in listFiles("."):
      if fstr.endsWith(".nim") and fstr.startsWith("." / "test_"):
        echo "Running ", fstr
        selfExec("cpp -r -d:release --nimcache:nimcache --outdir:build " & fstr)
