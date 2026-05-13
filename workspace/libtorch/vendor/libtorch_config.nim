# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Configuration switches
# ----------------------------------------------------------------
#
# TTT_LIBTORCH_SOURCE controls where libtorch comes from:
#   - "vendor" (default): use workspace/libtorch/vendor/libtorch/
#   - "venv": use .venv/lib/pythonX.Y/site-packages/torch/ (for development)
#   - "system": link -ltorch -lc10 from the system (for deployment, stub)
#
# TTT_LIBTORCH_VENV_PYTHON_LIB selects the Python lib directory inside .venv
# (e.g. "python3.14", "python3.13"). Defaults to python3.14.
#

const UseCuda* = defined(cuda)

const TTT_LIBTORCH_SOURCE* {.strdefine.} = "vendor"

const TTT_LIBTORCH_VENV_PYTHON_LIB* {.strdefine.} = "python3.14"
