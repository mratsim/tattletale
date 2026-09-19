# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Umbrella chattyninja suite, one build, one link, one run over every suite.
#
# Build and run
#   $ nim test_chattyninja
#
# Each suite stays individually runnable, e.g.
#   $ nim c -r t_parse.nim
#
# The allocation-counting build compiles t_corpus.nim alone under
# `-d:nimAllocStats -d:ChunkSize=7`, not here.
import t_corpus, t_expr, t_parse, t_twodriver

echo "t_all: all suites green"
