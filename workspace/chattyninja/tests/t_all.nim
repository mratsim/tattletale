# Umbrella chattyninja suite, one build, one link, one run over every suite.
#
# Build and run
#   $ nim test_chattyninja
#
# Each suite stays individually runnable, e.g.
#   $ nim c -r t_parse.nim
#
# The allocation-counting suites live in t_all_allocstats.nim, not here.
import t_compose, t_expr, t_parse, t_pull, t_render, t_scratch, t_shape, t_twodriver

echo "t_all: all suites green"
