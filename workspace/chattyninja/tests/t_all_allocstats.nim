# Allocation-counting chattyninja binary, the two suites whose alloc tests
# compile in under `-d:nimAllocStats`, built together with `-d:ChunkSize=7` so
# the counts are taken at the boundary-poking chunk size in the same build.
#
# Build and run
#   $ nim test_chattyninja
#
# Each suite stays individually runnable, e.g.
#   $ nim cpp -r -d:release -d:nimAllocStats -d:ChunkSize=7 t_pull.nim
import t_pull, t_scratch

echo "t_all_allocstats: allocation suites green"
