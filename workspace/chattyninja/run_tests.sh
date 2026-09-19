#!/usr/bin/env bash
# Test gate for the chattyninja template engine: compiles and runs every tests/t_*.nim.
# Builds use -d:release, so the allocstats suites count release-mode allocations. Exits nonzero
# when any suite fails to compile or run. Each build runs through its own binary. Build artifacts
# go to `build/chattyninja/` at the repo root, two levels up from this directory.
#
# Usage: ./run_tests.sh            every suite, each with its variants
#        ./run_tests.sh t_pull     one suite, with that suite's variants

set -u
cd "$(dirname "$0")"

build_dir="../../build/chattyninja"
bin_dir="$build_dir/bin"
cache_dir="$build_dir/nimcache"
mkdir -p "$bin_dir" "$cache_dir"

flags=(--experimental:views --hints:off --warnings:off -d:release --path:src --outdir:"$bin_dir" --nimcache:"$cache_dir")

# Variants: one per table row, `suite extra-defines output-name-suffix`, built whenever the base
# suite runs. Rationale per row:
# - t_twodriver at -d:ChunkSize=7 puts a chunk boundary inside every corpus render; the default
#   4096 never reaches one, so the small-chunk delivery path is only exercised by this build.
# - t_pull and t_scratch at -d:nimAllocStats enable their allocation probes, compiled out
#   without the define, so only this build exercises them.
variants=(
  "t_twodriver -d:ChunkSize=7 chunk7"
  "t_pull -d:nimAllocStats allocstats"
  "t_scratch -d:nimAllocStats allocstats"
)

if [ "$#" -gt 0 ]; then
  suites=()
  for a in "$@"; do
    case "$a" in
      tests/*.nim) suites+=("$a") ;;
      *) suites+=("tests/${a%.nim}.nim") ;;
    esac
  done
else
  suites=(tests/t_*.nim)
fi

# Compiles and runs one build of a suite, printing the FAIL compile, FAIL run or ok line with the
# first lines of the build log or binary output. A failure marks the whole gate failed.
run_build() {
  local f="$1" name="$2" defs="$3" label=""
  [ -n "$defs" ] && label=" ($defs)"
  local log="$bin_dir/$name.log"
  if ! nim c "${flags[@]}" $defs -o:"$bin_dir/$name" "$f" >"$log" 2>&1; then
    echo "FAIL compile  $f$label"
    sed -n '1,25p' "$log"
    status=1
    return
  fi
  if ! "$bin_dir/$name" >"$bin_dir/$name.out" 2>&1; then
    echo "FAIL run      $f$label"
    sed -n '1,40p' "$bin_dir/$name.out"
    status=1
    return
  fi
  echo "ok            $f$label"
  sed -n '1,20p' "$bin_dir/$name.out"
}

status=0
for f in "${suites[@]}"; do
  [ -e "$f" ] || { echo "missing suite: $f"; status=1; continue; }
  name=$(basename "$f" .nim)
  run_build "$f" "$name" ""
  for v in "${variants[@]}"; do
    read -r vbase vdefs vsuffix <<<"$v"
    [ "$vbase" = "$name" ] && run_build "$f" "$name-$vsuffix" "$vdefs"
  done
done

if [ "$status" -eq 0 ]; then
  echo "gate: PASS"
else
  echo "gate: FAIL"
fi
exit "$status"
