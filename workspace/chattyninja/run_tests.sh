#!/usr/bin/env bash
# Test gate for the chattyninja template engine: compiles and runs every tests/t_*.nim.
# Exits nonzero on the first compile failure or test failure. Each suite is a plain
# `nim c` build against `src/`, run through its own binary. Build artifacts go to
# `build/chattyninja/` at the repo root, two levels up from this directory.
#
# Usage: ./run_tests.sh            every suite
#        ./run_tests.sh t_shape    one suite

set -u
cd "$(dirname "$0")"

build_dir="../../build/chattyninja"
bin_dir="$build_dir/bin"
cache_dir="$build_dir/nimcache"
mkdir -p "$bin_dir" "$cache_dir"

flags=(--experimental:views --hints:off --warnings:off --path:src --outdir:"$bin_dir" --nimcache:"$cache_dir")

if [ "$#" -gt 0 ]; then
  files=()
  for a in "$@"; do
    case "$a" in
      tests/*.nim) files+=("$a") ;;
      *) files+=("tests/${a%.nim}.nim") ;;
    esac
  done
else
  files=(tests/t_*.nim)
fi

status=0
for f in "${files[@]}"; do
  [ -e "$f" ] || { echo "missing suite: $f"; status=1; continue; }
  name=$(basename "$f" .nim)
  log="$bin_dir/$name.log"
  if ! nim c "${flags[@]}" "$f" >"$log" 2>&1; then
    echo "FAIL compile  $f"
    sed -n '1,25p' "$log"
    status=1
    continue
  fi
  if ! "$bin_dir/$name" >"$bin_dir/$name.out" 2>&1; then
    echo "FAIL run      $f"
    sed -n '1,40p' "$bin_dir/$name.out"
    status=1
    continue
  fi
  echo "ok            $f"
  sed -n '1,20p' "$bin_dir/$name.out"
done

# `t_twodriver` also runs with a tiny chunk size, which puts a chunk boundary inside every
# corpus render; the default 4096 never reaches one. The small-chunk delivery path is only
# exercised by this build, so it ships as a second binary beside the plain one.
run_chunk7=false
if [ "$#" -eq 0 ]; then
  run_chunk7=true
else
  for a in "$@"; do
    [ "$(basename "${a%.nim}")" = "t_twodriver" ] && run_chunk7=true
  done
fi
if [ "$run_chunk7" = true ]; then
  f="tests/t_twodriver.nim"
  name="t_twodriver-chunk7"
  log="$bin_dir/$name.log"
  if ! nim c "${flags[@]}" -d:ChunkSize=7 -o:"$bin_dir/$name" "$f" >"$log" 2>&1; then
    echo "FAIL compile  $f (-d:ChunkSize=7)"
    sed -n '1,25p' "$log"
    status=1
  elif ! "$bin_dir/$name" >"$bin_dir/$name.out" 2>&1; then
    echo "FAIL run      $f (-d:ChunkSize=7)"
    sed -n '1,40p' "$bin_dir/$name.out"
    status=1
  else
    echo "ok            $f (-d:ChunkSize=7)"
    sed -n '1,20p' "$bin_dir/$name.out"
  fi
fi

# `t_pull` also runs with `-d:nimAllocStats`, which enables its allocation probe: pending-piece
# drain calls must allocate 0, and the full pull render must not allocate more than the string
# render. Without the define the probe is compiled out, so only this build exercises it.
run_allocstats=false
if [ "$#" -eq 0 ]; then
  run_allocstats=true
else
  for a in "$@"; do
    [ "$(basename "${a%.nim}")" = "t_pull" ] && run_allocstats=true
  done
fi
if [ "$run_allocstats" = true ]; then
  f="tests/t_pull.nim"
  name="t_pull-allocstats"
  log="$bin_dir/$name.log"
  if ! nim c "${flags[@]}" -d:nimAllocStats -o:"$bin_dir/$name" "$f" >"$log" 2>&1; then
    echo "FAIL compile  $f (-d:nimAllocStats)"
    sed -n '1,25p' "$log"
    status=1
  elif ! "$bin_dir/$name" >"$bin_dir/$name.out" 2>&1; then
    echo "FAIL run      $f (-d:nimAllocStats)"
    sed -n '1,40p' "$bin_dir/$name.out"
    status=1
  else
    echo "ok            $f (-d:nimAllocStats)"
    sed -n '1,20p' "$bin_dir/$name.out"
  fi
fi

# `t_scratch` also runs with `-d:nimAllocStats`, which enables its allocation checks: tojson of
# the tool schema must cost one allocation per call, the full pull render must stay at its
# measured total, and a container emit via scratch must add nothing beyond the loop baseline.
# Without the define the checks are compiled out, so only this build exercises them.
run_allocstats_scratch=false
if [ "$#" -eq 0 ]; then
  run_allocstats_scratch=true
else
  for a in "$@"; do
    [ "$(basename "${a%.nim}")" = "t_scratch" ] && run_allocstats_scratch=true
  done
fi
if [ "$run_allocstats_scratch" = true ]; then
  f="tests/t_scratch.nim"
  name="t_scratch-allocstats"
  log="$bin_dir/$name.log"
  if ! nim c "${flags[@]}" -d:nimAllocStats -o:"$bin_dir/$name" "$f" >"$log" 2>&1; then
    echo "FAIL compile  $f (-d:nimAllocStats)"
    sed -n '1,25p' "$log"
    status=1
  elif ! "$bin_dir/$name" >"$bin_dir/$name.out" 2>&1; then
    echo "FAIL run      $f (-d:nimAllocStats)"
    sed -n '1,40p' "$bin_dir/$name.out"
    status=1
  else
    echo "ok            $f (-d:nimAllocStats)"
    sed -n '1,20p' "$bin_dir/$name.out"
  fi
fi

if [ "$status" -eq 0 ]; then
  echo "gate: PASS"
else
  echo "gate: FAIL"
fi
exit "$status"
