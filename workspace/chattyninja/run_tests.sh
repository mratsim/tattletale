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

if [ "$status" -eq 0 ]; then
  echo "gate: PASS"
else
  echo "gate: FAIL"
fi
exit "$status"
