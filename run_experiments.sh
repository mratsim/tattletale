#!/usr/bin/env bash
set -euo pipefail
cd /home/beta/Programming/Perso/workspace-tattletale/tattletale

FLAGS_BASE="-d:simdArch=avx_fma --hints:off --warnings:off -d:danger"
NIMCACHE="/tmp/nimcache/exp"
OUTDIR="/tmp"

declare -A EXPERIMENTS
EXPERIMENTS["Baseline (all fast)"]=""
EXPERIMENTS["Exp A: no alignment"]="-d:expNoAlign"
EXPERIMENTS["Exp B: TensorView epilogue"]="-d:expNoRawEpilogue"
EXPERIMENTS["Exp C: fillWith+copySameShape A pack"]="-d:expNoExplicitPackA"
EXPERIMENTS["Exp D: slice-based inner loop"]="-d:expNoExplicitInner"
EXPERIMENTS["Exp E: copySameShape B pack"]="-d:expNoExplicitPackB"
EXPERIMENTS["ALL flags (≈ ref)"]="-d:expNoAlign -d:expNoRawEpilogue -d:expNoExplicitPackA -d:expNoExplicitPackB -d:expNoExplicitInner"

for label in \
  "Baseline (all fast)" \
  "Exp A: no alignment" \
  "Exp B: TensorView epilogue" \
  "Exp C: fillWith+copySameShape A pack" \
  "Exp D: slice-based inner loop" \
  "Exp E: copySameShape B pack" \
  "ALL flags (≈ ref)"; do
  flags="${EXPERIMENTS[$label]}"
  slug=$(echo "$label" | tr ' ' '_' | tr -d ':')
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  $label"
  echo "══════════════════════════════════════════════════════════════"
  echo ""

  nim cpp -r $FLAGS_BASE $flags \
    --nimcache:"${NIMCACHE}_${slug}" -o:"${OUTDIR}/exp_${slug}" \
    workspace/ceramic/benchmark/bench_experiment.nim 2>&1 || true
done
