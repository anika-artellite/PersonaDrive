#!/bin/bash
# Stage 1 - frame sentences, SigLIP + BGE-M3 embeddings and triplet mining,
# one style profile at a time.
#
# Usage (personadrive env, 1 GPU):
#   bash scripts/retrieval/run_stage1_mining.sh                 # default fast slow
#   bash scripts/retrieval/run_stage1_mining.sh fast slow       # a subset
#   bash scripts/retrieval/run_stage1_mining.sh normal          # PDM-lite (training_1_scenario)
#
# Reads   ${PERSONADRIVE_DATA_ROOT}/simlingo/{data,commentary,drivelm}/simlingo/<profile>/...
# Writes  ${PERSONADRIVE_ICL_ROOT}/processed_routes/<profile>/   frame sentences
#         ${PERSONADRIVE_ICL_ROOT}/embeddings_v2/<profile>/       SigLIP + BGE-M3 embeddings
#         ${PERSONADRIVE_ICL_ROOT}/triplets_v2/<profile>/         triplets.jsonl
# PERSONADRIVE_ICL_ROOT defaults to persona_drive_rag/. Profiles whose triplets
# already exist are skipped unless FORCE=1.
#
# Settings are those of the released artefacts: stride 5, combined similarity
# 0.5/0.5; `default` (small) was mined with pools 200/200 and hard percentile 50,
# fast/slow with the embed_mine.py defaults. The released `normal` index used
# per-command PDM-lite pivot frames (persona_drive_rag/rag_pivot_runner.py); here
# `normal` is mined from training_1_scenario with stride 5 instead.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${PERSONADRIVE_DATA_ROOT:?set PERSONADRIVE_DATA_ROOT (see datasets/README.md)}"
DB="${DATA_ROOT%%:*}/simlingo"
OUT="${PERSONADRIVE_ICL_ROOT:-${REPO}/persona_drive_rag}"
PROFILES=("$@")
[ ${#PROFILES[@]} -eq 0 ] && PROFILES=(default fast slow)

mkdir -p "${OUT}"
cd "${OUT}"    # embed_mine.py writes pipeline.log into the working directory

for p in "${PROFILES[@]}"; do
    case "${p}" in
        default) src=default; extra=(--top-k-pool 200 --bottom-k-pool 200 --hard-percentile 50) ;;
        fast|slow) src=${p}; extra=() ;;
        normal) src=training_1_scenario; extra=(--top-k-pool 200 --bottom-k-pool 200 --hard-percentile 50) ;;
        *) echo "unknown profile '${p}' (use default, fast, slow, normal)" >&2; exit 2 ;;
    esac
    if [ -s "${OUT}/triplets_v2/${p}/triplets.jsonl" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== ${p}: ${OUT}/triplets_v2/${p}/triplets.jsonl exists, skipping (FORCE=1 to redo)"
        continue
    fi
    echo "== ${p}: frame sentences (stride 5) from ${DB}/data/simlingo/${src}"
    python "${REPO}/persona_drive_rag/rag_data_processor_v2.py" --data-root "${DB}" --profile "${src}" \
        --frame-step 5 --output-dir "${OUT}/processed_routes/${p}"
    echo "== ${p}: embeddings + triplet mining"
    python "${REPO}/persona_drive_rag/embed_mine.py" \
        --dataset-json "${OUT}/processed_routes/${p}/${src}_dataset_processed.json" \
        --emb-dir "${OUT}/embeddings_v2/${p}" --triplets-dir "${OUT}/triplets_v2/${p}" "${extra[@]}"
    echo "== ${p}: $(wc -l < "${OUT}/triplets_v2/${p}/triplets.jsonl") anchors with triplets"
done
echo "Stage 1 done. Next: bash scripts/retrieval/train_retrieval_head.sh"
