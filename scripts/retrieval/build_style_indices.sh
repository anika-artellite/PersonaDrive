#!/bin/bash
# Stage 2b - embed every frame of each style database with a retrieval head and
# build one FAISS index per style (IndexFlatIP over L2-normalised embeddings).
#
# Usage (personadrive env, 1 GPU):
#   bash scripts/retrieval/build_style_indices.sh --checkpoint persona_drive_rag/checkpoints/head_paper/best_model.pt
#   bash scripts/retrieval/build_style_indices.sh --checkpoint <ckpt> fast slow     # a subset
#
# Inputs  per profile p, under ${PERSONADRIVE_ICL_ROOT} (default persona_drive_rag/):
#   embeddings_v2/<p>/{image_embeddings.pt,frame_ids.json} + processed_routes/<p>/*_dataset_processed.json
#   (the archive's PDM-lite profile: triplets/normal + embeddings/normal + normal_processed.json)
# Output  ${PERSONADRIVE_ICL_ROOT}/faiss_index/<p>/driving_index.{faiss,meta.pkl,info.json}
#         (or --output-dir DIR). Existing profile indices are kept unless FORCE=1.
#
# The agent, scripts/training/run_icl_local.sh and the SFT data must all use the SAME head and
# indices: after building, point PERSONADRIVE_ICL_CKPT at this checkpoint.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${PERSONADRIVE_ICL_ROOT:-${REPO}/persona_drive_rag}"
CKPT="${PERSONADRIVE_ICL_CKPT:-}"
INDEX_DIR=""
PROFILES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --output-dir) INDEX_DIR="$2"; shift 2 ;;
        *) PROFILES+=("$1"); shift ;;
    esac
done
[ ${#PROFILES[@]} -eq 0 ] && PROFILES=(default fast slow normal)
INDEX_DIR="${INDEX_DIR:-${OUT}/faiss_index}"
if [ -z "${CKPT}" ] || [ ! -s "${CKPT}" ]; then
    echo "ERROR: --checkpoint <retrieval head .pt> required (got '${CKPT}')" >&2
    exit 1
fi

ARGS=()
for p in "${PROFILES[@]}"; do
    if [ -d "${INDEX_DIR}/${p}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== ${p}: ${INDEX_DIR}/${p} exists, skipping (FORCE=1 to rebuild)"; continue
    fi
    processed=$(ls "${OUT}/processed_routes/${p}/"*_dataset_processed.json 2>/dev/null | head -1 || true)
    if [ -s "${OUT}/embeddings_v2/${p}/image_embeddings.pt" ] && [ -n "${processed}" ]; then
        ARGS+=(--profile "${p}" none "${OUT}/embeddings_v2/${p}/image_embeddings.pt" \
               "${OUT}/embeddings_v2/${p}/frame_ids.json" "${processed}")
    elif [ "${p}" = "normal" ] && [ -s "${OUT}/triplets/normal/triplets.jsonl" ]; then
        # layout of the released PDM-lite index: unique triplet anchors only
        ARGS+=(--profile normal "${OUT}/triplets/normal/triplets.jsonl" "${OUT}/embeddings/normal/image_embeddings.pt" \
               "${OUT}/embeddings/normal/frame_ids.json" "${OUT}/normal_processed.json")
    else
        echo "== ${p}: no Stage-1 outputs under ${OUT}, skipping"; continue
    fi
    echo "== ${p}: will build ${INDEX_DIR}/${p}"
done
if [ ${#ARGS[@]} -eq 0 ]; then
    echo "Nothing to build."; exit 0
fi

cd "${OUT}"
python "${REPO}/persona_drive_rag/build_faiss_index.py" --checkpoint "${CKPT}" --output_dir "${INDEX_DIR}" \
    --device cuda "${ARGS[@]}"
echo
echo "Indices in ${INDEX_DIR}. Use them with: export PERSONADRIVE_ICL_ROOT=${OUT} PERSONADRIVE_ICL_CKPT=${CKPT}"
