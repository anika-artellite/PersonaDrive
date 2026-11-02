#!/bin/bash
# Stage 3a: precompute the top-K retrieved context points (icl_examples.json.gz)
# for every training route, sequentially: normal -> slow -> fast.
# Needs a GPU (SigLIP) and the Stage 1-2 artefacts under PERSONADRIVE_ICL_ROOT.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Dataset root (first entry of PERSONADRIVE_DATA_ROOT, default
# ${PERSONADRIVE_ROOT}/database); must contain simlingo/data/simlingo/.
_DATA_ROOT="${PERSONADRIVE_DATA_ROOT:-${PERSONADRIVE_ROOT:-${REPO_ROOT}}/database}"
DB_BASE="${_DATA_ROOT%%:*}/simlingo"
if [ ! -d "${DB_BASE}/data/simlingo" ]; then
    echo "ERROR: ${DB_BASE}/data/simlingo not found; set PERSONADRIVE_DATA_ROOT (see datasets/README.md)" >&2
    exit 1
fi
echo "Resolved DB_BASE = ${DB_BASE}"

# Stage 1-2 artefacts (embeddings/, faiss_index/, checkpoints/).
ICL_ROOT="${PERSONADRIVE_ICL_ROOT:-${REPO_ROOT}/persona_drive_rag}"
EMB_SRC="${ICL_ROOT}/embeddings"                                     # embedding source (seed cache)
EMB_OUT="${DB_BASE}/embeddings"                                      # expanded full cache (output)
FAISS_BASE="${ICL_ROOT}/faiss_index"
# The retrieval head that built the released FAISS indices.
CHECKPOINT="${PERSONADRIVE_ICL_CKPT:-${ICL_ROOT}/checkpoints/best_model.pt}"

OVERWRITE_FLAG=""
if [ "${1:-}" = "--overwrite" ]; then
    OVERWRITE_FLAG="--overwrite"
    shift || true
fi

run_profile() {
    local PROFILE=$1
    local DATA_ROOT=$2
    local EXCLUDE="${3:-}"
    echo ""
    echo "============================================================"
    echo "  Profile: ${PROFILE}   $(date)"
    echo "============================================================"

    local SAVE_DIR="${EMB_OUT}/${PROFILE}_full"
    local EMB_FULL="${SAVE_DIR}/image_embeddings_full.pt"
    local IDX_FULL="${SAVE_DIR}/frame_ids_full.json"
    local EMB_PT IDX

    if [ -f "${EMB_FULL}" ]; then
        EMB_PT="${EMB_FULL}"
        IDX="${IDX_FULL}"
        echo "  Using expanded cache: ${SAVE_DIR}"
    else
        EMB_PT="${EMB_SRC}/${PROFILE}/image_embeddings.pt"
        IDX="${EMB_SRC}/${PROFILE}/frame_ids.json"
        echo "  Using original cache (will expand and save to tmp)"
    fi

    local EXCLUDE_FLAG=""
    if [ -n "${EXCLUDE}" ]; then
        EXCLUDE_FLAG="--exclude ${EXCLUDE}"
    fi

    # The seed cache is optional: frames missing from it are encoded live.
    local CACHE_FLAGS=""
    if [ -f "${EMB_PT}" ] && [ -f "${IDX}" ]; then
        CACHE_FLAGS="--emb-pt ${EMB_PT} --emb-index ${IDX}"
    else
        echo "  No embedding cache found for ${PROFILE}; SigLIP runs live for every frame"
    fi

    PYTHONUNBUFFERED=1 python scripts/training/generate_icl_examples.py \
        --checkpoint  "${CHECKPOINT}" \
        --faiss-index "${FAISS_BASE}/${PROFILE}/driving_index.faiss" \
        --faiss-meta  "${FAISS_BASE}/${PROFILE}/driving_index.meta.pkl" \
        --data-root   "${DATA_ROOT}" \
        ${CACHE_FLAGS} \
        --save-emb-dir "${SAVE_DIR}" \
        --device cuda \
        ${EXCLUDE_FLAG} \
        ${OVERWRITE_FLAG}

    echo "  ${PROFILE} done at $(date)"
}

cd "${REPO_ROOT}"

# Only the SFT routes get context points: the PDM-lite routes (normal index)
# and the expert fast/slow lb1_split routes (fast/slow indices). The human
# driver*/ routes under default/, fast/ and slow/ populate the retrieval
# databases but were not part of SFT, so they are excluded here.
run_profile "normal" "${DB_BASE}/data/simlingo" "/data/simlingo/fast/,/data/simlingo/slow/,/data/simlingo/default/"
run_profile "slow"   "${DB_BASE}/data/simlingo/slow/lb1_split"
run_profile "fast"   "${DB_BASE}/data/simlingo/fast/lb1_split"

echo ""
echo "All profiles done at $(date)"