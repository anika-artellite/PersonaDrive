#!/bin/bash
# Stage 2 - train the retrieval head (frozen SigLIP 768-d + ControlEncoder 128-d
# -> ProjectionMLP 896 -> 1024 -> L2-norm), pooled over every style profile.
#
# Usage (personadrive env, 1 GPU):
#   bash scripts/retrieval/train_retrieval_head.sh
#   bash scripts/retrieval/train_retrieval_head.sh --name my_head --epochs 5   # extra args go to embed_training.py
#
# Training: offline weighted triplet loss, margin 0.3, hard-negative weight 2.0,
# AdamW (weight decay 0.01), lr 1e-4, batch 128, 20 epochs.
#
# Inputs  (from Stage 1, or from the retrieval-artefacts archive), per profile p:
#   ${PERSONADRIVE_ICL_ROOT}/triplets_v2/<p>/triplets.jsonl + embeddings_v2/<p>/{image_embeddings.pt,frame_ids.json}
#   (the archive's PDM-lite profile uses triplets/normal + embeddings/normal; both layouts work)
# Output  ${PERSONADRIVE_ICL_ROOT}/checkpoints/<name>/best_model.pt   (default name: head_paper)
# PERSONADRIVE_ICL_ROOT defaults to persona_drive_rag/. Existing outputs are kept unless FORCE=1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${PERSONADRIVE_ICL_ROOT:-${REPO}/persona_drive_rag}"
NAME="head_paper"
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --name)   NAME="$2"; shift 2 ;;
        *)        PASS+=("$1"); shift ;;
    esac
done

TRAIN_ARGS=(--no_hard_mining --margin 0.3 --hard_neg_weight 2.0 --optimizer adamw
            --weight_decay 0.01 --lr 1e-4 --batch_size 128 --epochs 20)

PROFILE_ARGS=()
for p in default fast slow normal; do
    for layout in triplets_v2/${p}:embeddings_v2/${p} triplets/${p}:embeddings/${p}; do
        tri="${OUT}/${layout%%:*}/triplets.jsonl"; emb="${OUT}/${layout##*:}"
        if [ -s "${tri}" ] && [ -s "${emb}/image_embeddings.pt" ] && [ -s "${emb}/frame_ids.json" ]; then
            PROFILE_ARGS+=(--profile "${p}" "${tri}" "${emb}/image_embeddings.pt" "${emb}/frame_ids.json")
            echo "== profile ${p}: ${tri} ($(wc -l < "${tri}") anchors)"
            break
        fi
    done
done
if [ ${#PROFILE_ARGS[@]} -eq 0 ]; then
    echo "ERROR: no triplets found under ${OUT}; run scripts/retrieval/run_stage1_mining.sh first" >&2
    exit 1
fi

CKPT_DIR="${OUT}/checkpoints/${NAME}"
if [ -s "${CKPT_DIR}/best_model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: ${CKPT_DIR}/best_model.pt exists (FORCE=1 to overwrite, or --name another_name)" >&2
    exit 1
fi
mkdir -p "${CKPT_DIR}"
cd "${OUT}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
echo "== retrieval head -> ${CKPT_DIR}"
python "${REPO}/persona_drive_rag/embed_training.py" "${PROFILE_ARGS[@]}" --checkpoint_dir "${CKPT_DIR}" \
    --device cuda "${TRAIN_ARGS[@]}" "${PASS[@]}"

echo
echo "Retrieval head: ${CKPT_DIR}/best_model.pt"
echo "Next: bash scripts/retrieval/build_style_indices.sh --checkpoint ${CKPT_DIR}/best_model.pt"
echo "      and use the same head downstream: export PERSONADRIVE_ICL_CKPT=${CKPT_DIR}/best_model.pt"
