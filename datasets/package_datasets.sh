#!/bin/bash
# Build the PersonaDrive release archives (datasets, retrieval artefacts and the
# Stage-3 checkpoint) with file lists and SHA-256 checksums.
#
# Usage:
#   set -a; source .env; set +a
#   bash datasets/package_datasets.sh --list          # show what would be built
#   bash datasets/package_datasets.sh                 # build everything missing
#   bash datasets/package_datasets.sh --only style    # build one archive
#
# Inputs (env):
#   PERSONADRIVE_DATA_ROOT   dataset root containing simlingo/ (first entry used)
#   PERSONADRIVE_ICL_ROOT    Stage 1-2 working dir (embeddings*, triplets*, faiss_index, ...)
#   PERSONADRIVE_CKPT        Stage-3 checkpoint dir (<run>/checkpoints/last.ckpt)
#   PERSONADRIVE_RELEASE_DIR output dir (default: <repo>/datasets/release)
#
# Re-running skips archives whose .sha256 already exists. Archive paths are
# relative, so each one extracts straight into its target (see datasets/README.md).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${PERSONADRIVE_DATA_ROOT:?set PERSONADRIVE_DATA_ROOT}"
DATA_ROOT="${DATA_ROOT%%:*}"
ICL_ROOT="${PERSONADRIVE_ICL_ROOT:-${REPO}/persona_drive_rag}"
CKPT="${PERSONADRIVE_CKPT:-}"
OUT="${PERSONADRIVE_RELEASE_DIR:-${REPO}/datasets/release}"
VERSION="v1"

ONLY=""
LIST_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        *) sed -n '2,18p' "$0"; exit 1 ;;
    esac
done

if command -v pigz >/dev/null 2>&1; then GZ=(-I "pigz -p ${PIGZ_THREADS:-8}"); else GZ=(-z); fi
mkdir -p "${OUT}"

# archive <key> <name> <base-dir> <description> <path-or-glob>...
#   Paths are relative to <base-dir>; globs are expanded inside it.
archive() {
    local key=$1 name=$2 base=$3 desc=$4; shift 4
    [ -n "${ONLY}" ] && [ "${ONLY}" != "${key}" ] && return 0
    local file="${OUT}/${name}"
    local items=()
    local pat
    for pat in "$@"; do
        while IFS= read -r m; do items+=("$m"); done < <(cd "${base}" && compgen -G "${pat}" || true)
    done
    if [ ${#items[@]} -eq 0 ]; then
        echo "!! ${name}: nothing matched under ${base} — skipped"; return 0
    fi
    if [ "${LIST_ONLY}" = "1" ]; then
        local size; size=$(cd "${base}" && du -sch --apparent-size "${items[@]}" | tail -1 | cut -f1)
        printf '%-48s %8s  from %s\n' "${name}" "${size}" "${base}"
        printf '    %s\n' "${items[@]:0:6}"
        [ ${#items[@]} -gt 6 ] && echo "    ... ($((${#items[@]} - 6)) more)"
        return 0
    fi
    if [ -s "${file}.sha256" ]; then
        echo "== ${name}: already built (${file}.sha256 exists)"; return 0
    fi
    echo "== ${name}: packing ${#items[@]} paths from ${base}"
    local t0=$SECONDS
    tar -C "${base}" "${GZ[@]}" -cf "${file}.partial" \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' "${items[@]}"
    mv "${file}.partial" "${file}"
    tar "${GZ[@]}" -tf "${file}" > "${file}.filelist.txt"
    (cd "${OUT}" && sha256sum "${name}" > "${name}.sha256")
    printf '%s\t%s\t%s\t%s\n' "${name}" "$(stat -c %s "${file}")" "$(wc -l < "${file}.filelist.txt")" "${desc}" \
        > "${file}.meta.tsv"
    echo "   done in $((SECONDS - t0)) s: $(du -h --apparent-size "${file}" | cut -f1)"
}

# 1. Human style dataset: 15 drivers x {default=neutral, fast=aggressive, slow=conservative}.
archive style "personadrive_style_dataset_${VERSION}.tar.gz" "${DATA_ROOT}" \
    "Human style dataset (15 drivers x 3 styles; driving data + commentary + VQA)" \
    "simlingo/data/simlingo/default/driver*" "simlingo/data/simlingo/fast/driver*" "simlingo/data/simlingo/slow/driver*" \
    "simlingo/commentary/simlingo/default/driver*" "simlingo/commentary/simlingo/fast/driver*" "simlingo/commentary/simlingo/slow/driver*" \
    "simlingo/drivelm/simlingo/default/driver*" "simlingo/drivelm/simlingo/fast/driver*" "simlingo/drivelm/simlingo/slow/driver*"

# 2. Expert-collected fast/slow routes used in Stage-3 SFT (with their icl_examples.json.gz).
for s in fast slow; do
    archive "expert_${s}" "personadrive_expert_${s}_lb1split_${VERSION}.tar.gz" "${DATA_ROOT}" \
        "Expert ${s} lb1_split routes (350) incl. icl_examples.json.gz" \
        "simlingo/data/simlingo/${s}/lb1_split" "simlingo/commentary/simlingo/${s}/lb1_split" "simlingo/drivelm/simlingo/${s}/lb1_split"
done

# 3. Context points precomputed for the upstream PDM-lite routes (training_1_scenario).
archive icl_pdmlite "personadrive_icl_examples_pdmlite_${VERSION}.tar.gz" "${DATA_ROOT}" \
    "icl_examples.json.gz for the 2,848 PDM-lite training_1_scenario routes" \
    "simlingo/data/simlingo/training_1_scenario/routes_training/*/*/icl_examples.json.gz"

# 4. SigLIP embedding caches written by scripts/training/run_icl_local.sh (--save-emb-dir).
archive siglip_cache "personadrive_siglip_cache_${VERSION}.tar.gz" "${DATA_ROOT}" \
    "Expanded SigLIP image-embedding caches used by Stage 3a" \
    "simlingo/embeddings/*_full"

# 5. Stage 1-2 artefacts: processed sentences, embeddings, triplets, indices, heads.
archive retrieval "personadrive_retrieval_artifacts_${VERSION}.tar.gz" "${ICL_ROOT}" \
    "Stage 1-2 artefacts (processed routes, pivots, embeddings, triplets, FAISS indices, heads)" \
    "processed_routes/default/default_dataset_processed.json" \
    "processed_routes/fast/fast_dataset_processed.json" \
    "processed_routes/slow/slow_dataset_processed.json" \
    "normal_processed.json" "split_jsons/*_trimmed.json" \
    "embeddings_v2/default" "embeddings_v2/fast" "embeddings_v2/slow" \
    "embeddings/normal" "embeddings/fast" "embeddings/slow" \
    "triplets_v2/default/triplets.jsonl" "triplets_v2/fast/triplets.jsonl" "triplets_v2/slow/triplets.jsonl" \
    "triplets/normal/triplets.jsonl" \
    "faiss_index/default" "faiss_index/fast" "faiss_index/slow" "faiss_index/normal" \
    "checkpoints/best_model.pt" "checkpoints/multi_profile/best_model.pt"

# 6. Stage-3 PersonaDrive checkpoint with its hydra config (agent finds .hydra/ above it).
if [ -n "${CKPT}" ] && [ -d "${CKPT}" ]; then
    RUN_DIR="$(cd "${CKPT}/../.." && pwd)"
    STAGE="${OUT}/.stage_ckpt"
    if [ -z "${ONLY}" ] || [ "${ONLY}" = "checkpoint" ]; then
        mkdir -p "${STAGE}/personadrive_icl_sft/checkpoints"
        ln -sfn "${RUN_DIR}/.hydra" "${STAGE}/personadrive_icl_sft/.hydra"
        ln -sfn "${CKPT}" "${STAGE}/personadrive_icl_sft/checkpoints/last.ckpt"
    fi
    # -h follows the two staging symlinks so real files are archived.
    GZ+=(-h)
    archive checkpoint "personadrive_stage3_checkpoint_${VERSION}.tar.gz" "${STAGE}" \
        "Stage-3 PersonaDrive backbone (DeepSpeed dir) + .hydra config; extract into pretrained/" \
        "personadrive_icl_sft"
else
    echo "!! PERSONADRIVE_CKPT not set or missing — Stage-3 checkpoint archive skipped"
fi

[ "${LIST_ONLY}" = "1" ] && exit 0

# Checksums + manifest.
(cd "${OUT}" && cat ./*.sha256 > checksums.sha256 2>/dev/null || true)
{
    echo "# PersonaDrive release archives (${VERSION})"
    echo
    echo "Generated $(date -u +%Y-%m-%dT%H:%MZ) by datasets/package_datasets.sh"
    echo
    echo "| Archive | Size | Files | SHA-256 | Contents |"
    echo "|---|---:|---:|---|---|"
    for m in "${OUT}"/*.meta.tsv; do
        [ -f "$m" ] || continue
        IFS=$'\t' read -r name bytes nfiles desc < "$m"
        sha=$(cut -d' ' -f1 "${OUT}/${name}.sha256")
        printf '| `%s` | %s | %s | `%s` | %s |\n' "${name}" "$(numfmt --to=iec --suffix=B "${bytes}")" "${nfiles}" "${sha}" "${desc}"
    done
} > "${OUT}/MANIFEST.md"
cp "${OUT}/MANIFEST.md" "${REPO}/datasets/MANIFEST.md"
cp "${OUT}/checksums.sha256" "${REPO}/datasets/checksums.sha256"
echo
cat "${OUT}/MANIFEST.md"
