#!/bin/bash
# ============================================================================
# embed_slurm.sh - Submit SLURM array job for distributed embedding
# ============================================================================
#
# Usage:
#   bash embed_slurm.sh [JSON_DIR] [SHARDS_DIR] [MERGED_DIR]
#
# Defaults:
#   JSON_DIR    = processed_routes
#   SHARDS_DIR  = embeddings_shards
#   MERGED_DIR  = embeddings
#
# This script:
#   1. Discovers all *.json files in JSON_DIR
#   2. Writes a file list to SHARDS_DIR/json_files.txt
#   3. Submits a SLURM array job (one task per JSON file) for embedding
#   4. Submits a merge job that depends on all array tasks completing
#
# After both jobs finish, MERGED_DIR/ contains the unified embeddings
# ready for the main pipeline (embed_mine.py).
# ============================================================================

set -euo pipefail

# Args / defaults
JSON_DIR="${1:-processed_routes}"
SHARDS_DIR="${2:-embeddings_shards}"
MERGED_DIR="${3:-embeddings}"

DATA_ROOT="${PERSONADRIVE_DATA_ROOT:-/path/to/database}"
DATA_ROOT="${DATA_ROOT%%:*}/simlingo"
STRIP_PREFIX="database/simlingo"  # set to "" to disable prefix stripping
CONDA_ENV="${PERSONADRIVE_CONDA_ENV:-/path/to/conda/envs/personadrive}"
ACCOUNT_GPU="${PERSONADRIVE_SLURM_ACCOUNT_GPU:-YOUR_SLURM_ACCOUNT_GPU}"   # GPU partition account (embedding jobs)
ACCOUNT_CPU="${PERSONADRIVE_SLURM_ACCOUNT:-YOUR_SLURM_ACCOUNT}"       # CPU partition account (merge job)
PARTITION="gpu"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Discover JSON files
mkdir -p "${SHARDS_DIR}"

mapfile -t JSON_FILES < <(find "${JSON_DIR}" -maxdepth 1 -name "*.json" -type f | sort)
N_FILES=${#JSON_FILES[@]}

if [ "${N_FILES}" -eq 0 ]; then
    echo "ERROR: No JSON files found in ${JSON_DIR}"
    exit 1
fi

echo "Found ${N_FILES} JSON files in ${JSON_DIR}"

# Write file list for the array job to reference by $SLURM_ARRAY_TASK_ID
FILELIST="${SHARDS_DIR}/json_files.txt"
printf '%s\n' "${JSON_FILES[@]}" > "${FILELIST}"
echo "File list written to ${FILELIST}"

# Submit array job
ARRAY_SPEC="0-$((N_FILES - 1))"

# Throttle: max 8 concurrent tasks to avoid overwhelming the GPU queue.
# Adjust %N if your cluster allows more or fewer simultaneous GPU jobs.
ARRAY_THROTTLE=8

EMBED_JOB_ID=$(sbatch \
    --parsable \
    --account="${ACCOUNT_GPU}" \
    --partition=gpu \
    --job-name="embed_shard" \
    --array="${ARRAY_SPEC}%${ARRAY_THROTTLE}" \
    --gres=gpu:A30:1 \
    --mem=24G \
    --time=00:20:00 \
    --output="${SHARDS_DIR}/slurm_embed_%A_%a.out" \
    --error="${SHARDS_DIR}/slurm_embed_%A_%a.err" \
    --wrap="
#!/bin/bash

# Activate conda BEFORE strict mode - bashrc has unguarded variables
source ~/.bashrc
conda activate ${CONDA_ENV}

set -euo pipefail

# conda's libstdc++ (GLIBCXX_3.4.30) before the older system copy in /lib64
export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"

echo \"Python: \$(which python)\"
echo \"Python version: \$(python --version)\"
echo \"LD_LIBRARY_PATH: \${LD_LIBRARY_PATH}\"

# Read this task's JSON file
JSON_FILE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${FILELIST})
SHARD_NAME=\$(basename \"\${JSON_FILE}\" .json)
SHARD_OUT=\"${SHARDS_DIR}/\${SHARD_NAME}\"

echo \"Task \${SLURM_ARRAY_TASK_ID}: \${JSON_FILE} → \${SHARD_OUT}\"
echo \"GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)\"

python ${SCRIPT_DIR}/embed_single.py \\
    --json_file \"\${JSON_FILE}\" \\
    --output_dir \"\${SHARD_OUT}\" \\
    --data_root \"${DATA_ROOT}\" \\
    --device cuda \\
    --batch_size_img 512 \\
    --batch_size_txt 512 \\
    ${STRIP_PREFIX:+--strip_prefix \"${STRIP_PREFIX}\"}

echo \"Shard \${SHARD_NAME} done\"
")

echo "Submitted embedding array job: ${EMBED_JOB_ID}  (${N_FILES} tasks, max ${ARRAY_THROTTLE} concurrent)"

# Submit merge job (depends on all embedding tasks)
MERGE_JOB_ID=$(sbatch \
    --parsable \
    --account="${ACCOUNT_CPU}" \
    --partition="standard" \
    --job-name="merge_embeddings" \
    --dependency="afterok:${EMBED_JOB_ID}" \
    --mem=24G \
    --time=00:20:00 \
    --output="${SHARDS_DIR}/slurm_merge_%j.out" \
    --error="${SHARDS_DIR}/slurm_merge_%j.err" \
    --wrap="
#!/bin/bash

source ~/.bashrc
conda activate ${CONDA_ENV}

set -euo pipefail

export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"

echo \"Python: \$(which python)\"
echo \"Merging ${N_FILES} shards from ${SHARDS_DIR} → ${MERGED_DIR}\"

python ${SCRIPT_DIR}/embed_merge.py \\
    --shards_dir \"${SHARDS_DIR}\" \\
    --output_dir \"${MERGED_DIR}\" \\
    --shuffle \\
    --verify

echo \"Merge complete — ready to run main pipeline\"
")

echo "Submitted merge job: ${MERGE_JOB_ID}  (depends on ${EMBED_JOB_ID})"
echo ""
echo "Pipeline:"
echo "  1. ${N_FILES} embedding jobs  →  ${SHARDS_DIR}/<shard_name>/"
echo "  2. Merge job                  →  ${MERGED_DIR}/"
echo "  3. Then run:  python embed_mine.py --dataset-json <processed.json> --emb-dir ${MERGED_DIR} --triplets-dir <out>"
echo ""
echo "Monitor: squeue -u \$USER"