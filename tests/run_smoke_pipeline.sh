#!/bin/bash
# End-to-end smoke test of the PersonaDrive pipeline on a mini dataset.
#
# Runs every training stage from the README, in order, with the same scripts:
#   1   frame sentences, SigLIP + BGE-M3 embeddings, triplet mining
#                                  scripts/retrieval/run_stage1_mining.sh
#   2   retrieval head
#                                  scripts/retrieval/train_retrieval_head.sh
#   2b  per-style FAISS indices    scripts/retrieval/build_style_indices.sh
#   2c  online retriever           simlingo_training/utils/icl_retrieval.py
#   3a  ICL precompute             scripts/training/run_icl_local.sh -> scripts/training/generate_icl_examples.py
#   3b  in-context SFT (1 epoch)   simlingo_training/train.py experiment=personadrive_icl_sft
#
# Needs one GPU (>= 24 GB), the personadrive conda env (environment/README.md),
# and the inputs below. Every stage runs with PYTHON (default: that env's python).
#
#   set -a; source .env; set +a
#   python tests/make_mini_database.py --out "$PD_TEST_ROOT/mini_database"
#   bash tests/run_smoke_pipeline.sh                # on a GPU node, or:
#   sbatch -p <gpu-partition> --gres=gpu:1 -c 8 --mem=64G -t 03:00:00 \
#          --wrap "bash tests/run_smoke_pipeline.sh"
#
# Env: PD_TEST_ROOT (default ${PERSONADRIVE_ROOT}/tests/_work) holds
#      mini_database/ (input) and work/ (outputs, wiped at start).
#      SKIP_SFT=1 stops after stage 3a.
# Exit code 0 = every stage passed.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PD_TEST_ROOT="${PD_TEST_ROOT:-${REPO}/tests/_work}"
MINI="${PD_TEST_ROOT}/mini_database"
WORK="${PD_TEST_ROOT}/work"
LOGS="${WORK}/logs"
PY="${PYTHON:-${PERSONADRIVE_CONDA_ENV:+${PERSONADRIVE_CONDA_ENV}/bin/}python}"

if [ ! -d "${MINI}/simlingo/data/simlingo" ]; then
    echo "ERROR: ${MINI}/simlingo/data/simlingo missing — run tests/make_mini_database.py first" >&2
    exit 2
fi
rm -rf "${WORK}"
mkdir -p "${LOGS}" "${WORK}/bin"

# Call the env's python through a wrapper that puts the env's lib/ first on
# LD_LIBRARY_PATH (what `conda activate` + the env's activation hook does), so
# conda-provided shared libraries (e.g. libstdc++ for sentencepiece) are found.
wrap_env() {   # wrap_env <python> <wrapper-path>
    local py; py="$(command -v "$1")" || { echo "ERROR: $1 not found" >&2; exit 2; }
    local prefix; prefix="$(cd "$(dirname "${py}")/.." && pwd)"
    printf '#!/bin/bash\nexport LD_LIBRARY_PATH="%s/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\nexec "%s" "$@"\n' \
        "${prefix}" "${py}" > "$2"
    chmod +x "$2"
}
wrap_env "${PY}"  "${WORK}/bin/python"
PY="${WORK}/bin/python"
# Generated artefacts inside the mini dataset are recreated by this run.
find "${MINI}/simlingo/data" -name icl_examples.json.gz -delete
rm -rf "${MINI}/simlingo/embeddings"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PERSONADRIVE_ROOT="${REPO}"
export PERSONADRIVE_DATA_ROOT="${MINI}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

PASS=()
FAIL=()
run_step() {   # run_step <id> <description> <command...>
    local id=$1 desc=$2; shift 2
    local log="${LOGS}/${id}.log"
    echo "── [${id}] ${desc}"
    local t0=$SECONDS
    if ( "$@" ) > "${log}" 2>&1; then
        echo "   PASS  ($((SECONDS - t0)) s)  log: ${log}"
        PASS+=("${id}")
    else
        echo "   FAIL  ($((SECONDS - t0)) s)  log: ${log}"
        tail -25 "${log}" | sed 's/^/   | /'
        FAIL+=("${id}")
        summary
        exit 1
    fi
}
need() {       # need <file-or-dir>... : assert artefacts exist and are non-empty
    for f in "$@"; do
        if [ ! -s "$f" ] && [ ! -d "$f" ]; then echo "missing artefact: $f"; return 1; fi
    done
}
summary() {
    echo
    echo "PASSED: ${PASS[*]:-none}"
    echo "FAILED: ${FAIL[*]:-none}"
}

echo "PersonaDrive smoke test   repo=${REPO}"
echo "  mini dataset: ${MINI}"
echo "  outputs:      ${WORK}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
for interp in "${PY}"; do
    if ! timeout 600 "${interp}" -c "import sys, torch; print('  ', sys.executable, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"; then
        echo "ERROR: ${interp} cannot import torch / initialise CUDA within 10 min" >&2; exit 3
    fi
done

cd "${REPO}"
# The retrieval scripts call plain `python`; ${WORK}/bin (the wrapper) goes first on PATH.

# Stage 1: frame sentences + embeddings + triplet mining, per profile
for p in default fast slow normal; do
    run_step "1_${p}" "scripts/retrieval/run_stage1_mining.sh ${p}" bash -c '
        PATH="$1/bin:$PATH" PERSONADRIVE_ICL_ROOT="$1" bash scripts/retrieval/run_stage1_mining.sh "$2" &&
        ls "$1"/processed_routes/"$2"/*_dataset_processed.json &&
        test -s "$1/embeddings_v2/$2/image_embeddings.pt" && test -s "$1/embeddings_v2/$2/frame_ids.json" &&
        test -s "$1/triplets_v2/$2/triplets.jsonl"' _ "${WORK}" "${p}"
done

# Stage 2: retrieval head
run_step 2_head "scripts/retrieval/train_retrieval_head.sh (1 epoch)" bash -c '
    PATH="$1/bin:$PATH" PERSONADRIVE_ICL_ROOT="$1" bash scripts/retrieval/train_retrieval_head.sh --name head --epochs 1 &&
    test -s "$1/checkpoints/head/best_model.pt"' _ "${WORK}"
HEAD="${WORK}/checkpoints/head/best_model.pt"

# Stage 2b: one FAISS index per style, with that head
run_step 2b_faiss "scripts/retrieval/build_style_indices.sh" bash -c '
    PATH="$1/bin:$PATH" PERSONADRIVE_ICL_ROOT="$1" bash scripts/retrieval/build_style_indices.sh --checkpoint "$2" &&
    for n in default fast slow normal; do
        test -s "$1/faiss_index/$n/driving_index.faiss" && test -s "$1/faiss_index/$n/driving_index.meta.pkl" || exit 1
    done' _ "${WORK}" "${HEAD}"

# Stage 2c: the online retriever used by the CARLA agent
cd "${REPO}"
run_step 2c_retriever "IclRetriever + build_icl_prefix on every profile" "${PY}" - "${WORK}" <<'EOF'
import glob, os, sys
from simlingo_training.utils.icl_retrieval import IclRetriever
from simlingo_training.dataloader.dataset_driving import build_icl_prefix
work = sys.argv[1]
img = sorted(glob.glob(os.path.join(os.environ["PERSONADRIVE_DATA_ROOT"], "simlingo/data/simlingo/fast/lb1_split/**/rgb/*.jpg"), recursive=True))[20]
for prof in ("default", "fast", "slow", "normal"):
    r = IclRetriever(checkpoint=f"{work}/checkpoints/head/best_model.pt", profile=prof, root=work, device="cuda", verbose=False)
    ex = r.retrieve(img, {"speed": 5.0, "throttle": 0.5, "steer": 0.0, "brake": 0.0, "command": 4}, k=2)
    assert len(ex) == 2, (prof, len(ex))
    prefix, imgs = build_icl_prefix(ex, num_icl_examples=2, cut_bottom_quarter=True)
    assert prefix is not None and len(imgs) == 2 and prefix.count("<image>") == 2, prof
    print(f"{prof:8s} index={r.index.ntotal:5d}  top-1 score={ex[0]['_score']:.3f}  keys={sorted(ex[0])}")
EOF

# Stage 3a: ICL precompute with the documented driver
run_step 3a_icl "scripts/training/run_icl_local.sh (normal, slow, fast)" bash -c '
    PERSONADRIVE_ICL_ROOT="$1" PERSONADRIVE_ICL_CKPT="$1/checkpoints/head/best_model.pt" \
        PATH="$(dirname "$0"):$PATH" bash scripts/training/run_icl_local.sh &&
    n_sft=$(find "$2/simlingo/data/simlingo/training_1_scenario" "$2/simlingo/data/simlingo/fast/lb1_split" \
                 "$2/simlingo/data/simlingo/slow/lb1_split" -name icl_examples.json.gz | wc -l) &&
    n_human=$(find "$2/simlingo/data/simlingo" -path "*/driver*" -name icl_examples.json.gz | wc -l) &&
    echo "icl files: sft_routes=$n_sft human_routes=$n_human" &&
    [ "$n_sft" -eq 7 ] && [ "$n_human" -eq 0 ]' "$(command -v "${PY}")" "${WORK}" "${MINI}"

if [ "${SKIP_SFT:-0}" = "1" ]; then summary; exit 0; fi

# Stage 3b: in-context SFT, one epoch on the mini dataset
# DeepSpeed checks the CUDA toolkit version on import and needs CUDA_HOME.
# Envs built by environment/setup_env.sh ship nvcc and set it on activation.
if [ -z "${CUDA_HOME:-}" ]; then
    _main_prefix="$(sed -n 's|^export LD_LIBRARY_PATH="\(.*\)/lib.*|\1|p' "${PY}")"
    [ -x "${_main_prefix}/bin/nvcc" ] && export CUDA_HOME="${_main_prefix}"
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "ERROR: CUDA_HOME not set to a CUDA 12 toolkit (needed by DeepSpeed). Use an env built by" >&2
    echo "       environment/setup_env.sh, or load a toolkit (e.g. module load cuda/12.x)." >&2
    summary; exit 1
fi
echo "   CUDA_HOME=${CUDA_HOME}"
# data_path is resolved relative to the repo root, so expose the mini dataset there.
mkdir -p "${REPO}/tests/_work"
ln -sfn "${MINI}" "${REPO}/tests/_work/mini_database_link"
run_step 3b_sft "train.py experiment=personadrive_icl_sft (1 GPU, 1 epoch)" bash -c '
    "$0" simlingo_training/train.py experiment=personadrive_icl_sft gpus=1 \
        data_module.batch_size=2 accumulate_grad_batches=3 data_module.num_workers=4 max_epochs=1 debug=True \
        data_module.base_dataset.data_path=tests/_work/mini_database_link/simlingo \
        hydra.run.dir="$1/sft_run" &&
    ls -d "$1"/sft_run/checkpoints/*.ckpt' "${PY}" "${WORK}"

summary
