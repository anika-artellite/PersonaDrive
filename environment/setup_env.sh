#!/bin/bash
# Create the PersonaDrive conda environment, captured from the environment that
# trained and evaluated PersonaDrive. It runs every stage (frame sentences,
# SigLIP + BGE-M3 embeddings, triplet mining, retrieval head, FAISS indices,
# ICL precompute, SFT), the CARLA agent and evaluation. Python 3.10,
# torch 2.2.2+cu121, flash-attn 2.7.0.post2, transformers 4.49, FlagEmbedding 1.3.5.
#
# Usage:
#   bash environment/setup_env.sh -n personadrive            # named env
#   bash environment/setup_env.sh -p /path/to/envs/pd        # env at a prefix
#   FLASH_ATTN_FROM_SOURCE=1 bash environment/setup_env.sh -n personadrive
#
# Requirements: conda (or mamba) on PATH, Linux x86_64, an NVIDIA driver for
# CUDA 12.1 (driver >= 530). No system CUDA toolkit is needed: the env ships
# conda-forge cuda-nvcc 12.1 (DeepSpeed needs CUDA_HOME; the activation hook sets it).
# FLASH_ATTN_FROM_SOURCE=1 additionally needs ninja and enough RAM to compile.
#
# Re-running is safe: an existing env is reused and pip skips satisfied pins.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${ENV_DIR}/.." && pwd)"

usage() { sed -n '2,18p' "$0"; exit 1; }

TARGET_FLAG=""
TARGET=""
while getopts "n:p:h" opt; do
    case "${opt}" in
        n) TARGET_FLAG="-n"; TARGET="${OPTARG}" ;;
        p) TARGET_FLAG="-p"; TARGET="${OPTARG}" ;;
        *) usage ;;
    esac
done
[ -n "${TARGET}" ] || usage

CONDA_YAML="${ENV_DIR}/personadrive.yaml"
REQS="${ENV_DIR}/requirements-personadrive.txt"
TORCH_PKGS=("torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2")
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
ENV_VAR="PERSONADRIVE_CONDA_ENV"
CHECK_MODS="torch torchvision flash_attn deepspeed pytorch_lightning transformers peft FlagEmbedding hydra omegaconf faiss flask wandb cv2 carla pygame shapely filterpy"
FLASH_ATTN_VERSION="2.7.0.post2"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
DEEPSPEED_VERSION="0.14.4"
CARLA_VERSION="0.9.15"

CONDA_BIN="$(command -v mamba || command -v conda || true)"
if [ -z "${CONDA_BIN}" ]; then
    echo "ERROR: conda/mamba not found on PATH" >&2
    exit 1
fi

step() { echo; echo "==> [$(date +%H:%M:%S)] $*"; }

# 1. conda layer
if "${CONDA_BIN}" env list | awk '{print $1, $NF}' | grep -qE "(^| )${TARGET}( |$)"; then
    step "Updating conda layer of existing env ${TARGET} from ${CONDA_YAML#${REPO_ROOT}/}"
    "${CONDA_BIN}" env update ${TARGET_FLAG} "${TARGET}" -f "${CONDA_YAML}"
else
    step "Creating conda env ${TARGET} from ${CONDA_YAML#${REPO_ROOT}/}"
    "${CONDA_BIN}" env create ${TARGET_FLAG} "${TARGET}" -f "${CONDA_YAML}"
fi

# Resolve the env's python without needing `conda activate` in this shell.
if [ "${TARGET_FLAG}" = "-p" ]; then
    PY="${TARGET}/bin/python"
else
    PY="$("${CONDA_BIN}" run -n "${TARGET}" python -c 'import sys; print(sys.executable)')"
fi
PIP=("${PY}" -m pip)
"${PY}" -V

# 2. PyTorch
step "Installing ${TORCH_PKGS[*]} (${TORCH_INDEX##*/})"
"${PIP[@]}" install "${TORCH_PKGS[@]}" --index-url "${TORCH_INDEX}"

# 3. Pinned pip layer
# The pins are a full freeze of the working environment, so every dependency is
# listed. The training env was built incrementally and a few of its packages
# declare ranges the freeze does not satisfy (e.g. typer vs click); installing
# with --no-deps reproduces it exactly instead of letting the resolver refuse.
step "Installing pinned requirements (${REQS#${REPO_ROOT}/})"
"${PIP[@]}" install --no-deps -r "${REQS}"

# 4. flash-attention
step "Installing flash-attn ${FLASH_ATTN_VERSION}"
if [ "${FLASH_ATTN_FROM_SOURCE:-0}" = "1" ]; then
    MAX_JOBS="${MAX_JOBS:-8}" "${PIP[@]}" install --no-deps --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"
else
    "${PIP[@]}" install --no-deps "${FLASH_ATTN_WHEEL}"
fi

# 5. DeepSpeed (imports torch at build time; on a GPU machine it needs CUDA_HOME even
# with DS_BUILD_OPS=0, so point it at the env's own nvcc unless a toolkit is already set)
PREFIX="$("${PY}" -c 'import sys; print(sys.prefix)')"
[ -z "${CUDA_HOME:-}" ] && [ -x "${PREFIX}/bin/nvcc" ] && export CUDA_HOME="${PREFIX}"
step "Installing deepspeed ${DEEPSPEED_VERSION}"
DS_BUILD_OPS=0 "${PIP[@]}" install --no-deps --no-build-isolation "deepspeed==${DEEPSPEED_VERSION}"

# 6. CARLA 0.9.15 Python client
step "Installing carla ${CARLA_VERSION}"
"${PIP[@]}" install --no-deps "carla==${CARLA_VERSION}"

step "pip check (informational: the reference env reports the same notes)"
"${PIP[@]}" check || true

# Activation hook: put the env's own lib/ first on LD_LIBRARY_PATH
# Some wheels (e.g. sentencepiece/abseil from conda-forge) need the env's newer
# libstdc++ (GLIBCXX_3.4.30) rather than the system one.
mkdir -p "${PREFIX}/etc/conda/activate.d" "${PREFIX}/etc/conda/deactivate.d"
cat > "${PREFIX}/etc/conda/activate.d/personadrive_ld_path.sh" <<'HOOK'
export _PERSONADRIVE_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# DeepSpeed needs CUDA_HOME; use the env's own nvcc unless a toolkit is already set.
if [ -z "${CUDA_HOME:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export _PERSONADRIVE_SET_CUDA_HOME=1
fi
HOOK
cat > "${PREFIX}/etc/conda/deactivate.d/personadrive_ld_path.sh" <<'HOOK'
export LD_LIBRARY_PATH="${_PERSONADRIVE_OLD_LD_LIBRARY_PATH:-}"
unset _PERSONADRIVE_OLD_LD_LIBRARY_PATH
[ -z "${LD_LIBRARY_PATH}" ] && unset LD_LIBRARY_PATH
if [ -n "${_PERSONADRIVE_SET_CUDA_HOME:-}" ]; then unset CUDA_HOME _PERSONADRIVE_SET_CUDA_HOME; fi
HOOK
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 7. Self-check
step "Self-check"
"${PY}" - ${CHECK_MODS} <<'EOF'
import importlib, sys
failed = []
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m:18s} {getattr(mod, '__version__', '')}")
    except Exception as e:  # report every failure, then exit non-zero
        failed.append(m)
        print(f"  FAIL {m:18s} {type(e).__name__}: {e}")
import torch
print(f"  torch.version.cuda={torch.version.cuda}  cuda_available={torch.cuda.is_available()}")
if failed:
    raise SystemExit(f"import failures: {failed}")
EOF

step "Done. Activate with:  conda activate ${TARGET}"
echo "    and set ${ENV_VAR}=$("${PY}" -c 'import sys; print(sys.prefix)') in your .env"
echo "    (repo: ${REPO_ROOT})"
