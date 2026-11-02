# Environment

PersonaDrive runs in **one conda environment**, `personadrive`. It is a capture of the environment that trained and evaluated PersonaDrive, and it runs:
- Stages 1–2 in `persona_drive_rag/`: frame sentences, SigLIP + BGE-M3 embeddings, triplet mining, retrieval head, FAISS indices, `rag_app.py`;
- Stage 3a (ICL precompute) and Stage 3b (SFT);
- the CARLA agent, evaluation and the open-loop scripts.

| Python | Key packages |
|---|---|
| 3.10 | torch 2.2.2 + cu121, **flash-attn 2.7.0.post2**, DeepSpeed 0.14.4, PyTorch Lightning 2.4.0, transformers 4.49.0, peft 0.13.2, FlagEmbedding 1.3.5, sentence-transformers 5.4.1, hydra-core 1.3.2, faiss-cpu 1.13.2, carla 0.9.15 |

**Stages 1–2.**
- **History:** the paper's retrieval artefacts were built in a separate Python 3.11 environment (torch 2.10, transformers 4.57).
- **Same results here:** the same code in this environment gives identical results on the `default` style. The BGE-M3 and SigLIP embeddings and the FAISS vectors match on all 1,455 frames (cosine 1.00000), and the same 1,414 anchors get triplets.
- **FlagEmbedding pin:** 1.3.5, because 1.4.0 needs a newer transformers. With 1.4.0, Stage 1 fails with `XLMRobertaModel.__init__() got an unexpected keyword argument 'dtype'`.

## Install

```bash
# from the repository root; needs conda or mamba on PATH
bash environment/setup_env.sh -n personadrive            # or: -p /path/to/envs/personadrive
```

The installer runs these steps in order:

1. Creates or updates the conda layer (`personadrive.yaml`), which includes `cuda-nvcc=12.1`.
2. Installs torch from the CUDA 12.1 index.
3. Installs the pinned pip layer (`requirements-personadrive.txt`) with `--no-deps`.
4. Installs flash-attention from the prebuilt GitHub wheel (no CUDA toolkit or compilation needed), DeepSpeed (built against the installed torch), and `carla==0.9.15`.
5. Runs `pip check`, then an import self-check.

It is safe to re-run. It takes about 5–10 minutes with a warm pip cache.

<details>
<summary><b>Manual install</b> (the same commands <code>setup_env.sh</code> runs)</summary>

```bash
conda env create -n personadrive -f environment/personadrive.yaml
conda activate personadrive
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps -r environment/requirements-personadrive.txt
pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
DS_BUILD_OPS=0 pip install --no-deps --no-build-isolation deepspeed==0.14.4
pip install --no-deps carla==0.9.15
# setup_env.sh saves these two lines as a conda activation hook:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" CUDA_HOME="$CONDA_PREFIX"
```

</details>

**Requirements:**
- Linux x86_64.
- An NVIDIA driver supporting CUDA 12.1 (driver ≥ 530).
- An Ampere-or-newer GPU for flash-attention 2 and bf16 (A30, A100, L40S, RTX 30xx+).

The paper used 4× L40S for Stage 3. The end-to-end smoke test runs on one A30.

**Building flash-attention from source:** use `FLASH_ATTN_FROM_SOURCE=1 bash environment/setup_env.sh -n personadrive` if the wheel does not match your system. This needs `nvcc` 12.x and `ninja`; `MAX_JOBS` limits the number of compile jobs.

**Why `--no-deps`:** the requirements are a full `pip list --format=freeze` capture, so every dependency is listed. The environment was built incrementally, and a few packages declare ranges its frozen versions do not satisfy, so the resolver would refuse to install it. A fresh install reports the same ten `pip check` notes as the reference environment: missing dependencies of the Azure, deepdiff, gym, papermill and retry packages, and `typer` vs `click`. None of these packages is used by PersonaDrive.

**Deviations from the reference environment:**
- FlagEmbedding 1.3.5 instead of 1.4.0 (see above).
- The reference env also had a conda `faiss 1.9.0`, but Python imported the pip `faiss-cpu 1.13.2`, which is what is installed here.

Otherwise a fresh install has the same 247 pip packages, at the same versions, as the reference environment.

**CUDA toolkit for DeepSpeed:** on a GPU node, `import deepspeed` checks the installed CUDA toolkit and fails with `MissingCUDAException: CUDA_HOME does not exist` if none is set. The environment therefore includes conda-forge `cuda-nvcc=12.1`, and its activation hook sets `CUDA_HOME=$CONDA_PREFIX` unless you already point `CUDA_HOME` at a toolkit, for example via `module load cuda/12.x` on a cluster. The reference runs used the cluster's CUDA 12.2 module.

**Shared libraries:** some packages, such as conda-forge's sentencepiece and abseil, need the environment's own newer `libstdc++` (`GLIBCXX_3.4.30`) rather than an older system copy. The installer therefore adds a conda activation hook (`etc/conda/activate.d/personadrive_ld_path.sh`) that puts `$CONDA_PREFIX/lib` first on `LD_LIBRARY_PATH`, so `conda activate personadrive` is enough. If you call the environment's `python` without activating it, or use an environment not built by `setup_env.sh`, set the path yourself:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

The symptom of a missing path is `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`. The SLURM scripts in `persona_drive_rag/` export this variable themselves.

## Point the code at the environment

```bash
PERSONADRIVE_CONDA_ENV=$(conda run -n personadrive python -c 'import sys; print(sys.prefix)')
```

Put it in `.env`. The SLURM launchers activate it: the Stage 1–2 scripts (`embed_slurm.sh`, `embed_training_slurm.sh`, `rag_pivot_runner.py`), `scripts/eval/start_eval_simlingo.py` and `scripts/eval/test_one_lb1_route.sh`. `tests/run_smoke_pipeline.sh` and `tests/static_checks.sh` use it too.

## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | Installer. |
| `personadrive.yaml` | Conda layer. |
| `requirements-personadrive.txt` | pip layer. torch, flash-attn, deepspeed and carla are installed separately by the installer. |
| `upstream_simlingo_environment.yaml` | The upstream SimLingo export (python 3.8, torch 2.2.0). Reference only; PersonaDrive was not run with it. |
