# Tests

| Test | Needs | What it checks | Time |
|---|---|---|---|
| `static_checks.sh` | `personadrive` env, no GPU, no data | Every first-party `.py` parses and every `.sh` passes `bash -n`. No cluster paths or secret-shaped strings. Never-ship files are not tracked. Every CLI starts (`--help`). `experiment=personadrive_icl_sft` resolves. | ~5 min |
| `smoke_test_eval_structure.py` | `personadrive` env | Runs the real `scripts/eval/start_eval_simlingo.py` with `sbatch`/`squeue` mocked, then checks the output tree (`single_route_k<N>/`), the generated `eval.sh` and K plumbing. | ~1 min |
| `make_mini_database.py` | dataset root | Copies 10 complete routes (1 `driver*` route per style, 2 `lb1_split` routes per fast/slow, 3 PDM-lite) into a mini dataset with the real layout. | ~2 min, ~0.5 GB |
| `run_smoke_pipeline.sh` | 1 GPU (≥ 24 GB), `personadrive` env, mini dataset, `pretrained/RenzKa_simlingo` | End to end on the mini dataset with the documented scripts: `scripts/retrieval/run_stage1_mining.sh` → `train_retrieval_head.sh` → `build_style_indices.sh` → `IclRetriever` → `scripts/training/run_icl_local.sh` → one epoch of `train.py experiment=personadrive_icl_sft` (the released run's config, at the 24 GB setting `batch_size=2 accumulate_grad_batches=3`). Asserts the artefacts of every stage, and that context points are written only for the SFT routes (7) and not the human routes (0). | ~25 min on an A30 |
| `smoke_test_icl_k.py` | 1 GPU, released indices (`PERSONADRIVE_ICL_ROOT`) | Retrieval with the released head and indices for K = 0…4: K neighbours, K images, K + 1 `<image>` tokens. | ~2 min |
| `scripts/eval/test_one_lb1_route.sh` (repo root) | 1 GPU, CARLA, released checkpoint and indices | Closed loop: the agent drives one Bench2Drive route in CARLA. | ~15–30 min |

```bash
set -a; source .env; set +a                       # PERSONADRIVE_* incl. the conda env
export PD_TEST_ROOT=/scratch/$USER/personadrive_test   # anywhere with ~1 GB free (default: tests/_work)

bash tests/static_checks.sh
python tests/smoke_test_eval_structure.py
python tests/make_mini_database.py --out "$PD_TEST_ROOT/mini_database"
bash tests/run_smoke_pipeline.sh                  # on a GPU node, or e.g.:
#   sbatch -p <gpu-partition> --gres=gpu:1 -c 8 --mem=64G -t 02:00:00 --wrap "bash tests/run_smoke_pipeline.sh"
python tests/smoke_test_icl_k.py                  # GPU; uses PERSONADRIVE_ICL_ROOT / _CKPT / _PROFILE
```

`run_smoke_pipeline.sh` prints `PASS`/`FAIL` per stage and writes one log per stage to `$PD_TEST_ROOT/work/logs/`. The mini dataset's generated files (`icl_examples.json.gz`, embedding caches) are recreated on every run. `SKIP_SFT=1` stops after Stage 3a.

The mini dataset is far too small for meaningful numbers. These tests check that every documented command runs and produces correctly shaped outputs, not that it reproduces the paper's metrics.

## Results on the release machine (HPC3, UC Irvine, October 2026)

All tests below were run on this repository before publication, on one NVIDIA A30 (driver 580.82) unless the Environment column names another GPU.

T2, T5 and T5b ran with the former two-environment setup, with Stages 1–2 in a separate Python 3.11 environment. S1–S4 check the single `personadrive` environment.

| ID | Test | Environment | Result |
|---|---|---|---|
| T0 | `static_checks.sh` (final layout) | reference envs | **pass**: 111 `.py` files parse, 39 `.sh` scripts, 0 cluster paths or secrets, all 23 CLIs start, config resolves |
| T6 | `static_checks.sh` on a bare copy of only the files git would commit (26,406 files, 360 MB) | reference envs | **pass**: nothing depends on untracked files |
| T3a | `smoke_test_eval_structure.py` | `personadrive` | **pass**: 10/10 checks |
| T1 | `make_mini_database.py` | — | **pass**: 10 routes, 0.44 GB |
| T2 | `run_smoke_pipeline.sh` (new `scripts/` layout, released SFT config) | reference envs + cluster CUDA module | **pass**: 10/10 stages through `scripts/retrieval/*`, `scripts/training/run_icl_local.sh` and `experiment=personadrive_icl_sft`. Context points written for 7 SFT routes and 0 human routes. SimLingo checkpoint loads 992/992 tensors (0 missing, 0 unexpected); LoRA r 32 / α 64 (17.6 M trainable params), lr 3e-5, batch 2 × 3 accumulation; 793 batches in 9 min 20 s, train loss 0.76, val loss 0.50. |
| T7 | SFT memory | 1× L40S 48 GB / 1× A30 24 GB | Batch 8 per GPU: **OOM on L40S** (needs > 44 GiB); batch 6 per 48 GB GPU fits. On the A30, batch 2 fits (21.4 GB) and batch 4 does not, hence the 24 GB setting `batch_size=2 accumulate_grad_batches=3` used in T2. |
| T5 | `environment/setup_env.sh` (fresh, both envs) | new prefixes | **pass**: all imports. `pip check` identical to the reference envs. flash-attn 2.7.0.post2, nvcc 12.1. |
| T5b | `run_smoke_pipeline.sh` | **freshly built envs**, no cluster modules | **pass**: 14/14 stages (`CUDA_HOME` from the env's own nvcc) |
| T3b | `smoke_test_icl_k.py` with the released head and indices | `personadrive` | **pass**: `normal`, `default`, `fast`, `slow` for K = 0…4 |
| — | Released `personadrive_retrieval_artifacts_v1` extracted fresh + `IclRetriever` on the full dataset | `personadrive` | **pass**: checksums OK. Indices hold 4,559 / 8,960 / 848 / 54,040 vectors. Every retrieved frame path resolves under `PERSONADRIVE_DATA_ROOT`. |
| T4 | `scripts/eval/test_one_lb1_route.sh`: released Stage-3 checkpoint, `fast` index, K = 2, Bench2Drive route 00 (ParkingCutIn) | `personadrive` + CARLA 0.9.15 | **pass**: route completed, Driving Score 100 (no collisions, lane or light infractions), 86 debug frames, MP4 written, ~17 min |
| S1 | Stage 1 and the `default` index from the Hugging Face dataset, in `personadrive` and in the former Stage 1–2 env | `personadrive`, 1× L40S | **identical**: BGE-M3 and SigLIP embeddings match on all 1,455 frames (cosine 1.00000), the same 1,414 anchors get triplets, and the FAISS vectors are identical. |
| S2 | `environment/setup_env.sh` (fresh, then re-run) | new prefix, CPU node | **pass**: 20/20 imports, including FlagEmbedding 1.3.5. `pip check` gives the same 10 notes as the reference env. All 247 pip packages match the reference env except FlagEmbedding (1.3.5 vs 1.4.0). The re-run takes 2 min. |
| S3 | `static_checks.sh` | fresh env from S2 | **pass**: 322 `.py` files parse, 48 `.sh` scripts, 0 cluster paths or secrets, all 23 CLIs start (9 of them for Stages 1–2), config resolves |
| S4 | `run_smoke_pipeline.sh` | fresh env from S2, no cluster modules, 1× L40S | **pass**: 10/10 stages (Stage 1 for 4 styles, retrieval head, FAISS indices, retriever, ICL precompute, 1 epoch of SFT) in 32 min. Context points written for 7 SFT routes and 0 human routes. `CUDA_HOME` comes from the env's own nvcc. |

**Issues found and fixed during these runs:**
- FlagEmbedding 1.4.0 needs a newer transformers than the env's 4.49. It is pinned to 1.3.5, so Stages 1–2 run in the main environment.
- conda `libstdc++` must be on `LD_LIBRARY_PATH`.
- DeepSpeed needs `CUDA_HOME`.
- `nvidia-ml-py`, a DeepSpeed dependency, was missing from the captured requirements.
- `ffmpeg` may be missing from `PATH`, or built without libx264; the script then falls back to imageio-ffmpeg.
- CARLA's renderer hangs inside `srun --overlap` steps. Use a batch job or an interactive shell instead.

All of these are handled by the installer and scripts, or documented in [Verification and troubleshooting](../troubleshooting/README.md).
