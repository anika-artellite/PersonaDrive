# `scripts/`

Entry points for every step. Run them from the repository root, with the environment from [`../environment/`](../environment/README.md) and your `.env` loaded (`set -a; source .env; set +a`). Each script finds the repository root from its own location.

| Folder | Script | What it does | Env |
|---|---|---|---|
| `setup/` | `setup_carla.sh` | Downloads CARLA 0.9.15 and the additional maps into `software/carla0915`. | — |
| | `carla_exports.sh` | Template of the CARLA / leaderboard environment variables. | — |
| `retrieval/` | `run_stage1_mining.sh [profiles]` | **Stage 1**: frame sentences, SigLIP + BGE-M3 embeddings, triplet mining per style. | `personadrive` |
| | `train_retrieval_head.sh` | **Stage 2**: trains the retrieval head pooled over all profiles. | `personadrive` |
| | `build_style_indices.sh --checkpoint <head>` | **Stage 2b**: one FAISS index per style. | `personadrive` |
| | `fill_normal_index.py --release <dir> --out <dir>` | Completes the released `normal` index (vectors + frame list) from your SimLingo download. | `personadrive` |
| `training/` | `run_icl_local.sh` (→ `generate_icl_examples.py`) | **Stage 3a**: K retrieved context points per training frame (`icl_examples.json.gz`). | `personadrive` |
| | `train_simlingo_seed1.sh` | Upstream SimLingo baseline training (SLURM template). Stage 3 itself is `python simlingo_training/train.py experiment=personadrive_icl_sft`. | `personadrive` |
| `eval/` | `test_one_lb1_route.sh` | One Bench2Drive route in CARLA on one GPU, with debug frames and an MP4. | `personadrive` |
| | `start_eval_simlingo.py` | The 220 Bench2Drive routes on SLURM, per style and K. | `personadrive` |
| | `run_three_profiles.sh`, `run_three_profiles_parallel.sh`, `run_all_routes_profiles.sh` | Several styles or routes locally, sequentially or one GPU per style. | `personadrive` |
| | `start_carla.sh` | A standalone CARLA server for debugging. The evaluators start their own. | — |
| | `aggregate_b2d_metrics.py`, `aggregate_b2d_metrics_k.py` | Driving Score / Success Rate / Efficiency / Comfort / speed / acceleration per style and per K. | `personadrive` |
| | `paper_tables.py` | Runs both aggregators and prints the paper's Tables 1–3 (with the open-loop sweep and timing JSONs) in the paper's layout. | `personadrive` |
| `analysis/` | `style_retrieval_probe.py`, `style_conditioning_eval.py` | Do the per-style indices and the model separate by style? | `personadrive` |
| | `waypoint_accuracy_sweep.py`, `qa_accuracy_sweep.py`, `benchmark_icl_timing.py` | Open-loop ADE/FDE, QA accuracy and latency versus K. `waypoint_accuracy_sweep.py --samples-json openloop_samples_v1.json` evaluates the 200 frames of the paper's sweep. | `personadrive` |
| | `style_speed_stats.py` | Speed statistics of the human style data (paper appendix table), route-matched across the three instructions. | any Python 3 |
| `data_collection/` | `0_run_collect_dataset_slurm.sh` (→ `collect_dataset_slurm.py`) | PDM-lite data agent runs on SLURM. Parallel jobs and partition are read from `max_num_jobs.txt` / `partition.txt`. | `personadrive` |
| `data_processing/` | `run_pipeline.py`, `records_integrating.py`, `merge_measurements.py`, `reorganize_dataset.py` | Raw recordings → VQA/commentary labels → aggregated per-route files; see [`../datasets/raw/`](../datasets/raw/README.md). | `personadrive` |

The full workflow is in the main [README](../README.md#full-pipeline).
