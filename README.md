<div align="center">

<a href="https://faculty.sites.uci.edu/elmalaki/"><img src="assets/logo_lab.png" height="110" alt="Pervasive Autonomy Lab"></a>

# PersonaDrive

### Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation

[Mahmoud Srewa](https://mosrewa.github.io/) · [Praneetsai Vasu Iddamsetty](https://www.linkedin.com/in/praneet-iddamsetty/) · [Mohammad Abdullah Al Faruque](https://aicps.eng.uci.edu/) · [Salma Elmalaki](https://faculty.sites.uci.edu/elmalaki/)

University of California, Irvine &nbsp;·&nbsp; **WACV 2027**

[![arXiv](https://img.shields.io/badge/arXiv-2606.12616-b31b1b.svg)](https://arxiv.org/abs/2606.12616)
[![Project page](https://img.shields.io/badge/Project-Page-2ea44f.svg)](https://pervasiveautonomylab.github.io/personadrive-landpage/)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/PervasiveAutonomyLab/PersonaDrive)
[![Demo video](https://img.shields.io/badge/Video-Demo-red.svg)](https://pervasiveautonomylab.github.io/personadrive-landpage/static/videos/demo_full.mp4)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-8A2BE2.svg)](https://huggingface.co/datasets/PervasiveAutonomyLab/PersonaDrive)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](environment/README.md)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<a href="https://pervasiveautonomylab.github.io/personadrive-landpage/#demo"><img src="assets/media/hero_teaser.webp" width="88%" alt="PersonaDrive teaser"></a>

**A driving agent that drives carefully or aggressively, based on how real people drove.**<br>
<sub>Click the animation for the demo; full-quality videos are on the <a href="https://pervasiveautonomylab.github.io/personadrive-landpage/">project page</a>.</sub>

</div>

---

PersonaDrive conditions a vision-language-action (VLA) driving agent on **demonstrations retrieved from people who were asked to drive in a given style**:
- **Dataset.** 15 participants drove CARLA Leaderboard routes on a driver-in-the-loop rig under *aggressive*, *neutral* and *conservative* instructions.
- **Retrieval.** A lightweight retrieval head (about 2 M trained parameters) fuses what the car *sees* with what it is *doing*. It combines the front-camera image, encoded by a frozen SigLIP model, with the vehicle's sensor readings (speed, throttle, steering, brake) and its navigation command into one 1024-d embedding. Searching the selected style's database with this embedding returns moments that match both the scene and the vehicle's state, so the agent sees how that style handled a comparable situation.
- **Prediction.** The VLA backbone reads those moments as in-context demonstrations before predicting its own waypoints.

**Changing the style changes which database is searched; one set of weights serves every style.**

<table>
<tr>
<td align="center" width="33%"><b>A controlled style dataset</b><br><sub>15 drivers × 3 instructed styles, paired at the route level so style is decoupled from driver identity: <b>29,777 frames</b>, 55 routes, 21 scenario types.</sub></td>
<td align="center" width="33%"><b>Style from retrieved humans</b><br><sub>Triplet mining → retrieval head over per-style databases → in-context fine-tuning. No style tokens, proxy rewards or per-style models; switching style is a <b>database swap</b>.</sub></td>
<td align="center" width="33%"><b>Strong closed-loop driving</b><br><sub>Bench2Drive Driving Score <b>88.95</b> without style (+4.6 % over SimLingo). With style: <b>87.55–89.62</b> in every style, while speed and acceleration rise <b>19.8 %</b> and <b>24.9 %</b> from conservative to aggressive.</sub></td>
</tr>
</table>

Demos, how the method works, the dataset and the results are in **[More information](more_info/README.md)**.

## Contents

1. [Installation](#installation)
2. [Data and released artefacts](#data-and-released-artefacts)
3. [Quick start: evaluate the released model](#quick-start-evaluate-the-released-model)
4. [Full pipeline](#full-pipeline)
5. [Evaluation and the paper's tables](#evaluation-and-the-papers-tables)
6. [Repository layout](#repository-layout)
7. [What is (not) in this repository](#what-is-not-in-this-repository)
8. [Citation](#citation)

Elsewhere: [More information](more_info/README.md) (demos, method, dataset, results) · [Verification and troubleshooting](troubleshooting/README.md)

## Installation

**1. Clone and set up CARLA 0.9.15.** `scripts/setup/setup_carla.sh` installs CARLA into `software/carla0915`.

```bash
git clone https://github.com/PervasiveAutonomyLab/PersonaDrive.git
cd PersonaDrive
bash scripts/setup/setup_carla.sh
```

**2. Create the conda environment.** One environment runs every stage, the CARLA agent and evaluation. The installer ends with an import self-check:

```bash
bash environment/setup_env.sh -n personadrive          # or -p /path/to/prefix
conda activate personadrive
```

Package versions, the manual install and the flash-attention options are in [environment/README.md](environment/README.md).

**3. Configure paths.** Every machine-specific path is read from environment variables.

```bash
cp .env.example .env        # then edit: PERSONADRIVE_ROOT, PERSONADRIVE_DATA_ROOT, CARLA_ROOT,
                            #            PERSONADRIVE_CONDA_ENV, ...
set -a; source .env; set +a
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PERSONADRIVE_ROOT}:${PYTHONPATH:-}"
```

`.env.example` documents every variable and its default.

## Data and released artefacts

| What | Where | Size |
|---|---|---:|
| PersonaDrive model (Stage 3) | Hugging Face model [PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/PervasiveAutonomyLab/PersonaDrive) | 1.96 GB |
| Style dataset: `default` (neutral, normal driving), `fast` (aggressive), `slow` (conservative) | Hugging Face dataset [PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/datasets/PervasiveAutonomyLab/PersonaDrive), one zip per style plus the annotations | 32.9 GB |
| Retrieval head (Stage 2) | this repository: `persona_drive_rag/checkpoints/best_model.pt` | 24 MB |
| Per-style FAISS indices | built from the dataset with the retrieval head (step 3 below) | — |
| `normal` (no-style) index: vectors and frame list | Hugging Face model [PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/PervasiveAutonomyLab/PersonaDrive), `retrieval/normal/`; completed from your SimLingo download (step 3 below) | 0.2 GB |
| Data-collection code (driver-in-the-loop rig) | this repository: [`human_data_collection/`](human_data_collection/README.md) | — |
| SimLingo checkpoint (Stage-3 initialisation) and PDM-lite data | upstream, from the SimLingo authors ([`RenzKa/simlingo`](https://huggingface.co/RenzKa/simlingo)) | — |

[datasets/README.md](datasets/README.md) has the dataset layout, the profile ↔ style mapping and how to add newly collected driving data.

### 1. Download the model

```bash
hf download PervasiveAutonomyLab/PersonaDrive --local-dir pretrained \
    --include "personadrive_icl_sft/.hydra/*" "personadrive_icl_sft/checkpoints/epoch=000.ckpt/*"
export PERSONADRIVE_CKPT=$PWD/pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt
```

- **What it is:** the complete Stage-3 model in one file: SimLingo's frozen weights plus the trained LoRA adapters and heads, with the `.hydra/` config of its run.
- **Loading:** it loads directly, with no DeepSpeed consolidation.
- **Who reads it:** every evaluation and analysis script reads `PERSONADRIVE_CKPT`.

### 2. Download the dataset

```bash
export PERSONADRIVE_DATA_ROOT=/path/to/personadrive_data          # will contain simlingo/
hf download PervasiveAutonomyLab/PersonaDrive --repo-type dataset --local-dir downloads/personadrive_data
for z in downloads/personadrive_data/*.zip; do unzip -q "$z" -d "$PERSONADRIVE_DATA_ROOT"; done
```

To evaluate a single style, `data_<style>.zip`, `commentary.zip` and `drivelm.zip` are enough. For example, normal driving only:

```bash
hf download PervasiveAutonomyLab/PersonaDrive data_default.zip commentary.zip drivelm.zip \
    --repo-type dataset --local-dir downloads/personadrive_data
```

### 3. Build the retrieval indices

At every tick, the agent searches a FAISS index of the selected style and opens the camera frames of the retrieved demonstrations. Build one index per style from the dataset with the released retrieval head (1 GPU):

```bash
conda activate personadrive
# frame sentences (stride 5) + SigLIP image / BGE-M3 text embeddings, per style
bash scripts/retrieval/run_stage1_mining.sh default fast slow
# one FAISS index per style, embedded with the released retrieval head
bash scripts/retrieval/build_style_indices.sh --checkpoint persona_drive_rag/checkpoints/best_model.pt default fast slow
```

The outputs go to `persona_drive_rag/{processed_routes,embeddings_v2,triplets_v2,faiss_index}/<style>/`. The agent's defaults `PERSONADRIVE_ICL_ROOT=persona_drive_rag` and `PERSONADRIVE_ICL_CKPT=persona_drive_rag/checkpoints/best_model.pt` point there.

- **Model cache:** the embedding models are downloaded into `PERSONADRIVE_HF_CACHE` if it is set, otherwise into Hugging Face's default cache (`$HF_HOME`, else `~/.cache/huggingface`).
- **Time:** on one L40S, Stage 1 and the index for `default` (1,455 frames at stride 5) take about 10 minutes; `fast` and `slow` have more frames and take longer.

The style-agnostic `normal` index retrieves from a slice of the SimLingo dataset (PDM-lite autopilot data, 54,040 frames). The SimLingo licence does not allow redistributing that data, so the release holds the index's vectors and frame list, and you complete it from your own SimLingo download:
1. Download the SimLingo splits `training_1_scenario`, `training_3_scenarios` and `lb1_split` (data and commentary, about 550 GB) and extract them under `$PERSONADRIVE_DATA_ROOT` (see [datasets/README.md](datasets/README.md)).
2. Download the index and complete it:

```bash
hf download PervasiveAutonomyLab/PersonaDrive --include "retrieval/normal/*" --local-dir downloads/personadrive_model
python scripts/retrieval/fill_normal_index.py --release downloads/personadrive_model/retrieval/normal \
    --out persona_drive_rag/faiss_index/normal
```

To build a `normal` index of your own instead, run `bash scripts/retrieval/run_stage1_mining.sh normal`, then `bash scripts/retrieval/build_style_indices.sh --checkpoint persona_drive_rag/checkpoints/best_model.pt normal`. It mines `training_1_scenario` at stride 5.

Stage 3 starts from the SimLingo checkpoint. Download it to the path the config expects:

```bash
hf download RenzKa/simlingo simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt simlingo/.hydra/config.yaml \
    --local-dir pretrained/RenzKa_simlingo
```

InternVL2-1B (`OpenGVLab/InternVL2-1B`), SigLIP (`google/siglip-base-patch16-224`) and BGE-M3 (`BAAI/bge-m3`) are downloaded from Hugging Face on first use.

## Quick start: evaluate the released model

With the model downloaded, the dataset extracted and the style indices built (steps 1–3 above), and the `personadrive` env active:

```bash
# one Bench2Drive route on one GPU, normal (neutral) driving style, K = 2; writes PNGs, telemetry and an MP4 of the run
PERSONADRIVE_ICL_PROFILE=default \
ROUTE_XML=$PERSONADRIVE_ROOT/leaderboard/data/bench2drive_split/bench2drive_00.xml \
OUT_DIR=/tmp/personadrive_route00 \
bash scripts/eval/test_one_lb1_route.sh
```

- **Styles:** switching style only changes `PERSONADRIVE_ICL_PROFILE`: `default` (neutral), `fast` (aggressive), `slow` (conservative), or `normal` (style-agnostic).
- **CARLA:** the Bench2Drive evaluator starts its own CARLA server from `CARLA_ROOT`. Do not start one yourself (`scripts/eval/start_carla.sh` is only for debugging CARLA in isolation).
- **Inputs:**
  - The script reads `AGENT_CKPT`, which defaults to `${PERSONADRIVE_CKPT}` or the released model from step 1 (`pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt`).
  - It also reads `PERSONADRIVE_ICL_PROFILE`, `PERSONADRIVE_ICL_ROOT` and `PERSONADRIVE_NUM_ICL` (0 disables retrieval).
- **Keep `PERSONADRIVE_DATA_ROOT` set:** retrieved frames are loaded from it. Frames that cannot be found are skipped, and if none can be loaded the agent silently drives without retrieval.
- **Checkpoint:** the released model is a single file and loads directly. A checkpoint you train yourself (`<run>/checkpoints/last.ckpt`) is a DeepSpeed directory instead.
  - Consolidate it once before evaluating: `python scripts/eval/consolidate_checkpoint.py <run>/checkpoints/last.ckpt`. This writes `consolidated_fp32.pt` inside it.
  - Otherwise the first job consolidates and caches it during setup, which can hit the 600 s setup timeout when many jobs start together.
- **Outputs:** without `OUT_DIR`, results go to `eval_results/single_route/<route>/<timestamp>/`.

If a step fails, see [Verification and troubleshooting](troubleshooting/README.md).

## Full pipeline

Each stage has one entry point. Per-stage details, file formats and provenance are in [persona_drive_rag/README.md](persona_drive_rag/README.md).

### Stages 1–2 – triplet mining, retrieval head, style indices (`personadrive` env, 1 GPU)

Anyone can train the retrieval head with three scripts. Each one reads and writes `PERSONADRIVE_ICL_ROOT`, which defaults to `persona_drive_rag/`, and skips work that already exists unless `FORCE=1`.

```bash
conda activate personadrive

# Stage 1: structured frame sentences (stride 5) → SigLIP + BGE-M3 embeddings → triplets, per style
bash scripts/retrieval/run_stage1_mining.sh                     # default fast slow  (add `normal` for PDM-lite)

# Stage 2: train the retrieval head, pooled over every profile found
bash scripts/retrieval/train_retrieval_head.sh                  # -> persona_drive_rag/checkpoints/head_paper/best_model.pt

# Stage 2b: one FAISS index per style, built with that head
bash scripts/retrieval/build_style_indices.sh --checkpoint persona_drive_rag/checkpoints/head_paper/best_model.pt
export PERSONADRIVE_ICL_CKPT=$PERSONADRIVE_ROOT/persona_drive_rag/checkpoints/head_paper/best_model.pt   # use it downstream
```

Stage 1 needs the style dataset (and PDM-lite for `normal`), see [Data and released artefacts](#data-and-released-artefacts). `train_retrieval_head.sh` trains on the triplets of every profile found under `PERSONADRIVE_ICL_ROOT`, with a weighted triplet loss (margin 0.3, hard-negative weight 2.0, AdamW, lr 1e-4, weight decay 0.01, batch 128, 20 epochs).

**Which head to use:** the released head `persona_drive_rag/checkpoints/best_model.pt` built every released index and context point, so use it with the released indices and checkpoint. If you train your own head, use it consistently: rebuild the indices with it, re-run Stage 3a with `PERSONADRIVE_ICL_CKPT` pointing at it, and fine-tune Stage 3 on those context points.

Extra arguments are passed through to `persona_drive_rag/embed_training.py`, for example `--epochs 5` or `--name my_head`. The exact per-profile commands that produced the released artefacts, including the PDM-lite pivot pipeline for the `normal` index, are in [persona_drive_rag/README.md](persona_drive_rag/README.md#commands-that-produced-the-shipped-artefacts).

### Stage 3a – precompute retrieved context points (`personadrive` env, 1 GPU)

```bash
conda activate personadrive
cd "$PERSONADRIVE_ROOT"
bash scripts/training/run_icl_local.sh            # normal -> PDM-lite routes, slow/fast -> {slow,fast}/lb1_split; add --overwrite to redo
# uses PERSONADRIVE_ICL_ROOT (indices) and PERSONADRIVE_ICL_CKPT (head); default: the released ones
```

This writes `icl_examples.json.gz` (K = 2 context points per frame, from the matching style index) next to each route's `measurements.json`. It reads `PERSONADRIVE_DATA_ROOT` and `PERSONADRIVE_ICL_ROOT` (default `persona_drive_rag/`). The `lb1_split` routes in the Hugging Face dataset already contain them.

### Stage 3b – in-context fine-tuning of the backbone (`personadrive` env)

```bash
ln -s "${PERSONADRIVE_DATA_ROOT%%:*}" database       # data_path in the config is relative to the repo root
python simlingo_training/train.py experiment=personadrive_icl_sft
```

`personadrive_icl_sft` is the configuration of the run that produced the [released checkpoint](https://huggingface.co/PervasiveAutonomyLab/PersonaDrive) (a verbatim copy of `experiment/debug.yaml`):

| Setting | Value |
|---|---|
| Initialisation | SimLingo `epoch=013` (`pretrained/RenzKa_simlingo/...`) |
| Trainable | LoRA on the language model (r = 32, α = 64, dropout 0.1); vision encoder frozen |
| Optimiser / lr | AdamW, 3e-5 (OneCycle, `pct_start` 0.05, weight decay 0.1) |
| Batch | 6 per GPU × 4 GPUs, DeepSpeed ZeRO-2, bf16 |
| Epochs | 1 (validation every 0.25 epoch; top-2 checkpoints by `val/loss`) |
| K | 2 retrieved context points (`num_icl_examples`) |
| Data | All routes with `icl_examples.json.gz`: 2,848 PDM-lite + 350 `fast/lb1_split` + 350 `slow/lb1_split`; upstream 99/1 random route split |
| Hardware / time | 4× NVIDIA L40S (48 GB), ≈ 12 h |
| Smaller GPUs | Batch 6 per GPU needs a 48 GB GPU. On 24 GB GPUs (A30), keep the same effective batch with `data_module.batch_size=2 accumulate_grad_batches=3`. |

Training logs to Weights & Biases (`debug=True` switches it to offline).

## Evaluation and the paper's tables

Every number in the paper's result tables comes from one of the scripts below. With the released checkpoint downloaded and the indices built (or your own), run the evaluations, then print all tables in the paper's layout with `scripts/eval/paper_tables.py`.

| Paper table | Run | Aggregation |
|---|---|---|
| Table 1, PersonaDrive (no style) | Closed loop, profile `normal`, K = 2 | `scripts/eval/aggregate_b2d_metrics.py`, row `normal` |
| Table 2 and appendix K table: DS, SR, Eff., Comf. | Closed loop, profile `normal`, K = 1, 3, 4. K = 2 is the Table 1 run; K = 0 is SimLingo's published result. | `scripts/eval/aggregate_b2d_metrics_k.py` |
| Table 2 and appendix K table: ADE, FDE, tokens, TTFT, TTLT, inference total | `scripts/analysis/waypoint_accuracy_sweep.py --use-cot` on the paper's 200 frames (`scripts/analysis/openloop_samples_v1.json`) | — |
| Appendix K table: retrieval (RAG) overhead | `scripts/analysis/benchmark_icl_timing.py` (embedding + lookup) | — |
| Table 3, PersonaDrive rows | Closed loop, profiles `slow`, `default`, `fast` (Conservative, Neutral, Aggressive), K = 2 | `scripts/eval/aggregate_b2d_metrics.py`, rows `slow` / `default` / `fast` |
| Appendix: human speed statistics | `scripts/analysis/style_speed_stats.py` (style dataset only, no GPU) | — |

```bash
set -a; source .env; set +a      # PERSONADRIVE_CKPT, PERSONADRIVE_ICL_ROOT, PERSONADRIVE_EVAL_OUT_ROOT, SLURM accounts, ...

# 1. Closed loop: Bench2Drive, 220 routes per condition, one SLURM job per route.
#    Each launch keeps submitting until its routes are done and skips finished routes when restarted;
#    run the four launches in separate shells to evaluate the conditions in parallel.
PERSONADRIVE_PROFILES=normal,slow,default,fast python scripts/eval/start_eval_simlingo.py   # K = 2 -> $PERSONADRIVE_EVAL_OUT_ROOT
PERSONADRIVE_PROFILES=normal PERSONADRIVE_NUM_ICL=1 python scripts/eval/start_eval_simlingo.py   # -> ${PERSONADRIVE_EVAL_OUT_ROOT}_k1
PERSONADRIVE_PROFILES=normal PERSONADRIVE_NUM_ICL=3 python scripts/eval/start_eval_simlingo.py   # -> ${PERSONADRIVE_EVAL_OUT_ROOT}_k3
PERSONADRIVE_PROFILES=normal PERSONADRIVE_NUM_ICL=4 python scripts/eval/start_eval_simlingo.py   # -> ${PERSONADRIVE_EVAL_OUT_ROOT}_k4

# 2. Open loop and per-tick cost (1 GPU, no CARLA; needs the PDM-lite and lb1_split routes with icl_examples.json.gz)
python scripts/analysis/waypoint_accuracy_sweep.py --use-cot --database "$PERSONADRIVE_DATA_ROOT/simlingo" \
    --samples-json scripts/analysis/openloop_samples_v1.json --out-json waypoint_sweep_results.json
python scripts/analysis/benchmark_icl_timing.py --k-list 0,1,2,3,4 --n-iter 10 --out-json bench_icl_timing.json

# 3. Human speed statistics (CPU)
python scripts/analysis/style_speed_stats.py

# 4. All tables in the paper's layout (runs the two aggregators; CSVs go to paper_tables/)
python scripts/eval/paper_tables.py --sweep-json waypoint_sweep_results.json --timing-json bench_icl_timing.json
```

- **Leave `PERSONADRIVE_NUM_ICL` unset for K = 2.** The training config's K = 2 is then used and results go to `$PERSONADRIVE_EVAL_OUT_ROOT`, which the per-style aggregator reads. Setting it to 2 writes to `..._k2/` instead.
- **Routes averaged.** The aggregators average over the routes that produced a `result.json`. When a route was run more than once (the launcher retries up to 3 times), they keep its run with the highest Driving Score. `paper_tables.py` prints how many of the 220 routes each number covers.
- **Comfort ×100.** Comfort is printed ×100, as in the paper. Speed and acceleration in Table 3 are the mean ego speed and longitudinal acceleration from the run logs.
- **Hardware dependence.** Latency depends on the GPU. Open-loop ADE/FDE varies by about 0.01–0.04 m between GPU types, because the commentary is generated in bf16.
- **Without SLURM.** `scripts/eval/run_three_profiles.sh`, `scripts/eval/run_three_profiles_parallel.sh` and `scripts/eval/run_all_routes_profiles.sh` run routes locally against a CARLA server started by `scripts/eval/start_carla.sh`. Upstream's `Bench2Drive/tools/merge_route_json.py` also works on the per-route results.

**Other analyses** (not in the paper's tables):

| Script | Measures |
|---|---|
| `scripts/analysis/style_retrieval_probe.py` | Mean speed of the frames retrieved from each index |
| `scripts/analysis/style_conditioning_eval.py` | Query-style × index matrix of predicted speed |
| `scripts/analysis/qa_accuracy_sweep.py` | QA accuracy versus K |

**Language evaluation (upstream):** `simlingo_training/eval.py` followed by `eval_metrics.py`.

## Repository layout

```
PersonaDrive/
├── persona_drive_rag/      Stages 1–2 code: frame sentences, embeddings, triplet mining, retrieval head, FAISS indices
├── simlingo_training/      Stage 3: the VLA backbone, ICL prompt prefix, online retriever, training configs
├── team_code/              CARLA agents (PersonaDrive agent: team_code/agent_simlingo.py; PDM-lite expert)
├── scripts/
│   ├── setup/              setup_carla.sh, carla_exports.sh
│   ├── retrieval/          run_stage1_mining.sh → train_retrieval_head.sh → build_style_indices.sh   (Stages 1–2)
│   ├── training/           run_icl_local.sh + generate_icl_examples.py (Stage 3a), train_simlingo_seed1.sh (baseline)
│   ├── eval/               Bench2Drive: test_one_lb1_route.sh, start_eval_simlingo.py, run_*_profiles.sh, aggregate_b2d_metrics*.py
│   ├── analysis/           open-loop experiments: style probes, ADE/FDE and QA vs K, latency
│   ├── data_collection/    PDM-lite data agent runs on SLURM
│   └── data_processing/    raw recordings → labelled, aggregated dataset (run_pipeline.py, records_integrating.py, …)
├── datasets/               dataset documentation and a drop zone for new recordings
├── human_data_collection/  driver-in-the-loop recording rig (wheel + pedals), label generation, conversion to the dataset layout
├── environment/            conda envs: *.yaml (conda layer), requirements-*.txt (pip pins), setup_env.sh
├── tests/                  static checks, mini dataset, end-to-end smoke test, retrieval / LoRA / eval tests
├── more_info/             demos, method, dataset and results
├── troubleshooting/       installation checks and common problems
├── assets/                 README figures and animations
└── Bench2Drive/, leaderboard*/, scenario_runner*/, dataset_generation/, data/, tools/
                            inherited from SimLingo / CARLA garage (benchmark harness, expert, label generators, route files)
```

## What is (not) in this repository

**Included:**
- All code for Stages 1–3, the CARLA agent and evaluation.
- The two retrieval heads (24 MB each), with provenance of the runs that produced the released indices.
- Route files and CARLA map assets.
- The data-collection rig code ([human_data_collection/](human_data_collection/README.md)).
- Environment captures and tests.

**Released separately:**
- The style dataset, on Hugging Face ([PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/datasets/PervasiveAutonomyLab/PersonaDrive), dataset).
- The Stage-3 checkpoint, on Hugging Face ([PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/PervasiveAutonomyLab/PersonaDrive), model).

**Not included:**
- CARLA itself.
- Upstream model weights (Hugging Face).
- The upstream PDM-lite dataset.

## Acknowledgements

<p align="center">
<a href="https://faculty.sites.uci.edu/elmalaki/"><img src="assets/logo_lab.png" height="70" alt="Pervasive Autonomy Lab"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.nsf.gov/"><img src="assets/logo_nsf.png" height="70" alt="NSF"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="assets/logo_properai.png" height="70" alt="UCI ProperAI Institute">
</p>

This work is supported by the U.S. National Science Foundation under grant 2339266 and partially supported by the UCI ProperAI Institute. PersonaDrive builds directly on SimLingo, CARLA garage, DriveLM / PDM-Lite and Bench2Drive; we thank their authors for releasing their code. The project page design is adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io).

## Citation

```bibtex
@article{srewa2026personadrive,
  title   = {PersonaDrive: Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation},
  author  = {Srewa, Mahmoud and Iddamsetty, Praneetsai Vasu and Al Faruque, Mohammad Abdullah and Elmalaki, Salma},
  journal = {arXiv preprint arXiv:2606.12616},
  year    = {2026}
}
```

Please also cite the works this code builds on:

```bibtex
@InProceedings{Renz2025cvpr,
  title     = {SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author    = {Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
@inproceedings{Sima2024ECCV,
  title     = {DriveLM: Driving with Graph Visual Question Answering},
  author    = {Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Jens Beißwenger and Ping Luo and Andreas Geiger and Hongyang Li},
  booktitle = {Proc. of the European Conf. on Computer Vision (ECCV)},
  year      = {2024}
}
@inproceedings{Jia2024NeurIPS,
  title     = {Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author    = {Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle = {NeurIPS 2024 Datasets and Benchmarks Track},
  year      = {2024}
}
```

## License

The code is released under the Apache 2.0 [LICENSE](LICENSE) inherited from SimLingo. [LICENCE_Dataset](LICENCE_Dataset) is SimLingo's licence for the upstream PDM-lite dataset.

This work was made using the SimLingo dataset, provided by Wayve Technologies Ltd under licence terms available at https://github.com/RenzKa/simlingo. The released model and retrieval heads were trained with that dataset. Its terms therefore restrict them:
- they may only be distributed free of charge and on an academic or scientific basis;
- they must not be used in, or to assist in, the operation of a vehicle or robot, or for any high-risk purpose.
