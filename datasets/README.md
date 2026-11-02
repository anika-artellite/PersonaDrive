# Datasets

PersonaDrive uses two kinds of data:

1. **The PersonaDrive style dataset (ours).** Driving recorded in CARLA under three style instructions:
   - `default`: neutral, normal driving;
   - `fast`: aggressive;
   - `slow`: conservative.

   It is on Hugging Face: **[PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/datasets/PervasiveAutonomyLab/PersonaDrive)** (dataset).
2. **The upstream SimLingo PDM-lite dataset.** The SimLingo authors publish it, and it is not repackaged here. It is needed for Stage-3 fine-tuning and for the style-agnostic `normal` index.

All data lives outside the repository under one **dataset root**: the directory that *contains* `simlingo/`. Point `PERSONADRIVE_DATA_ROOT` at it (see `.env.example`).

## The style dataset

| File | Size | Contents |
|---|---:|---|
| `data_default.zip` | 2.37 GB | Normal (neutral) driving: camera frames, per-frame measurements, surrounding objects (9,613 frames). |
| `data_fast.zip` | 11.69 GB | Aggressive driving (39,734 frames), including the retrieved context points used for fine-tuning (`icl_examples.json.gz`). |
| `data_slow.zip` | 18.84 GB | Conservative driving (64,215 frames), organized like `data_fast.zip`. |
| `commentary.zip` | 3.2 MB | Per-route `commentary.json` for all three styles. |
| `drivelm.zip` | 27.8 MB | Per-route `vqa.json` (DriveLM-style question–answer pairs) for all three styles. |

Every zip unpacks into the layout below. Download, verify and extract:

```bash
export PERSONADRIVE_DATA_ROOT=/path/to/personadrive_data        # will contain simlingo/
hf download PervasiveAutonomyLab/PersonaDrive --repo-type dataset --local-dir downloads/personadrive_data
# or only some of the zips, e.g.:  hf download PervasiveAutonomyLab/PersonaDrive data_default.zip commentary.zip drivelm.zip SHA256SUMS --repo-type dataset --local-dir downloads/personadrive_data
(cd downloads/personadrive_data && sha256sum -c SHA256SUMS --ignore-missing)
mkdir -p "$PERSONADRIVE_DATA_ROOT"
for z in downloads/personadrive_data/*.zip; do unzip -q "$z" -d "$PERSONADRIVE_DATA_ROOT"; done
```

[`MANIFEST.md`](MANIFEST.md) and [`checksums.sha256`](checksums.sha256) list the size, file count and SHA-256 of each zip.

**Upstream PDM-lite data.**
- **Where to get it:** download it from the SimLingo dataset on Hugging Face (`RenzKa/simlingo`).
- **Which splits:**
  - Stage-3 fine-tuning uses `training_1_scenario/routes_training/random_weather_seed_1_balanced_150` (2,848 routes, about 108 GB).
  - The `normal` index retrieves frames from three splits: `training_1_scenario/routes_training/random_weather_seed_1_balanced_150`, `training_3_scenarios/routes_training/random_weather_seed_3_balanced_100` and `lb1_split/routes_training/` (SimLingo's own `lb1_split`, not the `fast`/`slow` one of our dataset). It needs their `data_simlingo_<split>_*` and `commentary_simlingo_<split>_*` archives, about 274, 202 and 72 GB of data.
- **Where to put it:** extract it so that it ends up at `$PERSONADRIVE_DATA_ROOT/simlingo/{data,commentary,drivelm}/simlingo/<split>/...`.
- **Context points:** Stage 3a (`scripts/training/run_icl_local.sh`) writes their `icl_examples.json.gz` files.

**What you need for each task:**

| Task | Zips |
|---|---|
| Evaluate the released model with a style (`default` / `fast` / `slow`) | `data_<style>`, `commentary` and `drivelm`. The per-style FAISS index is built from them (main README, [Build the retrieval indices](../README.md#3-build-the-retrieval-indices)), and the agent opens the frames it retrieves. |
| Evaluate with the style-agnostic `normal` index | No zips. The released `normal` index (Hugging Face model, `retrieval/normal/`), completed with `scripts/retrieval/fill_normal_index.py` from the three SimLingo splits above (main README, [Build the retrieval indices](../README.md#3-build-the-retrieval-indices)). |
| Retrain Stage 3 | `data_fast`, `data_slow`, `commentary`, `drivelm`, plus PDM-lite |
| Re-run Stages 1–2 from scratch | all five zips, plus PDM-lite for the `normal` index |

Retrieved frames whose image is missing are skipped. If none can be loaded, the agent runs without retrieval, so keep `PERSONADRIVE_DATA_ROOT` pointing at the extracted data when you evaluate.

## On-disk layout (the contract every script relies on)

```
$PERSONADRIVE_DATA_ROOT/
└── simlingo/
    ├── data/simlingo/
    │   ├── default/driver<N>/routes/<Scenario>/<id>/<Town..>/           # normal (neutral) driving
    │   ├── fast/driver<N>/routes/<Scenario>/<id>/<Town..>/              # aggressive
    │   ├── fast/lb1_split/routes_training/<Scenario>/<inst>/<Town..>/   # aggressive (Stage-3 fine-tuning)
    │   ├── slow/driver<N>/routes/<Scenario>/<id>/<Town..>/              # conservative
    │   ├── slow/lb1_split/routes_training/<Scenario>/<inst>/<Town..>/   # conservative (Stage-3 fine-tuning)
    │   ├── training_1_scenario/routes_training/<weather>/<Town..>/      # SimLingo PDM-lite (Stage 3, `normal` index)
    │   ├── training_3_scenarios/routes_training/<weather>/<Town..>/     # SimLingo PDM-lite (`normal` index)
    │   └── lb1_split/routes_training/<Scenario>/<Town..>/               # SimLingo PDM-lite (`normal` index)
    ├── commentary/simlingo/<same relative paths>/commentary.json
    ├── drivelm/simlingo/<same relative paths>/vqa.json
    └── embeddings/{normal,fast,slow}_full/                           # Stage-3a SigLIP caches (written by Stage 3a)
```

Every route leaf (`<Town..>/`) contains:

| File | Contents |
|---|---|
| `rgb/NNNN.jpg` | Front camera at 4 Hz (dataset frames). |
| `rgb_augmented/NNNN.jpg` | Shifted-camera augmentation. |
| `measurements.json` | Per-frame list: `speed`, `throttle`, `steer`, `brake`, `command`, `next_command`, `target_point(_next)`, `ego_matrix`, `route`, `aim_wp`, … |
| `boxes.json` | Per-frame objects (`driver<N>` routes). |
| `result.json`, `records.json.gz` | Simulator logs. |
| `icl_examples.json.gz` | Written by Stage 3a for the Stage-3 training routes (`lb1_split` and PDM-lite). Maps frame index → K retrieved context points: `rgb_path`, `speed_history`, `throttle_history`, `command`, `target_point(_next)`, `commentary`, `waypoints` (10×2). |

`commentary.json` and `vqa.json` are per-frame lists aligned with `measurements.json`; missing frames are `null`.
- **Coverage:** of the 169 `driver<N>` recordings, 168 have `commentary.json` and 157 have `vqa.json`.
- **Frames without VQA:** the Stage-1 sentence carries "No data found for perception/prediction/planning." lines instead of question–answer pairs (`rag_data_processor_v2.py`).

| Profile name in code | Paper style | Source | Frames |
|---|---|---|---:|
| `default` | neutral (normal driving) | `default/driver*` | 9,613 |
| `fast` | aggressive | `fast/driver*` + `fast/lb1_split` | 39,734 |
| `slow` | conservative | `slow/driver*` + `slow/lb1_split` | 64,215 |
| `normal` | none (style-agnostic) | PDM-lite | — |

**Frame paths inside metadata.**
- **In the released zips:** the frame paths in `commentary.json` and `vqa.json` are relative to the dataset root (`simlingo/data/simlingo/...`).
- **Absolute paths:** FAISS metadata (`driving_index.meta.pkl`) and `icl_examples.json.gz` store the absolute frame paths of the machine they were built on.
- **How they are resolved:** at load time, `simlingo_training/dataloader/_path_resolve.py` (and its copies in `icl_retrieval.py`, `build_faiss_index.py` and `rag_app.py`) strips leading path components until `$PERSONADRIVE_DATA_ROOT/simlingo/...` exists. As long as the layout above is respected, the metadata works on any machine without rewriting.

## Adding newly collected driving data

Put raw recordings in [`raw/`](raw/README.md) and follow the conversion steps there. They produce the layout above, after which the Stage 1–3 commands in the main README apply unchanged.
