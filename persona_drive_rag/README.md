# `persona_drive_rag/`: Stages 1–2 (triplet mining, retrieval head, style indices)

This directory holds the retrieval side of PersonaDrive:
- **Stage 1** builds a structured sentence per frame, embeds frames with frozen SigLIP (image) and BGE-M3 (sentence), and mines anchor / positive / hard-negative / easy-negative triplets.
- **Stage 2** trains the retrieval head (frozen SigLIP ⊕ `ControlEncoder` → `ProjectionMLP` → L2-norm) and builds one FAISS index per style.

Stage 3 and the online retriever used in CARLA (`simlingo_training/utils/icl_retrieval.py`) live elsewhere in the repository.

Run everything here in the **`personadrive`** environment (see [`../environment/README.md`](../environment/README.md)), with this directory as the working directory. All scripts use paths relative to it, and every path below is relative to it.

## Quick path: three scripts

To train your own retrieval head and indices, use the wrappers in [`../scripts/retrieval/`](../scripts/retrieval). They run the files below with the right arguments:

```bash
bash scripts/retrieval/run_stage1_mining.sh            # Stage 1 for default fast slow (add normal for PDM-lite)
bash scripts/retrieval/train_retrieval_head.sh          # Stage 2
bash scripts/retrieval/build_style_indices.sh --checkpoint persona_drive_rag/checkpoints/head_paper/best_model.pt
```

They read and write `PERSONADRIVE_ICL_ROOT` (default: this directory) and are run from the repository root. The rest of this page documents the underlying scripts and the exact commands behind the released artefacts.

## Files

| Stage | File | Role |
|---|---|---|
| 1a | `rag_data_processor_v2.py` | Frame sentences for one profile. Strides frames (`--frame-step 5`) from frame 10. Writes `<out>/<profile>_dataset_processed.json`. Imports the helpers in `rag_data_processor.py`. |
| 1a | `rag_data_processor.py` | Shared helpers (RoadOption → command map, waypoint extraction), plus the older per-frame CLI. |
| 1a (`normal`) | `rag_pivot_runner.py`, `processed_routes/process.py` | PDM-lite pivot frames from `split_jsons/*_trimmed.json`, then injection of `Target Point` lines into their sentences. |
| 1b (`normal`) | `embed_slurm.sh` → `embed_single.py` → `embed_merge.py` | Sharded SigLIP + BGE-M3 embedding on SLURM, then merge. Used for the 54k-frame `normal` profile. |
| 1b–c | `embed_mine.py` | Embeds (SigLIP `google/siglip-base-patch16-224`, BGE-M3 `BAAI/bge-m3`), computes the combined similarity and mines triplets. Reuses cached embeddings when present. |
| 2 | `embed_training.py`, `embed_training_slurm.sh` | Retrieval-head training. |
| 2b | `build_faiss_index.py` | Embeds frames with the trained head and writes `faiss_index/<profile>/driving_index.{faiss,meta.pkl,info.json}`. |
| tool | `rag_app.py` | Flask page: upload an image and a vehicle state, see the top-k frames from every index. |
| tool | `triplet_viewer/app_v2.py` | Flask page showing mined (anchor, positive, hard negative, easy negative) quadruples from `../triplets_v2/<profile>/triplets.jsonl`. |
| tool | `logs.py` | Flask SSE tail of `slurm_logs/`. |
| — | `checkpoints/best_model.pt` | **The retrieval head used by every index and every reported result.** |
| — | `checkpoints/multi_profile/best_model.pt` | Later multi-profile run, not used by any index or result. Shipped for completeness. |
| — | `provenance/` | FAISS `info.json`, mining thresholds and logs, and Stage-2 wandb metadata of the runs that produced the shipped artefacts. |

The tools serve local files by absolute path. Run them on a trusted machine or through an SSH tunnel; never expose them publicly.

## Profiles

| Profile | Paper style | Index contents (shipped) | Stage-1 input |
|---|---|---|---|
| `default` | neutral (normal driving) | 848 frames (`driver1`–`driver8`) | `--profile default` |
| `fast` | aggressive | 4,559 frames (≈87 % `lb1_split`, rest `driver*`) | `--profile fast` |
| `slow` | conservative | 8,960 frames (≈87 % `lb1_split`, rest `driver*`) | `--profile slow` |
| `normal` | none (style-agnostic) | 54,040 PDM-lite pivot frames | `rag_pivot_runner.py` |

## Commands that produced the shipped artefacts

The commands below produced the paper's indices (the "shipped" column above). Those Stage 1–2 outputs are not part of the Hugging Face release. To build the per-style indices from the released dataset, see [step 3 of the main README](../README.md#3-build-the-retrieval-indices).

```bash
conda activate personadrive
set -a; source ../.env; set +a
cd "$PERSONADRIVE_ROOT/persona_drive_rag"
DB="${PERSONADRIVE_DATA_ROOT%%:*}/simlingo"
```

**Stage 1a: frame sentences** (May 2026):

```bash
for p in default fast slow; do
  python rag_data_processor_v2.py --data-root "$DB" --profile $p --frame-step 5 --output-dir processed_routes/$p
done
```

**Stage 1b–c: embeddings and triplets.** `default` was mined with smaller pools and a stricter hard-negative gate than `fast`/`slow`; `fast`/`slow` use the file defaults.

```bash
python embed_mine.py --dataset-json processed_routes/default/default_dataset_processed.json \
    --emb-dir embeddings_v2/default --triplets-dir triplets_v2/default \
    --top-k-pool 200 --bottom-k-pool 200 --hard-percentile 50
for p in fast slow; do
  python embed_mine.py --dataset-json processed_routes/$p/${p}_dataset_processed.json \
      --emb-dir embeddings_v2/$p --triplets-dir triplets_v2/$p
done
```

`provenance/thresholds/*.json` records the exact pools, percentiles and resulting thresholds of each run. `provenance/run_logs/` holds the run logs.

**`normal` profile: PDM-lite pivots** (April 2026). This needs the upstream PDM-lite data and the pivot lists `split_jsons/*_trimmed.json`, which are part of the paper's Stage 1–2 outputs.

```bash
python rag_pivot_runner.py --json-folder split_jsons          # -> processed_pivots/*_processed.json
(cd processed_pivots && python ../processed_routes/process.py) # adds Target Point lines, in place, to every *.json in cwd
bash embed_slurm.sh processed_pivots embeddings_shards embeddings/normal   # SLURM array + merge
python embed_mine.py --dataset-json processed_pivots --emb-dir embeddings/normal \
    --triplets-dir triplets/normal --top-k-pool 200 --bottom-k-pool 200 --hard-percentile 50
```

The generator of the `split_jsons` pivot lists (per-command samples from the PDM-lite splits) is not part of this repository. The lists themselves are released.

**Stage 2: retrieval head.** The shipped `checkpoints/best_model.pt` (14 Apr 2026) was trained on the `normal` triplets with `embed_training_slurm.sh`, which runs:

```bash
python embed_training.py --jsonl_path triplets/triplets.jsonl --image_emb_path embeddings/image_embeddings.pt \
    --frame_index_path embeddings/frame_ids.json --batch_size 512 --epochs 20 --lr 1e-4
```

`checkpoints/multi_profile/best_model.pt` (28 Apr 2026) is the pooled run. Its exact arguments are in `provenance/stage2_multi_profile_wandb-metadata.json`.

```bash
python embed_training.py --balanced \
    --profile normal triplets/normal/triplets.jsonl embeddings/normal/image_embeddings.pt embeddings/normal/frame_ids.json \
    --profile fast   triplets/fast/triplets.jsonl   embeddings/fast/image_embeddings.pt   embeddings/fast/frame_ids.json \
    --profile slow   triplets/slow/triplets.jsonl   embeddings/slow/image_embeddings.pt   embeddings/slow/frame_ids.json \
    --checkpoint_dir checkpoints/multi_profile --epochs 20 --batch_size 512 --device cuda
```

**Stage 2b: FAISS indices.** These were built with `checkpoints/best_model.pt`. `default`, `fast` and `slow` were built on 5 May 2026; `normal` on 29 Apr 2026.

```bash
python build_faiss_index.py --checkpoint checkpoints/best_model.pt --output_dir faiss_index --device cuda \
  --profile default none embeddings_v2/default/image_embeddings.pt embeddings_v2/default/frame_ids.json processed_routes/default/default_dataset_processed.json \
  --profile fast    none embeddings_v2/fast/image_embeddings.pt    embeddings_v2/fast/frame_ids.json    processed_routes/fast/fast_dataset_processed.json \
  --profile slow    none embeddings_v2/slow/image_embeddings.pt    embeddings_v2/slow/frame_ids.json    processed_routes/slow/slow_dataset_processed.json
python build_faiss_index.py --checkpoint checkpoints/best_model.pt --output_dir faiss_index --device cuda \
  --profile normal triplets/normal/triplets.jsonl embeddings/normal/image_embeddings.pt embeddings/normal/frame_ids.json normal_processed.json
```

`JSONL=none` indexes every frame in `frame_ids.json`. Passing a `triplets.jsonl` instead indexes only its unique anchors, which is what `normal` uses. The optional fifth argument enriches every entry with its full context point. Inspect the result with `python rag_app.py` (it auto-discovers `faiss_index/*`).

Then continue with Stage 3a (`../scripts/training/run_icl_local.sh`) as described in the main README.

## File formats

| File | Contents |
|---|---|
| `processed_routes/<p>/<p>_dataset_processed.json` | List of frames. Each has `frame_path` (absolute), `sentence`, `command`, `next_command`, `target_point(_next)`, `future_waypoints` (10×2), `temporal_waypoints_org`, `route_adjusted`, `scenario`, `profile`, `driver`, `run`. |
| `embeddings_v2/<p>/frame_ids.json` | `{frame_ids, image_paths, actions, vehicle_states}`, row-aligned with `image_embeddings.pt` (N×768, SigLIP) and `text_embeddings.pt` (N×1024, BGE-M3). |
| `embeddings_v2/<p>/thresholds.json` | Mining percentiles, pool sizes and the resulting similarity thresholds. |
| `triplets_v2/<p>/triplets.jsonl` | One record per anchor: `{"anchor": {frame_id, frame_path, state, action}, "positives": [...], "hard_negatives": [...], "negatives": [...]}`. Each candidate has `text_sim`, `image_sim`, `combined_sim`. |
| `faiss_index/<p>/driving_index.faiss` | `IndexFlatIP` over L2-normalised 1024-d head outputs (cosine similarity). |
| `faiss_index/<p>/driving_index.meta.pkl` | Per vector, the context point: `frame_path`, `state`, `state_1_back`, `state_2_back`, `command`, `target_point(_next)`, `future_waypoints`, `commentary`, `vqa`, `sentence`, `scenario`. |
| `faiss_index/<p>/driving_index.info.json` | Build summary, including the checkpoint used. |

## Paper ↔ code names

| Paper | Code |
|---|---|
| Retrieval head $f_{\text{ret}}$ (+ $f_c$) | `DrivingEmbeddingModel` = `ControlEncoder` + `ProjectionMLP` (`embed_training.py`, mirrored in `build_faiss_index.py`, `rag_app.py`, `../simlingo_training/utils/icl_retrieval.py`) |
| Frame sentence $e_\tau$ | `sentence` in `*_dataset_processed.json` |
| Combined similarity $\sigma$ | `ALPHA * text_sim + BETA * image_sim` in `embed_mine.py` (0.5 / 0.5) |
| Positive / negative pools $N^{+}$ / $N^{-}$ | `TOP_K_POOL` / `BOTTOM_K_POOL` (absolute counts) plus percentile gates `POS/HARD/NEG_PERCENTILE` |
| Context point $\xi_\tau$ | One `driving_index.meta.pkl` entry, serialised into `icl_examples.json.gz` by `../scripts/training/generate_icl_examples.py` |
| Style $p$ / database $\mathcal{H}^p$ | profile / `faiss_index/<profile>/` |
| $K$ | `num_icl_examples` (training config), `PERSONADRIVE_NUM_ICL` (evaluation override) |

