# `datasets/raw/` — drop zone for newly collected driving data

Everything in this folder except this README is gitignored. Use it, or any other directory, to stage new recordings before converting them into the dataset layout described in [`../README.md`](../README.md#on-disk-layout-the-contract-every-script-relies-on).

## What a recording must contain

The style dataset was recorded on a driver-in-the-loop rig running CARLA 0.9.15 with the code in [`human_data_collection/`](../../human_data_collection/README.md). There, `human_agent.py` subclasses SimLingo's data agent and records the wheel and pedals. That folder also generates the labels and converts recordings into the dataset layout; the steps below describe the same conversion with this repository's scripts. The agent saves one folder per route, with one file per frame:

```
<Town..>/
├── rgb/NNNN.jpg                 front camera
├── rgb_augmented/NNNN.jpg       shifted camera (augmentation)
├── measurements/NNNN.json.gz    ego state, controls, route, target points, hazards
├── boxes/NNNN.json.gz           surrounding objects (needed for VQA / commentary labels)
└── (lidar/, results.json.gz, ... as written by the agent)
```

Place each recording at the path it will have in the final dataset:

```
<root>/simlingo/data/simlingo/<style>/driver<N>/routes/<Scenario>/<id>/<Town..>/
```

`<style>` is `default` (neutral instruction), `fast` (aggressive) or `slow` (conservative).

## Conversion steps

1. **Language labels.** VQA and commentary are produced by the upstream SimLingo/DriveLM generators in `dataset_generation/language_labels/`. They need the auxiliary simulator state that the data agent saves in `measurements/` and `boxes/`.

   ```bash
   python dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py \
       --data-directory <root>/simlingo --output-directory <root>/simlingo/drivelm
   python dataset_generation/language_labels/commentary/carla_commentary_generator_main.py \
       --data-directory <root>/simlingo --output-directory <root>/simlingo/commentary
   ```

   `scripts/data_processing/run_pipeline.py --base-dir <root>/simlingo --skip-reorg` runs both generators.

   > **Limitation (upstream code, unchanged):** the generators look for routes at `data/simlingo/*/*/*/*/boxes/` (`carla_vqa_generator.py:66`, `carla_commentary_generator.py:81`). They also parse PDM-lite style route names, `TownXX_RepN_M_routeK...`, in `language_labels/utils.py:get_scenario_name`. They therefore work as-is for PDM-lite and `lb1_split` recordings. The human `driver<N>/routes/<Scenario>/<id>/<Town..>` routes are two levels deeper and use different names, so labelling them requires adapting that glob and the name parsing. The released style dataset already contains its `vqa.json` and `commentary.json`. The adapted labelling scripts are in [`human_data_collection/label_gen/`](../../human_data_collection/README.md#from-recordings-to-the-personadrive-dataset).

2. **Aggregate per-frame files into one file per route.** The PersonaDrive loaders read `measurements.json`, `vqa.json` and `commentary.json`.

   ```bash
   python scripts/data_processing/records_integrating.py --dataset-root <root>/simlingo --workers 8
   ```

   This writes null-padded per-route arrays, so list index = frame index: `measurements.json` next to `measurements/`, `vqa.json` next to `vqa/`, and so on. It skips routes that are already converted. `scripts/data_processing/merge_measurements.py --data-root <dir>` does the same for `measurements/` only.

3. **Check the layout.** Run `python persona_drive_rag/rag_data_processor_v2.py --data-root <root>/simlingo --profile <style> --frame-step 5 --output-dir /tmp/check`. It must report the new routes and write non-empty frame sentences.

4. **Continue with the main pipeline.** Run Stages 1–2 to add the frames to the per-style database (FAISS index). New styles only need a new profile directory and their own index; the retrieval head and the backbone stay unchanged.

The `lb1_split` routes went through the same steps, plus `scripts/data_processing/reorganize_dataset.py`, which moved the CARLA leaderboard output into `lb1_split/routes_training/<Scenario>/...`.
