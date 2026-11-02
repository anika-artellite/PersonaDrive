# PersonaDrive — data collection

Code for recording the PersonaDrive style dataset. It is released on Hugging Face as [PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/datasets/PervasiveAutonomyLab/PersonaDrive).

A human drives CARLA Leaderboard 2.0 routes through a Logitech force-feedback wheel and pedals. Each route is driven three times, under three persona instructions (`slow`, `default`, `fast`), while the full sensor stack records.

`human_agent.py` subclasses SimLingo's `DataAgent` and replaces only the control source.
- **Sensors:** sensor recording is SimLingo's and is untouched.
- **Autopilot:** the privileged autopilot still runs every frame, so each frame also carries the autopilot's controls alongside the human's.

No recorded data belongs in this folder: recorded paths embed participant names. Use anonymous participant IDs (`driver<N>`, see [Output](#output)) and keep `DATASET_ROOT` outside the repository.

## What is bundled

Only the simulator is external. Everything else runs from this folder.

| Path | What | Notes |
|---|---|---|
| `leaderboard/` | CARLA Leaderboard 2.0 | Carries the two collection patches: `route_scenario.py` debug drawing, `route_parser.py` for Python 3.9. |
| `scenario_runner/` | CARLA scenario_runner | |
| `PythonAPI/carla/` | CARLA's pure-Python `agents` helpers | Install the client itself with `pip install carla==0.9.15`. |
| `simlingo/team_code/` | SimLingo's `DataAgent`, autopilot and helpers, in the versions the dataset was recorded with | These differ from PersonaDrive's `team_code/`: newer numpy/webcolors support and extra recorded fields. |
| `simlingo/dataset_generation/` | SimLingo's VQA / commentary generators, the sources that `label_gen/` patches | |
| `simlingo/simlingo_training/utils/transfuser_utils.py` | Geometry helpers used by the label generators | |
| `simlingo/data/simlingo/long_routes/` | The long routes used by `collect_session.sh <name> long` | |

Some content is identical to files elsewhere in this repository, so it is linked rather than copied:
- **Links:** `simlingo/team_code/{birds_eye_view,speed_limits}` (map rasters and speed-limit tables), `simlingo/leaderboard`, `simlingo/data/augmented_templates` and the route sets `simlingo/data/simlingo/{training_1_scenario,training_3_scenarios}` (13,458 route XMLs).
- **Keep the folder inside the repository** so these relative links resolve.
- **Removed:** the leaderboard's documentation website, CARLA's `PythonAPI/util` and `examples`, and `scenario_runner`'s docs and metrics example logs were not needed for collection.

## Setup

**Requirements:**
- CARLA **0.9.15** and a Python client of the **same version** (`pip install carla==0.9.15`; a 0.9.16 client cannot talk to a 0.9.15 server);
- Python 3.10;
- a Logitech G923;
- Linux with a real display (WSLg works).

Without a wheel the agent falls back to the keyboard, which is fine for testing but not for recording. The rig uses its own environment, separate from the PersonaDrive environment:

```bash
pip install -r requirements.txt
```

Wheel permissions, once per machine. The agent reads the raw HID node, which is root-only by default and renumbered on every re-attach:

```bash
sudo tee /etc/udev/rules.d/99-logitech-g923.rules > /dev/null <<'EOF'
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c267", MODE="0660", GROUP="input"
SUBSYSTEM=="input", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c267", MODE="0660", GROUP="input"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG input "$USER"      # then log out and back in
```

`046d:c267` is the G923; substitute the IDs `lsusb` reports for another wheel.

Then set the two paths that cannot be bundled:

```bash
cp env.example.sh env.sh
$EDITOR env.sh      # CARLA_ROOT and DATASET_ROOT
source env.sh
```

Everything else defaults to the bundled copies. `env.example.sh` documents the optional overrides (`ARCHIVE_ROOT`, `ROUTES_DIRS`, `CACHE_DIR`, diagnostics).

## Running

```bash
"$CARLA_ROOT"/CarlaUE4.sh -quality-level=Epic   # separate terminal

source env.sh
python3 check_wheel.py                 # confirm the wheel reaches the agent (optional argument: seconds)
bash collect_session.sh debug 0        # smoke-test on the devtest route
bash collect_session.sh driver16 5     # record: 5 routes x 3 personas
```

`collect_session.sh <participant> <num_routes>` picks routes weighted toward globally undersampled scenario types, never repeating one already driven, and drives each three times.
- **Before each route:** it shows a bird's-eye preview.
- **Before each drive:** it prints the persona instruction to read aloud.
- **After each drive:** `R` redrives and any other key keeps the recording.
- **Resuming:** completion is tracked per persona, so an interrupted session resumes where it stopped.

Pass `0` for the devtest route or `long` for the long routes.

`run_scenario.sh [--day] <ScenarioType>` drives one route into `$SCRATCH_ROOT` for demos, leaving the dataset untouched; `--list` shows the types.

## Output

```
$DATASET_ROOT/<persona>/<participant>/<scenario_type>/<route>/
    result.json
    <Town>_Rep0_None/
        rgb/NNNN.jpg  rgb_augmented/NNNN.jpg  lidar/NNNN.laz
        boxes/NNNN.json.gz
        measurements/NNNN.json.gz    autopilot controls
        measurements/NNNN.json       human controls
```

Frames record every 5th tick: 4 Hz at CARLA's 20 fps.

- **`<participant>`:** use an anonymous ID of the form `driver<N>`. It ends up in every recorded path, and `convert_for_hpc.py` only picks up `driver*` folders.
- **Two sets of controls:**
  - the vehicle's motion (speed, pose, trajectory) is the human's driving;
  - `measurements/NNNN.json.gz` holds the privileged autopilot's controls for the same frame;
  - `measurements/NNNN.json` holds the human's wheel and pedal inputs.

## From recordings to the PersonaDrive dataset

**1. Language labels.** `label_gen/` holds SimLingo's VQA and commentary generators, patched for the layout above. To re-create the patched files from `simlingo/dataset_generation/`, run `python3 label_gen/apply_patches.py`; it is idempotent. The generators import `dataset_generation`, so put `$SIMLINGO_ROOT` on `PYTHONPATH`:

```bash
source env.sh
export LABELS_ROOT=/path/to/labels          # outside the repository
export PYTHONPATH="$SIMLINGO_ROOT:${PYTHONPATH:-}"
python3 label_gen/carla_vqa_generator_main.py        --data-directory "$DATASET_ROOT" --output-directory "$LABELS_ROOT/vqa"
python3 label_gen/carla_commentary_generator_main.py --data-directory "$DATASET_ROOT" --output-directory "$LABELS_ROOT/commentary"
```

**2. Convert to the PersonaDrive layout.** `convert_for_hpc.py` writes one tar per driver.
- **Layout changes:** it inserts the `routes/` level, collapses the per-frame `boxes/` and `measurements/` files into `boxes.json` / `measurements.json`, collapses the labels into `vqa.json` / `commentary.json`, and drops `lidar/`.
- **Controls:** `measurements.json` carries the **autopilot** controls from the `.json.gz` files, as in the released dataset.
- **Image paths:** the frame paths inside the labels are rewritten to `--hpc-root`.

```bash
python3 convert_for_hpc.py --src "$DATASET_ROOT" --vqa "$LABELS_ROOT/vqa" --commentary "$LABELS_ROOT/commentary" \
    --hpc-root "$PERSONADRIVE_DATA_ROOT/simlingo" --out /path/to/tars --drivers driver16
for t in /path/to/tars/driver*.tar; do tar -xf "$t" -C "$PERSONADRIVE_DATA_ROOT/simlingo"; done
```

The routes then sit at `$PERSONADRIVE_DATA_ROOT/simlingo/{data,commentary,drivelm}/simlingo/<persona>/driver<N>/routes/...`, the layout described in [`../datasets/README.md`](../datasets/README.md). To add them to a style's retrieval database, rebuild that style's index: Stages 1–2 in the [main README](../README.md#3-build-the-retrieval-indices).
