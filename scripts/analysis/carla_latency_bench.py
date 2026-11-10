#!/usr/bin/env python
"""
Measure client <-> CARLA server latency under evaluation-like conditions.

Mirrors the closed-loop loop in leaderboard/scenarios/scenario_manager.py: a
synchronous 20 Hz world, background traffic under the Traffic Manager, and the
agent's front camera (1024x512). No model is loaded, so these numbers are the
simulator and transport cost only; add scripts/analysis/benchmark_icl_timing.py
for inference. Use --inference-ms to hold the server idle as an agent would.

Per tick, each client records (milliseconds in the report):
    rpc         get_server_version(): acknowledged no-op, the transport round trip
    push_async  apply_control(): fire-and-forget, the client-side send cost only
    push_acked  apply_batch_sync(): the same control, waiting for the server's ack
    tick        world.tick(): send tick, server steps and renders, reply received
    frame       tick sent -> camera frame received (the data coming back)
    copy        raw_data -> numpy, the first thing an agent does with a frame

Probe overhead is measured, not assumed: rpc and push probes run on even ticks
only, so comparing tick time on even and odd ticks gives their cost.

Client-count topologies:
    shared  one CARLA server, N clients: one ticks, the rest observe. Each client
            adds a vehicle and a camera, so each adds a render pass.
    paired  N CARLA+client pairs on one GPU. Evaluation jobs run this way.

Subcommands:
    run     benchmark this node (boots CARLA, or --external-host to use one)
    serve   boot one CARLA server and block, for a client on another node
    submit  sbatch one `run` per GPU type
    report  merge results into summary.csv and summary.md
    client  one measuring client (started by `run`)

Examples:
    python scripts/analysis/carla_latency_bench.py run --clients 1,2,4 --out latency_results
    python scripts/analysis/carla_latency_bench.py submit --out latency_results
    python scripts/analysis/carla_latency_bench.py report --out latency_results

`run` and `client` need the personadrive env (carla==0.9.15). `submit` and
`report` only need the standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["client", "role", "tick_index", "probed", "rpc", "push_async", "push_acked",
          "tick", "frame", "copy", "frame_bytes"]
METRICS = ["rpc", "push_async", "push_acked", "tick", "frame", "copy"]
# (gres type, partition). Mirrors the pools in scripts/eval/start_eval_simlingo.py,
# on the free queues; override with --partition / --gpus.
GPU_POOLS = {"A30": "free-gpu", "A100": "free-gpu", "V100": "free-gpu",
             "L40S": "free-gpu32", "RTX6000": "free-gpu32"}
TEARDOWN_TIMEOUT_S = 30.0
BOOT_TIMEOUT_S = 900  # a cold boot from BeeGFS can take 5-8 minutes
RPC_TIMEOUT_S = 600.0  # --timeout in scripts/eval/start_eval_simlingo.py
# Below this much GPU memory per server, CARLA is almost certainly not rendering
# on the GPU (for example a Vulkan fallback to the CPU), and timings are not
# comparable across GPUs.
MIN_GPU_MEM_PER_SERVER_MB = 1000


# Client ---------------------------------------------------------------------

def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def timer_overhead_ns(samples=100_000):
    start = time.perf_counter()
    for _ in range(samples):
        time.perf_counter()
    return (time.perf_counter() - start) / samples * 1e9


def spawn_ego(world, spawn_points, client_id, tm_port, args, carla):
    """Spawn this client's vehicle with the agent's front camera."""
    library = world.get_blueprint_library()
    vehicle_bp = library.find("vehicle.lincoln.mkz_2020")
    # Required, as in the leaderboard: on large maps CARLA only keeps actors near a
    # "hero" awake, and putting this camera to sleep segfaults the server.
    vehicle_bp.set_attribute("role_name", "hero")
    ego = None
    for offset in range(len(spawn_points)):
        spawn = spawn_points[(client_id + offset) % len(spawn_points)]
        ego = world.try_spawn_actor(vehicle_bp, spawn)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("no free spawn point for ego vehicle")
    ego.set_autopilot(True, tm_port)

    camera_bp = library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", "110")
    mount = carla.Transform(carla.Location(x=-1.5, z=2.0))  # config_simlingo_base.py
    camera = world.spawn_actor(camera_bp, mount, attach_to=ego)
    frames = queue.Queue()
    camera.listen(lambda image: frames.put((time.perf_counter(), image)))
    return ego, camera, frames


def spawn_traffic(world, spawn_points, count, tm_port):
    # Nearest first, around the ego spawn: on a large map, distant vehicles sit in
    # unloaded tiles and add no load.
    origin = spawn_points[0].location
    spawn_points = sorted(spawn_points[1:], key=lambda spawn: spawn.location.distance(origin))
    library = world.get_blueprint_library().filter("vehicle.*")
    four_wheeled = [bp for bp in library if bp.get_attribute("number_of_wheels").as_int() == 4]
    vehicles = []
    for index, spawn in enumerate(spawn_points[:count]):
        bp = four_wheeled[index % len(four_wheeled)]
        vehicle = world.try_spawn_actor(bp, spawn)
        if vehicle is not None:
            vehicle.set_autopilot(True, tm_port)
            vehicles.append(vehicle)
    return vehicles


def next_frame(frames, min_frame, timeout):
    """Return (receive_time, image) for the first frame at or after min_frame."""
    deadline = time.perf_counter() + timeout
    while True:
        received, image = frames.get(timeout=max(0.001, deadline - time.perf_counter()))
        if image.frame >= min_frame:
            return received, image


def probe(client, ego, carla, row):
    """Time the send paths. get_control() reads client-side state, so no extra RPC."""
    control = ego.get_control()
    t0 = time.perf_counter()
    client.get_server_version()
    t1 = time.perf_counter()
    ego.apply_control(control)
    t2 = time.perf_counter()
    client.apply_batch_sync([carla.command.ApplyVehicleControl(ego.id, control)], False)
    t3 = time.perf_counter()
    row.update(rpc=t1 - t0, push_async=t2 - t1, push_acked=t3 - t2)


def receive(frames, min_frame, start, row, numpy):
    received, image = next_frame(frames, min_frame, RPC_TIMEOUT_S)
    t0 = time.perf_counter()
    numpy.frombuffer(image.raw_data, dtype=numpy.uint8).copy()
    row.update(frame=received - start, copy=time.perf_counter() - t0,
               frame_bytes=len(image.raw_data))


def tick_until(world, sync_dir, client_count):
    """Tick until every client has checked in, so nobody starves in sync mode."""
    deadline = time.time() + BOOT_TIMEOUT_S
    while len(list(sync_dir.glob("ready_*"))) < client_count:
        if time.time() > deadline:
            raise TimeoutError("timed out waiting for the other clients to attach")
        world.tick(RPC_TIMEOUT_S)


def run_ticker(args, client, world, ego, frames, carla, numpy, writer):
    for index in range(args.warmup + args.ticks):
        row = {"client": args.id, "role": "ticker", "tick_index": index,
               "probed": int(index % 2 == 0)}
        if row["probed"]:
            probe(client, ego, carla, row)
        t0 = time.perf_counter()
        frame_id = world.tick(RPC_TIMEOUT_S)
        row["tick"] = time.perf_counter() - t0
        receive(frames, frame_id, t0, row, numpy)
        if args.inference_ms:
            time.sleep(args.inference_ms / 1000)
        if index >= args.warmup:
            writer.writerow(row)


def run_observer(args, client, world, ego, frames, stop_file, carla, numpy, writer):
    index = 0
    while not stop_file.exists():
        try:
            snapshot = world.wait_for_tick(5.0)
            t0 = time.perf_counter()
            row = {"client": args.id, "role": "observer", "tick_index": index, "probed": 1}
            probe(client, ego, carla, row)
            receive(frames, snapshot.frame, t0, row, numpy)
        except (RuntimeError, queue.Empty):
            continue  # ticker finished or stalled; the stop file decides
        if index >= args.warmup:
            writer.writerow(row)
        index += 1


def release(client, world, actors, args, ticker, sync_dir, original_settings):
    """
    Tear down in the order the Traffic Manager needs. It throws on its own thread,
    killing the process, if a vehicle is destroyed while still registered with it.
    """
    vehicles = [actor for actor in actors if "vehicle" in actor.type_id]
    sensors = [actor for actor in actors if "sensor" in actor.type_id]
    for sensor in sensors:
        sensor.stop()
        sensor.destroy()
    for vehicle in vehicles:
        vehicle.set_autopilot(False, args.tm_port)
    if not ticker:
        for vehicle in vehicles:
            vehicle.destroy()
        (sync_dir / f"detached_{args.id}").touch()
        return
    if world is None or original_settings is None:
        return
    deadline = time.time() + 30
    while (len(list(sync_dir.glob("detached_*"))) < args.client_count - 1
           and time.time() < deadline):
        world.tick(TEARDOWN_TIMEOUT_S)
    world.tick(TEARDOWN_TIMEOUT_S)  # lets the Traffic Manager drop the vehicles
    for vehicle in vehicles:
        vehicle.destroy()
    client.get_trafficmanager(args.tm_port).set_synchronous_mode(False)
    world.apply_settings(original_settings)


def command_client(args):
    import carla  # pylint: disable=import-outside-toplevel
    import numpy  # pylint: disable=import-outside-toplevel

    sync_dir = Path(args.sync_dir)
    world_ready = sync_dir / f"world_ready_{args.group}"
    stop_file = sync_dir / f"stop_{args.group}"
    client = carla.Client(args.host, args.port)
    client.set_timeout(RPC_TIMEOUT_S)
    ticker = args.role == "ticker"
    actors = []
    original_settings = None
    world = None
    meta = {"client": args.id, "role": args.role, "timer_overhead_ns": timer_overhead_ns()}
    try:
        if ticker:
            log(f"loading {args.town}")
            t0 = time.perf_counter()
            world = client.load_world(args.town, reset_settings=False)
            meta["load_world_s"] = time.perf_counter() - t0
            original_settings = world.get_settings()
            # Create the Traffic Manager first: its setup needs the world to be
            # advancing, and in synchronous mode nothing is ticking yet.
            log("starting Traffic Manager")
            traffic_manager = client.get_trafficmanager(args.tm_port)
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / args.fps
            settings.tile_stream_distance = 650  # leaderboard_evaluator.py
            settings.actor_active_distance = 650
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_random_device_seed(args.seed)
            world.tick(RPC_TIMEOUT_S)
            log("reading map, spawning traffic")
            spawn_points = world.get_map().get_spawn_points()
            actors += spawn_traffic(world, spawn_points, args.npcs, args.tm_port)
            meta["npcs_spawned"] = len(actors)
            world_ready.touch()
        else:
            deadline = time.time() + BOOT_TIMEOUT_S
            while not world_ready.exists():
                if time.time() > deadline:
                    raise TimeoutError("ticker never loaded the world")
                time.sleep(0.2)
            world = client.get_world()
            spawn_points = world.get_map().get_spawn_points()

        log("spawning ego and camera")
        ego, camera, frames = spawn_ego(world, spawn_points, args.id, args.tm_port, args, carla)
        actors += [ego, camera]
        (sync_dir / f"ready_{args.id}").touch()

        out_path = Path(args.csv)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, FIELDS)
            writer.writeheader()
            if ticker:
                tick_until(world, sync_dir, args.client_count)
                log("measuring")
                start = time.perf_counter()
                run_ticker(args, client, world, ego, frames, carla, numpy, writer)
                meta["ticks_per_s"] = (args.warmup + args.ticks) / (time.perf_counter() - start)
                stop_file.touch()
            else:
                run_observer(args, client, world, ego, frames, stop_file, carla, numpy, writer)
        Path(args.csv).with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    finally:
        stop_file.touch()
        release(client, world, actors, args, ticker, sync_dir, original_settings)


# CARLA server lifecycle -----------------------------------------------------

def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def carla_root():
    root = os.environ.get("CARLA_ROOT") or str(REPO_ROOT / "software" / "carla0915")
    if not Path(root, "CarlaUE4.sh").exists():
        sys.exit(f"CarlaUE4.sh not found under {root}; set CARLA_ROOT")
    return root


def boot_server(port, adapter, log_path):
    """Boot CARLA in its own process group, so stopping it only touches this server."""
    for needed in (port, port + 1, port + 2):
        if not port_free(needed):
            sys.exit(f"port {needed} is in use; choose another --base-port")
    log = open(log_path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
    start = time.perf_counter()
    server = subprocess.Popen(  # pylint: disable=consider-using-with
        [f"{carla_root()}/CarlaUE4.sh", "-RenderOffScreen", "-nosound",
         f"-carla-rpc-port={port}", f"-carla-streaming-port={port + 1}",
         f"-carla-secondary-port={port + 2}", f"-graphicsadapter={adapter}"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    server.rpc_port = port
    while time.perf_counter() - start < BOOT_TIMEOUT_S:
        if server.poll() is not None:
            sys.exit(f"CARLA exited with {server.returncode} during boot; see {log_path}")
        if not port_free(port):
            return server, time.perf_counter() - start
        time.sleep(1)
    stop_server(server)
    sys.exit(f"CARLA did not open port {port} in {BOOT_TIMEOUT_S}s; see {log_path}")


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait()
    # The wrapper shell exits first; the UE binary can linger in I/O on BeeGFS.
    deadline = time.time() + 120
    while not port_free(server.rpc_port) and time.time() < deadline:
        time.sleep(1)


class GpuSampler(threading.Thread):
    """Samples GPU memory and utilisation; the peak shows whether CARLA used the GPU."""

    def __init__(self):
        super().__init__(daemon=True)
        self.halt = threading.Event()
        self.memory_mb = []
        self.utilization = []

    @staticmethod
    def read():
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=20).stdout
        if not output.strip():
            return None
        memory, utilization = output.splitlines()[0].split(",")
        return float(memory), float(utilization)

    def run(self):
        while not self.halt.wait(2.0):
            sample = self.read()
            if sample:
                self.memory_mb.append(sample[0])
                self.utilization.append(sample[1])


def gpu_name():
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return output.splitlines()[0].replace("NVIDIA ", "").strip()


# run ------------------------------------------------------------------------

def client_command(args, sync_dir, client_id, role, group, client_count, host, port, csv_path):
    return [sys.executable, str(Path(__file__).resolve()), "client",
            # Cases can reuse a server (--external-host), and it keeps the previous
            # Traffic Manager's port registered, so each case gets its own.
            "--host", host, "--port", str(port), "--tm-port", str(port + 50 + client_count),
            "--id", str(client_id), "--role", role, "--group", str(group),
            "--client-count", str(client_count), "--sync-dir", str(sync_dir),
            "--csv", str(csv_path), "--town", args.town, "--npcs", str(args.npcs),
            "--ticks", str(args.ticks), "--warmup", str(args.warmup), "--fps", str(args.fps),
            "--width", str(args.width), "--height", str(args.height),
            "--inference-ms", str(args.inference_ms), "--seed", str(args.seed)]


def run_case(args, out_root, topology, count, gpu):
    """One measurement: `count` clients in the given topology. Returns True on success."""
    case_dir = out_root / f"{gpu.replace(' ', '_')}_{socket.gethostname()}" / f"{topology}_n{count}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    sync_dir = case_dir / "sync"
    sync_dir.mkdir(parents=True)
    external = bool(args.external_host)
    host = args.external_host or "127.0.0.1"
    server_count = count if topology == "paired" else 1
    idle = GpuSampler.read()
    servers, boot_seconds = [], []
    workers = []
    sampler = GpuSampler()
    failed = False
    try:
        if not external:
            for index in range(server_count):
                server, seconds = boot_server(args.base_port + 100 * index, args.adapter,
                                              case_dir / f"carla_{index}.log")
                servers.append(server)
                boot_seconds.append(seconds)
        sampler.start()
        for index in range(count):
            paired = topology == "paired"
            port = args.base_port + 100 * index if paired else args.base_port
            role = "ticker" if paired or index == 0 else "observer"
            group = index if paired else 0
            command = client_command(args, sync_dir, index, role, group, count, host, port,
                                     case_dir / f"client_{index}.csv")
            with (case_dir / f"client_{index}.log").open("w", encoding="utf-8") as log:
                workers.append(subprocess.Popen(  # pylint: disable=consider-using-with
                    command, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.time() + args.case_timeout
        while any(worker.poll() is None for worker in workers):
            crashed = [index for index, server in enumerate(servers)
                       if server.poll() is not None]
            if crashed:
                print(f"CARLA server(s) {crashed} died; see {case_dir}", flush=True)
            if crashed or time.time() > deadline:
                failed = True
                break
            if any(worker.poll() not in (None, 0) for worker in workers):
                break
            time.sleep(1)
        failed = failed or any(worker.poll() not in (None, 0) for worker in workers)
    finally:
        sampler.halt.set()
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
        for server in servers:
            stop_server(server)

    idle_mb = idle[0] if idle else None
    peak_mb = max(sampler.memory_mb, default=None)
    gpu_used = None
    if not external and idle_mb is not None and peak_mb is not None:
        gpu_used = (peak_mb - idle_mb) / server_count >= MIN_GPU_MEM_PER_SERVER_MB
    meta = {"gpu": gpu, "node": socket.gethostname(), "topology": topology, "clients": count,
            "servers": server_count, "external_host": args.external_host, "town": args.town,
            "npcs": args.npcs, "width": args.width, "height": args.height,
            "inference_ms": args.inference_ms, "boot_s": boot_seconds,
            "gpu_idle_mb": idle_mb, "gpu_peak_mb": peak_mb,
            "gpu_util_mean": statistics.fmean(sampler.utilization) if sampler.utilization else None,
            "gpu_rendering_confirmed": gpu_used, "failed": failed}
    (case_dir / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(sync_dir, ignore_errors=True)
    verdict = "FAILED" if failed else "ok"
    print(f"[{verdict}] {gpu} {topology} clients={count} peak_gpu_mem={peak_mb}MB "
          f"-> {case_dir}", flush=True)
    return not failed


def command_run(args):
    out_root = Path(args.out).resolve()
    gpu = args.label or gpu_name()
    counts = [int(value) for value in args.clients.split(",")]
    topologies = args.topologies.split(",")
    if args.external_host and "paired" in topologies:
        sys.exit("paired boots its own servers; use --topologies shared with --external-host")
    results = []
    for topology in topologies:
        for count in counts:
            if topology == "paired" and count == 1 and "shared" in topologies:
                continue  # identical to shared with one client
            results.append(run_case(args, out_root, topology, count, gpu))
    sys.exit(0 if all(results) else 1)


def command_serve(args):
    log_path = Path(args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server, seconds = boot_server(args.base_port, args.adapter, log_path)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    print(f"CARLA up on {socket.gethostname()}:{args.base_port} after {seconds:.0f}s", flush=True)
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server(server)


# submit ---------------------------------------------------------------------

def sbatch_script(args, gpu, partition, out_root):
    python = os.environ.get("PERSONADRIVE_CONDA_ENV")
    python = f"{python}/bin/python" if python else "python"
    script = Path(__file__).resolve()
    account = f"#SBATCH --account={args.account}\n" if args.account else ""
    shared_args = (f"--town {args.town} --npcs {args.npcs} --ticks {args.ticks} "
                   f"--warmup {args.warmup} --width {args.width} --height {args.height} "
                   f"--inference-ms {args.inference_ms} --out {out_root}")
    header = f"""#!/bin/bash
#SBATCH --job-name=carla_latency_{gpu}{'_remote' if args.remote else ''}
#SBATCH --partition={partition}
{account}#SBATCH --nodes={2 if args.remote else 1}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --gres=gpu:{gpu}:1
#SBATCH --time={args.time}
#SBATCH -o {out_root}/slurm/%x_%j.log

set -euo pipefail
export CARLA_ROOT={carla_root()}
BASE_PORT=$((21000 + SLURM_JOB_ID % 200 * 100))
"""
    if not args.remote:
        return header + (f'{python} -u {script} run --label {gpu} --clients {args.clients} '
                         f'--topologies {args.topologies} --base-port "$BASE_PORT" {shared_args}\n')
    return header + f"""mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
srun --nodes=1 --ntasks=1 --nodelist="${{NODES[0]}}" {python} -u {script} serve \\
    --base-port "$BASE_PORT" --log {out_root}/slurm/carla_$SLURM_JOB_ID.log &
SERVER_STEP=$!
trap 'kill $SERVER_STEP 2>/dev/null || true' EXIT
for _ in $(seq 1 {BOOT_TIMEOUT_S}); do
    timeout 1 bash -c "</dev/tcp/${{NODES[0]}}/$BASE_PORT" 2>/dev/null && break
    sleep 1
done
srun --nodes=1 --ntasks=1 --nodelist="${{NODES[1]}}" {python} -u {script} run \\
    --label {gpu}_remote --external-host "${{NODES[0]}}" --topologies shared \\
    --clients {args.clients} --base-port "$BASE_PORT" {shared_args}
"""


def command_submit(args):
    out_root = Path(args.out).resolve()
    (out_root / "slurm").mkdir(parents=True, exist_ok=True)
    for gpu in args.gpus.split(","):
        partition = args.partition or GPU_POOLS.get(gpu)
        if partition is None:
            sys.exit(f"unknown GPU {gpu}; pass --partition")
        suffix = "_remote" if args.remote else ""
        path = out_root / "slurm" / f"{gpu}{suffix}.sbatch"
        path.write_text(sbatch_script(args, gpu, partition, out_root), encoding="utf-8")
        if args.dry_run:
            print(f"wrote {path}")
        else:
            output = subprocess.run(["sbatch", str(path)], capture_output=True, text=True,
                                    check=True).stdout
            print(f"{gpu}: {output.strip()}")


# report ---------------------------------------------------------------------

def percentile(ordered, fraction):
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarize_case(case_dir):
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    rows = []
    client_meta = []
    for path in sorted(case_dir.glob("client_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows += list(csv.DictReader(handle))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            client_meta.append(json.loads(meta_path.read_text(encoding="utf-8")))

    def values(metric, role="ticker", probed=None):
        return sorted(float(row[metric]) * 1000 for row in rows
                      if row[metric] and row["role"] == role
                      and (probed is None or row["probed"] == probed))

    result = {key: case[key] for key in ("gpu", "node", "topology", "clients")}
    result["ticker_samples"] = len(values("tick"))
    for metric in METRICS:
        ordered = values(metric)
        if ordered:
            result[f"{metric}_p50_ms"] = round(percentile(ordered, 0.50), 3)
            result[f"{metric}_p95_ms"] = round(percentile(ordered, 0.95), 3)
        observed = values(metric, "observer")
        if observed:
            result[f"observer_{metric}_p50_ms"] = round(percentile(observed, 0.50), 3)
            result[f"observer_{metric}_p95_ms"] = round(percentile(observed, 0.95), 3)
    with_probes, without_probes = values("tick", probed="1"), values("tick", probed="0")
    if with_probes and without_probes:
        result["probe_overhead_ms"] = round(statistics.median(with_probes)
                                            - statistics.median(without_probes), 3)
    rates = [meta["ticks_per_s"] for meta in client_meta if "ticks_per_s" in meta]
    if rates:
        result["ticks_per_s_per_client"] = round(statistics.fmean(rates), 2)
        result["ticks_per_s_total"] = round(sum(rates), 2)
    sizes = [int(row["frame_bytes"]) for row in rows if row["frame_bytes"]]
    if sizes:
        result["frame_mb"] = round(statistics.median(sizes) / 1e6, 3)
    overheads = [meta["timer_overhead_ns"] for meta in client_meta]
    if overheads:
        result["timer_overhead_ns"] = round(statistics.fmean(overheads), 1)
    if case["boot_s"]:
        result["boot_s"] = round(statistics.fmean(case["boot_s"]), 1)
    load_times = [meta["load_world_s"] for meta in client_meta if "load_world_s" in meta]
    if load_times:
        result["load_world_s"] = round(statistics.fmean(load_times), 1)
    result["gpu_peak_mb"] = case["gpu_peak_mb"]
    result["gpu_util_mean"] = case["gpu_util_mean"] and round(case["gpu_util_mean"], 1)
    result["gpu_rendering_confirmed"] = case["gpu_rendering_confirmed"]
    result["failed"] = case["failed"]
    return result


def command_report(args):
    out_root = Path(args.out).resolve()
    summaries = [summarize_case(path.parent) for path in sorted(out_root.glob("*/*/case.json"))]
    if not summaries:
        sys.exit(f"no results under {out_root}")
    summaries.sort(key=lambda row: (row["gpu"], row["topology"], row["clients"]))
    columns = []
    for row in summaries:
        columns += [key for key in row if key not in columns]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(summaries)

    shown = ["gpu", "topology", "clients", "rpc_p50_ms", "push_acked_p50_ms", "tick_p50_ms",
             "tick_p95_ms", "frame_p50_ms", "observer_rpc_p50_ms", "observer_push_acked_p50_ms",
             "ticks_per_s_per_client", "ticks_per_s_total", "probe_overhead_ms",
             "gpu_rendering_confirmed", "failed"]
    lines = ["| " + " | ".join(shown) + " |", "|" + " --- |" * len(shown)]
    lines += ["| " + " | ".join(str(row.get(key, "")) for key in shown) + " |"
              for row in summaries]
    bad = [f"{row['gpu']} {row['topology']} n={row['clients']}" for row in summaries
           if row["gpu_rendering_confirmed"] is False]
    if bad:
        lines += ["", "**GPU rendering not confirmed, do not compare these rows:** "
                  + "; ".join(bad)]
    text = "\n".join(lines) + "\n"
    (out_root / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_root / 'summary.csv'} and {out_root / 'summary.md'}")


# CLI ------------------------------------------------------------------------

def add_workload(parser):
    parser.add_argument("--town", default="Town12", help="Town12 has 104 of the 220 routes")
    parser.add_argument("--npcs", type=int, default=30, help="background vehicles")
    parser.add_argument("--ticks", type=int, default=600, help="measured ticks (30 s sim time)")
    parser.add_argument("--warmup", type=int, default=100, help="ticks discarded at the start")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20, help="evaluation runs at 20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-ms", type=float, default=0.0,
                        help="idle time per tick, standing in for the agent")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="benchmark this node")
    add_workload(run)
    run.add_argument("--clients", default="1,2,4", help="client counts to sweep")
    run.add_argument("--topologies", default="shared,paired")
    run.add_argument("--out", default="latency_results")
    run.add_argument("--label", help="GPU label; defaults to the nvidia-smi name")
    run.add_argument("--base-port", type=int, default=21000)
    run.add_argument("--adapter", type=int, default=0, help="CARLA -graphicsadapter")
    run.add_argument("--external-host", help="use a CARLA already serving on this host")
    run.add_argument("--case-timeout", type=int, default=1800)
    run.set_defaults(func=command_run)

    serve = commands.add_parser("serve", help="boot one CARLA server and block")
    serve.add_argument("--base-port", type=int, default=21000)
    serve.add_argument("--adapter", type=int, default=0)
    serve.add_argument("--log", default="latency_results/carla_serve.log")
    serve.set_defaults(func=command_serve)

    submit = commands.add_parser("submit", help="sbatch one run per GPU type")
    add_workload(submit)
    submit.add_argument("--gpus", default=",".join(GPU_POOLS))
    submit.add_argument("--clients", default="1,2,4")
    submit.add_argument("--topologies", default="shared,paired")
    submit.add_argument("--out", default="latency_results")
    submit.add_argument("--partition", help="override the partition for every GPU")
    submit.add_argument("--account", default=os.environ.get("PERSONADRIVE_SLURM_ACCOUNT_GPU"))
    submit.add_argument("--cpus", type=int, default=16)
    submit.add_argument("--mem", default="64G")
    submit.add_argument("--time", default="04:00:00")
    submit.add_argument("--remote", action="store_true",
                        help="CARLA on one node, clients on another (holds two GPUs)")
    submit.add_argument("--dry-run", action="store_true", help="write scripts, do not submit")
    submit.set_defaults(func=command_submit)

    report = commands.add_parser("report", help="merge results into summary.csv/.md")
    report.add_argument("--out", default="latency_results")
    report.set_defaults(func=command_report)

    client = commands.add_parser("client", help="one measuring client (started by run)")
    add_workload(client)
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, required=True)
    client.add_argument("--tm-port", type=int, required=True)
    client.add_argument("--id", type=int, required=True)
    client.add_argument("--role", choices=["ticker", "observer"], required=True)
    client.add_argument("--group", type=int, required=True, help="clients sharing a server")
    client.add_argument("--client-count", type=int, required=True)
    client.add_argument("--sync-dir", required=True)
    client.add_argument("--csv", required=True)
    client.set_defaults(func=command_client)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
