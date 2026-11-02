# Verification and troubleshooting

[← Back to the main README](../README.md)

## Verify your installation

[tests/README.md](../tests/README.md) has the full list and the expected output.

```bash
bash tests/static_checks.sh                                                   # no GPU, no data: syntax, paths, CLIs, config
python tests/smoke_test_eval_structure.py                                     # no GPU, no data
python tests/make_mini_database.py --out "$PD_TEST_ROOT/mini_database"         # 10 routes copied from your dataset root (~0.5 GB)
bash tests/run_smoke_pipeline.sh                                              # GPU: Stage 1a → 3b end to end
python tests/smoke_test_icl_k.py                                              # GPU: retrieval with the released indices
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found` (sentencepiece, Stage 1) | The env's own `lib/` is not on `LD_LIBRARY_PATH`. `conda activate` an env built by `environment/setup_env.sh` (it installs a hook), or `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`. |
| `MissingCUDAException: CUDA_HOME does not exist` when training starts | DeepSpeed needs a CUDA toolkit. The training env ships `nvcc` and sets `CUDA_HOME` on activation; otherwise load one, e.g. `module load cuda/12.x`. |
| `XLMRobertaModel.__init__() got an unexpected keyword argument 'dtype'` (Stage 1) | FlagEmbedding 1.4 or newer is installed; it needs a newer transformers than the env's 4.49. Install the pinned version: `pip install --no-deps FlagEmbedding==1.3.5`. |
| CARLA exits with `GameThread timed out waiting for RenderThread after 60.00 secs` / `Signal 11` | CARLA's Vulkan renderer could not start in this process context. We saw it inside `srun --overlap` job steps, while the same node works from a batch job or an interactive session. Run evaluation as a normal SLURM job or interactive shell. On multi-GPU nodes without device isolation, set `GPU_RANK` to the physical index of your GPU. |
| Evaluation jobs fail with `Agent couldn't be set up` or a setup timeout | The agent consolidated the DeepSpeed checkpoint during setup. Run `python scripts/eval/consolidate_checkpoint.py <ckpt_dir>` once; jobs then load `consolidated_fp32.pt`. |
| Retrieval is slow, or prints `[ICL] reading index frames from their original paths outside the data roots` | The index frames were found only at the absolute paths stored when the index was built. Set `PERSONADRIVE_DATA_ROOT` to a local copy of the database; the configured roots are searched first. |
| `pip` refuses to install the pinned requirements | Use `environment/setup_env.sh`. It installs the frozen pins with `--no-deps`; see [environment/README.md](../environment/README.md). |
