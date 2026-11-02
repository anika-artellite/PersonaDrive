# PersonaDrive style dataset (v1)

The zips of [PervasiveAutonomyLab/PersonaDrive](https://huggingface.co/datasets/PervasiveAutonomyLab/PersonaDrive) on Hugging Face. Paths inside every zip start with `simlingo/`; extract them into `$PERSONADRIVE_DATA_ROOT`.

| File | Size | Files | SHA-256 | Contents |
|---|---:|---:|---|---|
| `data_default.zip` | 2.37 GB | 19,394 | `94dae5095b5617dff77e2a8057cba968385b671c5a1dfd7ee777d72cde8f601b` | Normal (neutral) driving, `default/driver*` |
| `data_fast.zip` | 11.69 GB | 81,033 | `7893b90671b7194b236a12551b882e234502eb626f9a20cdaef9d1438b083e95` | Aggressive driving, `fast/driver*` and `fast/lb1_split` (with `icl_examples.json.gz`) |
| `data_slow.zip` | 18.84 GB | 130,004 | `51b6f4778b2cf33d9742ff2ca031e697280a4e3e4e6c73cc299f8adb4f21bc44` | Conservative driving, `slow/driver*` and `slow/lb1_split` (with `icl_examples.json.gz`) |
| `commentary.zip` | 3.2 MB | 786 | `88844d8755cdc62cd6f203356c036b675a3cb5b57bd91377a36a09dd79e2cae9` | `commentary.json` per route, all three styles |
| `drivelm.zip` | 27.8 MB | 775 | `79362c38cc9a89fa2ee9649c82e8dfb74f325dbd12f3e3b6e97e4da260fdaf30` | `vqa.json` per route, all three styles |
