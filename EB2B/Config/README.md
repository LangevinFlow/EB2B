# EB2B Dataset Configuration

`superres_datasets.json` lists the benchmark datasets stored under `/mnt/DATA/SuperResolution`.
It provides:

- Repository roots (`data_root`, `output_root`).
- Default EB2B hyper-parameters shared across datasets.
- Output layout (LR/HR/Recon directories, whether to emit combined figures).
- Per-dataset entries specifying the CSV manifest, whether HR images are present, optional parameter overrides, and the directory that will host results.
- Optional `simdata_root` for the synthetic-data generator (used by `SimData/generate_sim_data.py`).

Use this JSON file as the single source of truth when extending EB2B to batch-process datasets.
