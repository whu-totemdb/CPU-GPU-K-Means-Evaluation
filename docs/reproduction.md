# Reproduction guide

Run commands from the repository root. Paths under `inputs/`, `datasets/`, and `artifacts/` in these examples are caller-owned. Create local configurations from the documented contracts before running commands that reference those files.

## Training and input preparation

The CPU runner calls the shared configuration and dataset parser from the GPU runner, then dispatches through its CPU-specific registry. The GPU runner selects a device, starts a child with one visible GPU, and dispatches its configured adapter. Both read a case configuration plus a dataset registry and emit per-run JSONL on execution. The training tool records failures along with successful runs. Preserve initialization, dataset-slice, and run IDs.

```sh
python benchmarks/backend/run_cpu_experiments.py --config benchmarks/configs/cpu_example.json --data-root datasets
python benchmarks/backend/run_experiments.py --config benchmarks/configs/gpu_example.json --data-root datasets
```

Plan-only commands. On Linux, to execute your own configuration, pass `--execute` with a fresh `--output-dir`; the GPU runner also accepts `--gpu`. To relocate datasets use `--data-root` and `--registry`. The default registry provides registered formats and names. It intentionally omits download locations. External library installation and dataset authorizations are the caller's responsibility. The 118-task catalog is a task specification. It does not bind machine-specific binaries, historical array hashes, or run seeds.

The config writer creates a local plan for specific catalog tasks. It keeps the supplied data seed fixed while varying the explicitly supplied initialization seeds. For example, this command generates a config only:

```sh
python -B benchmarks/configs/make_training_config.py --candidate cpu:sklearn_lloyd:fp32:t4 --task boundary_gaussian_n1000000_d128_k1024 --data-seed 42 --init-seed 7 --init-seed 8 --init-seed 9 --output artifacts/example_plan.json
```

The seeds in this example are illustrative. Researchers should check whether the configured adapter actually consumes an initialization seed, and retain its seed/readback evidence before aggregating runs. Nine catalog tasks use three dataset IDs missing from the retained loader registry (`nyc_taxi`, `power`, and `roadnetwork`). These tasks need a caller-supplied registry and the correct historical representation. Pass the same `--registry` to the config writer and runner.

- `benchmarks/backend/prepare_superkmeans_input.py --help` is the GPU SuperKMeans data preparation helper. It accepts either HDF5 `train` data or a synthetic specification and emits FP32 plus provenance metadata.
- `benchmarks/assets/prepare_assets_v2.py --help` describes the prepared-asset worker. It reads a job from a local manifest, validates the split and identity bindings, and writes the base, 500+500 query arrays, and exact top-10 outputs. POSIX locks, Linux, a compatible FAISS build, h5py, and NumPy are required.
- `benchmarks/assets/materialize_exact_gt.py --help` describes the earlier exact-GT worker. Inputs are a frozen exact-GT plan and zero-norm audit. It emits neighbor arrays and metadata.
- `benchmarks/training/train_prepared_candidate.py --help` describes a hash-bound prepared-input execution worker. Its job manifest supplies the backend and effective arrays, input and source hashes, adapter options, and an owned output path. It does not create a replacement manifest. `quality_profiles_v2.py` and the legacy `profile_source_v1.py` follow that worker in the older prepared pipeline.

Training seed and build obligations are independent. The source material located for the training catalog includes a historical seed plan and previously aggregated tables. It does not provide the current manuscript's exact per-seed aggregation program with linked input runs.

## Cal3

### Metadata screening

`cal3/metadata_screening/screen.py` consumes the 15-configuration registry, development labels, static workload metadata, and a digest-bound fit receipt. Its nested family holdout, input whitelist, weighting, alpha selection, and baseline order are implemented in `develop`, `encode_metadata`, and `fit_ridge`. The paths in its adjacent `config.json` are relative to that module. Provide compatible local inputs at those paths before fitting. Fitted weights and development results are absent from this package.

```sh
python cal3/metadata_screening/screen.py --fit-development --freeze inputs/development_freeze.json --output artifacts/development_model
```

### Profiles and policies

```sh
python cal3/profile.py --config inputs/profile.json --output artifacts/profile.json
python cal3/selection.py --config inputs/selection.json --output artifacts/decision.json
```

Profile configuration lists relative files for `base`, `calibration_queries`, `ground_truth`, `centroids`, `base_order`, and `query_order`, keyed by the same names in `sha256`. Supply frozen row-order arrays and disjoint calibration/evaluation query ID lists. Ground truth contains the exact neighbor IDs in the same base coordinate system. Use the recorded target, metric, sample counts, and probe grid. The output contains geometry statistics and a row per prefix pair under `profiles`.

The selection stage takes three local row sets, named by `metadata_rows`, `profile_rows`, and optional `calibration_rows` in its JSON. Files can contain arrays of JSON rows or CSV. For each task, keep one profile row per candidate at the declared sample sizes and include eligibility evidence. Metadata rows need `candidate_id`, `compatible`, and `metadata_score`. Freeze `registry_order`, the full development `baseline_order`, `fixed_baseline` for the task, `M`, the trial budget, target and probe grid before calibration. The deployed method description uses `M=10`, up to three trials, and `ranking_field="estimated_scan"`. The earlier work-score replay uses `ranking_field="W"` and remains the compatibility default in the recovered module.

Calibration rows need an explicit `split="calibration"`, candidate ID and probe, qualifying recall and the frozen latency statistic. `calibration_prefix_summary` can form a summary from timed query rows using the median of round means. The policy code refuses evaluation-labeled rows at this stage. A traceable failed recall test is different from missing evidence.

## Retrieval execution

For Linux FAISS workers, provision the frozen effective arrays, assignments, list sizes, result-identity records, and calibration/evaluation assets under the local `--root`. The manifest binds task, candidate, grid, and worker identity. Match the local process CPU affinity to `--cpu-ids` and bind memory externally if reproducing the recorded NUMA protocol.

```sh
python faiss/evaluate_frozen.py --help
python faiss/evaluate_full_grid.py --help
```

The former evaluates only selected operating points; the full-grid worker supports the separately scoped curve study. The FAISS package must expose the required IVF list and search APIs. No FAISS or database worker was executed in this packaging review. See [the SeekDB boundary](../seekdb/README.md).

## Verified commands

The following were run in the review environment:

```sh
python -B -m unittest discover -s tests
python -B -m unittest discover -s cal3/metadata_screening
python -B cal3/profile.py --help
python -B cal3/selection.py --help
python -B cal3/metadata_screening/screen.py --help
```

The config-generation example and the Cal3 CLI were also run against temporary fixtures. The portable tests cover screening, missing evidence, profile estimates, and calibration without evaluation inputs. CPU and GPU example config schemas were checked without entering the execution workers. The checks started no training, CUDA work, or database services.
