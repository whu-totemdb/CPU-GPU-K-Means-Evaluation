# CPU/GPU K-Means for IVF Indexing

Code accompanying *Modern CPU and GPU K-Means for IVF Indexing: Evaluation and Profile-Assisted Selection*.

This study evaluates 15 CPU and GPU K-Means configurations and examines how the returned centroids affect IVF retrieval. Cal3 screens configurations, estimates scan work from partition profiles, and calibrates recall and latency for up to three trial indexes, including a fixed baseline. The study uses SeekDB for database-level validation and FAISS to probe matched-centroid behavior.

This package contains source code, input schemas and configuration, tests, and documentation. Bring your own authorized input files. See [SeekDB integration](seekdb/README.md) for requirements specific to the database experiment.

## Architecture

1. **Training evaluation** runs CPU or GPU adapters in fresh child processes and records time, centroid quality, and resource use.
2. **Cal3** uses metadata-only screening, sampled estimates of list sizes and query routing, and calibration-only policy decisions.
3. **IVF validation** uses a shared partition when comparing FAISS and SeekDB. The FAISS worker loads external centroids, checks actual list assignments, and seals recall-qualified probes before reading the evaluation queries. See [SeekDB integration](seekdb/README.md) for the database-side dependencies.

## Repository structure

| Path | Purpose |
| --- | --- |
| `benchmarks/backend/` | CPU and GPU entry points and shared execution, registry, and quality contracts |
| `benchmarks/configs/` | Historical 118-workload catalog and two small example configurations |
| `benchmarks/assets/`, `benchmarks/training/` | Query splits, exact neighbors, prepared-input execution, and quality profiles |
| `cal3/` | Metadata screening, profiles, shortlist construction, and calibration decisions |
| `faiss/` | Matched-centroid calibration, deployment measurements, and full-grid variant |
| `tests/`, `docs/` | Lightweight tests, commands, provenance, and release status |

## Supported configurations

The registry is [frozen_candidate_registry.json](cal3/metadata_screening/inputs/frozen_candidate_registry.json). The registry describes the configurations studied. Each adapter still needs a compatible backend.

| Device | Code | Configuration | Precision | Policy |
| --- | --- | --- | --- | --- |
| CPU | FC | FAISS CPU | FP32 | Exact |
| CPU | HM | mlpack Hamerly | FP64 | Exact |
| CPU | MO | MopBucket | FP32 | Exact |
| CPU | EL | scikit-learn Elkan | FP32 | Exact |
| CPU | LL | scikit-learn Lloyd | FP32 | Exact |
| CPU | MB | scikit-learn MiniBatch | FP32 | Approximate |
| CPU | SC | SuperKMeans CPU | FP32 | Approximate |
| GPU | CU | cuML K-Means | FP32 | Exact |
| GPU | FG | FAISS GPU | FP32 | Exact |
| GPU | TR | FastKMeans with Triton | FP32 | Exact |
| GPU | F16 | Flash K-Means | FP16 | Approximate |
| GPU | F32 | Flash K-Means | FP32 | Exact |
| GPU | YY | KMCUDA Yinyang | FP32 | Exact |
| GPU | S03 | SuperKMeans GPU, sampling 0.3 | FP32 | Approximate |
| GPU | S10 | SuperKMeans GPU, sampling 1.0 | FP32 | Approximate |

## Hardware and software

The training study used a dual Intel Xeon Platinum 8458P host with 88 physical cores, 881 GiB RAM, and six NVIDIA H800 GPUs. Each CPU training job used four workers and each GPU job used one device. The original six-workload SeekDB experiment used a four-CPU quota and 8 GiB memory. Separate matched-centroid experiments used the P920 CPU host. See [hardware](docs/hardware.md) for the execution domains and [environment](docs/environment.md) for recorded versions and external dependencies.

## Installation

Set up a Python environment at the repository root:

```sh
python -m venv .venv
# Linux or macOS
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `\.venv\Scripts\Activate.ps1` from PowerShell. The equivalent Conda specification is `environment.yml`, intended to be used from the repository root. Linux is required for the physical experiment workers. Install the correct FAISS build and other native or CUDA dependencies in the runtime separately. See [environment](docs/environment.md).

## Running experiments

Run the lightweight checks first:

```sh
python -B -m unittest discover -s tests
python -B -m unittest discover -s cal3/metadata_screening
```

Use the runner examples to inspect the command and plan without training. These commands require Linux:

```sh
python benchmarks/backend/run_cpu_experiments.py --config benchmarks/configs/cpu_example.json --data-root datasets
python benchmarks/backend/run_experiments.py --config benchmarks/configs/gpu_example.json --data-root datasets
```

The examples describe small synthetic workloads. The [reproduction guide](docs/reproduction.md) lists the verified entry points, external inputs, expected outputs, and the distinction between inspected, fixture-tested, and hardware-dependent paths. Use the [workload catalog](benchmarks/configs/workloads_118.json) to locate each historical task.

## Results, data, and limitations

The distribution contains the experiment execution and selection code. Use local, authorized vectors and other inputs when running the workers. Results, analysis and plotting scripts, prepared data, indexes, and fitted model weights are outside this repository. The workload catalog retains task parameters only. Neither the historical source hashes nor the sample configurations establish the identity of a new run.

Training initialization seeds, repeated index builds with shared centroids, and timed query passes are separate units. The located files do not establish a complete per-seed-to-aggregate lineage for the current manuscript’s three-seed training summaries. The historical study also depends on specific hardware, native bindings, a compatible SeekDB build, and input snapshots. See the [paper map](docs/paper-map.md) for the code and input requirements.

## Citation

Cite the manuscript using [CITATION.cff](CITATION.cff). Publication details are pending.
