# Runtime dependencies

The requirements list the portable Python dependencies. Select the native and CUDA libraries for the benchmark host. The lightweight fixtures were checked on Windows with Python 3.13.5 and NumPy 2.4.3. Physical runner validation requires Linux.

| Component | Observed or required version | Notes |
| --- | --- | --- |
| Historical P920 CPU stack | NumPy 1.26.4, scikit-learn 1.4.2, FAISS CPU 1.8.0, mlpack 4.8.0 | Recorded in the recovered deployment dependency file. Use a Python version supported by these builds. FAISS 1.8.0 covers this CPU domain. |
| H800 stack | PyTorch 2.7 recorded in the Flash adapter. Flash K-Means 0.3.0 and FastKMeans 0.5.0 recorded in the recovered deployment notes. | Use CUDA packages compatible with the target driver, Triton, `nvidia-smi`, and the matching Flash/FastKMeans versions. The archive also contains a CUDA 12.8 SuperKMeans build directory. That directory does not identify one universal CUDA version for all recorded runs. |
| Native implementations | Seed-aware `superkmeans` CPU binding, `mopbucket_py`, CUDA `gpu_generic_superkmeans.out`, `libKMCUDA`, and `cuml.cluster.KMeans` with CuPy | Install separately with the appropriate version, license, and hardware checks. Provide candidate-specific Python and library paths via runner configuration where supported. |
| GPU FAISS | CUDA-enabled FAISS with one visible device and an actual GpuIndex | Install independently. A CPU-only wheel cannot serve the GPU configuration. The package ports the route check from the recovered GPU correction. The original private frozen wrapper and GPU memory sampling evidence are separate from this portable entry point. |
| SeekDB | Compatible database source revision and deployment supporting approved external-centroid ingestion | A public version number alone does not identify the experimental build. The build, ABI, and flags require approval. |

Dependencies are installed independently. For license details consult the upstream source at the version used. The files below serve as entry points for that check:

- [FAISS license](https://github.com/facebookresearch/faiss/blob/main/LICENSE)
- [scikit-learn license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
- [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [mlpack distribution and license](https://github.com/mlpack/mlpack)

The historical adapters are Python integration code. Native bindings, GPU libraries, and database implementation source are not vendored.
