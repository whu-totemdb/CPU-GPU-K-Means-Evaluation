# SeekDB integration boundary

The manuscript’s database experiment starts from externally trained centroids and uses the same IVF-Flat assignment, list construction, and query execution path for every source. Native sampling and training are a separate path. A private build needs permission review before release.

An approved backend adapter must accept the effective centroid matrix and metric, confirm dimensions and finite values, load the centers into the index build, expose the serving probe setting, and return ranked IDs and measured latency for each query. It must attest to effective centroid identity and index readiness so FAISS and SeekDB can be paired on the same partition. For normalized inner product, use the same effective normalized inputs in both executors. Keep lifecycle costs and query timing boundaries separate.

The server patch, loading ABI and serializer, overlay tooling, deployment manifests, binaries, and the private SQL runner are outside this review package. A stock SeekDB installation is insufficient to reproduce the external-centroid experiments. The portable decision code remains available in [cal3/selection.py](../cal3/selection.py), and the separate FAISS worker is in [faiss/](../faiss/).
