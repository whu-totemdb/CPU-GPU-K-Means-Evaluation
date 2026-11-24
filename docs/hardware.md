# Hardware and measurements

| Domain | Recorded conditions |
| --- | --- |
| Training benchmark | Dual Intel Xeon Platinum 8458P, 88 physical cores, 881 GiB host RAM, six NVIDIA H800 GPUs. One GPU per GPU run and four workers per CPU run. Capture host and device peaks separately. |
| Six-workload SeekDB study | Xeon/H800 host, CPU serving, four-CPU quota, 8 GiB memory. SIFT and Contriever ran serially. Road, Power, and Taxi had a distinct background-build condition. Power also has two separately reported serial blocks. |
| Matched-centroid and CPU replay | Lenovo ThinkStation P920, four workers, NUMA node 1, serial submission. For a local reproduction, select a valid four-CPU affinity on the target host and verify NUMA placement before timing. |

Training time spans initialization through the return of host-resident centroids. Quality measurement is separate. The FAISS worker logs search-call and client-loop times separately. SeekDB request timing has its own SQL boundary. Keep hardware domains, build repetitions, and the two executor timing boundaries explicit.
