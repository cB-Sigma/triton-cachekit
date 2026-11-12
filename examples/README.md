# Examples

| File | What it shows |
| --- | --- |
| [`kernels.py`](kernels.py) | Two small kernels (row-wise RMS norm, SiLU) to prewarm. |
| [`prewarm.toml`](prewarm.toml) | A prewarm manifest with three specialisations for `cuda:80`. |
| [`simulate_deploy.sh`](simulate_deploy.sh) | prewarm, inspect, pack, unpack at another path, simulated `verify-hit`. Runs without a GPU. |
| [`verify_on_gpu.sh`](verify_on_gpu.sh) | What to run on a real GPU host to test a prewarmed cache. **Not yet run on real hardware by this project.** |

```console
$ pip install git+https://github.com/cb-Sigma/triton-cachekit triton
$ examples/simulate_deploy.sh
```

To find out why *your* application misses the cache, record two runs and compare them:

```console
$ triton-cachekit record -o before.jsonl -- python serve.py   # run that filled the cache
$ triton-cachekit record -o after.jsonl  -- python serve.py   # run that recompiled
$ triton-cachekit explain before.jsonl after.jsonl
```
