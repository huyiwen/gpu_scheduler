# GPU Scheduler

A very simple GPU tasks scheduler. Support for:

- minimal & maximal GPU amount to start a queued task
- set an estimated memory usage for each task to avoid out-of-memory errors
- dynamic polling of available GPUs with user-defined condition
- logging of stdout and stderr of each task

## Quick Start

```python
with GPUScheduler(min_gpu_count=1, max_gpu_count=2) as g:
    g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
    g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
    g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
    g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
```

Schedule four tasks requiring 1 GPU with 15GiB of memory. When maximal GPU count is reached, the scheduler will wait for a GPU to be freed. The first two tasks will start immediately, the other two will wait for a GPU to be freed.

## Installation

We use `nvitop` to get the GPU usage. `coloredlogs` is optional. Install them with:

```bash
pip install nvitop coloredlogs
```
