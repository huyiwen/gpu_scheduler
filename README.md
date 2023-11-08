# GPU Scheduler

A very simple GPU tasks scheduler. Support for:

- minimal & maximal GPU amount to start a queued task
- set an estimated memory usage for each task to avoid out-of-memory errors
- dynamic polling of available GPUs with user-defined condition
