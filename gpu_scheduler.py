import subprocess
import time
from collections import defaultdict
from typing import List, Tuple, Dict

from nvitop import Device


class GPUScheduler:

    time_interval = 10

    def __init__(
        self,
        min_gpu_count=1,
        max_gpu_count=None,
    ):
        self.devices = Device.cuda.all()
        self.min_gpu_count = min_gpu_count
        if max_gpu_count is None:
            print("Warning: max_gpu_count is not set, will use min_gpu_count instead.")
            self.max_gpu_count = min_gpu_count
        else:
            self.max_gpu_count = max_gpu_count

        self.gpus_used_by_proc: Dict[int, Tuple[subprocess.Popen, List[int]],  float] = dict()
        self.estimated_memory_usage_per_gpu: Dict[int, float] = defaultdict(float)
        self.tasks = []

        assert 1 <= self.min_gpu_count <= len(self.devices), f"min_gpu_count should be less than {len(self.devices)}"

    @property
    def _used_gpu_count(self):
        """Get the amount of GPU in use. Duplicate GPU will be counted only once."""
        gpus = set()
        for _, devices_list, _ in self.gpus_used_by_proc.values():
            gpus.update(devices_list)
        return len(gpus)

    def _wait_devices(self, min_mem=15000 * (1 << 20)) -> List[int]:
        """Await available devices.

        See more about nvitop:
            https://github.com/XuehaiPan/nvitop#callback-functions-for-machine-learning-frameworks
        """
        while True:
            available_gpus = [int(d.cuda_index) for d in self.devices if d.memory_free() >= min_mem + self.estimated_memory_usage_per_gpu[d.cuda_index]]
            self._wait_procs(False)

            if len(available_gpus) >= self.min_gpu_count:
                return available_gpus[:self.min_gpu_count]
            else:
                time.sleep(self.time_interval)

    def _wait_procs(self, wait_all: bool):
        """Wait all processes to finish or wait maximum used gpu."""
        if wait_all:
            wait_condition = lambda self: len(self.gpus_used_by_proc) == 0
        else:
            wait_condition = lambda self: self._used_gpu_count + self.min_gpu_count <= self.max_gpu_count

        while True:
            flag = True
            # print(wait_all, wait_condition(self))
            for pid, (proc, devices_list, estimated_memory_usage) in self.gpus_used_by_proc.items():
                # print(proc, devices_list, estimated_memory_usage)
                if proc.poll() is not None:
                    self._unset_task_gpu(pid)
                    flag = False
                    print("Finished:", proc.args)
                    print(proc.stdout.read().decode('utf-8'))
                    break  # break to avoid dictionary changed size during iteration
            if wait_condition(self):
                return
            if flag:
                time.sleep(10)

    def _set_task_gpu(self, proc: subprocess.Popen, devices_list: List[int], estimated_memory_usage: float):
        """Set the GPU used by the process."""
        self.gpus_used_by_proc[proc.pid] = (proc, devices_list, estimated_memory_usage)
        for d in devices_list:
            self.estimated_memory_usage_per_gpu[d] += estimated_memory_usage

    def _unset_task_gpu(self, pid: int):
        """Unset the GPU used by the process."""
        _, devices_list, estimated_memory_usage = self.gpus_used_by_proc[pid]
        for d in devices_list:
            self.estimated_memory_usage_per_gpu[d] -= estimated_memory_usage
        self.gpus_used_by_proc.pop(pid)

    def add(
        self,
        cmd_command: List[str],
        min_mem_in_mib: float,
        estimated_memory_usage_in_mib: float = 0
    ):
        """Add a task to the scheduler."""
        min_mem = min_mem_in_mib * (1 << 20)
        estimated_memory_usage = estimated_memory_usage_in_mib * (1 << 20)
        self.tasks.append((cmd_command, min_mem, estimated_memory_usage))

    def run(self):
        """Start to run all tasks."""
        for cmd_command, min_mem, estimated_memory_usage in self.tasks:
            devices_list = self._wait_devices(min_mem)

            env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, devices_list))}
            print("STARTING ==> CUDA_VISIBLE_DEVICES:", env["CUDA_VISIBLE_DEVICES"])
            proc = subprocess.Popen(cmd_command, env=env, stdout=subprocess.PIPE)
            self._set_task_gpu(proc, devices_list, estimated_memory_usage)

        self.tasks = []

    def __del__(self):
        if len(self.tasks) > 0:
            print("Warning: some processes are not executed. Use run() to execute them.")
        else:
            self._wait_procs(True)


if __name__ == '__main__':

    gpu_get = GPUScheduler(min_gpu_count=1, max_gpu_count=2)

    gpu_get.add(["/usr/bin/python3", "which_gpu_am_i.py"], 9000)
    gpu_get.add(["/usr/bin/python3", "which_gpu_am_i.py"], 9000)
    gpu_get.add(["/usr/bin/python3", "which_gpu_am_i.py"], 9000)
    gpu_get.add(["/usr/bin/python3", "which_gpu_am_i.py"], 9000)
    gpu_get.run()
