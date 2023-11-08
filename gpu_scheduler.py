import subprocess
import time
import os
import signal
import threading
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime

from nvitop import Device

try:
    import coloredlogs
except ImportError:
    logging.basicConfig(level=logging.INFO)
else:
    coloredlogs.install("INFO")


class GPUScheduler:

    time_interval = 10
    logs_dir = "logs"
    global_min_mem_in_gib = 10

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

        self.gpus_used_by_proc: Dict[int, Tuple[subprocess.Popen, List[int], str, float]] = dict()
        self.estimated_memory_usage_per_gpu: Dict[int, float] = defaultdict(float)
        self.tasks = []

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        assert 1 <= self.min_gpu_count <= len(self.devices), f"min_gpu_count should be less than {len(self.devices)}"

    @property
    def _used_gpu_count(self):
        """Get the amount of GPU in use. Duplicate GPU will be counted only once."""
        gpus = set()
        for _, devices_list, _, _ in self.gpus_used_by_proc.values():
            gpus.update(devices_list)
        return len(gpus)

    def _wait_devices(self, min_mem=15000 * (1 << 20)) -> List[int]:
        """Await available devices.

        See more about nvitop:
            https://github.com/XuehaiPan/nvitop#callback-functions-for-machine-learning-frameworks
        """
        while True:
            available_gpus = [int(d.cuda_index) for d in self.devices if d.memory_free() >= min_mem + self.estimated_memory_usage_per_gpu[d.cuda_index]]
            self._wait_usable_devices()

            if len(available_gpus) >= self.min_gpu_count:
                return available_gpus[:self.min_gpu_count]
            else:
                time.sleep(self.time_interval)

    def _join_procs(self):
        """Wait all processes to finish"""
        while len(self.gpus_used_by_proc) > 0:
            time.sleep(self.time_interval)

    def _wait_usable_devices(self):
        """Wait maximum used gpu."""
        while self._used_gpu_count + self.min_gpu_count > self.max_gpu_count:
            time.sleep(self.time_interval)

    def _wait_when_error(self):
        try:
            signal.alarm(10)
            kill = input("KeyboardInterrupt: Input 'k' to kill all processes: ") == "k"
        except (Exception, KeyboardInterrupt):
            kill = False
        print()
        if kill:
            for  p, _, _, _ in self.gpus_used_by_proc.values():
                self.logger.info(f"Killing {p.pid}")
                p.kill()
        else:
            self.logger.warning(" Waiting for all processes to finish...")
            for p, _, _, _ in self.gpus_used_by_proc.values():
                p.wait()

    def _set_task_gpu(self, proc: subprocess.Popen, devices_list: List[int], task_name: str, estimated_memory_usage: float):
        """Set the GPU used by the process."""
        self.gpus_used_by_proc[proc.pid] = (proc, devices_list, task_name, estimated_memory_usage)
        for d in devices_list:
            self.estimated_memory_usage_per_gpu[d] += estimated_memory_usage

    def _unset_task_gpu(self, proc: subprocess.Popen):
        """Unset the GPU used by the process."""
        pid = proc.pid
        _, devices_list, task_name, estimated_memory_usage = self.gpus_used_by_proc[pid]
        for d in devices_list:
            self.estimated_memory_usage_per_gpu[d] -= estimated_memory_usage
        self.gpus_used_by_proc.pop(pid)
        self.logger.info(f"[{task_name}] {pid} Finished: {proc.args}")

        str_stdout = proc.stdout.read().decode('utf-8')
        with open(f"{self.logs_dir}/{task_name}_{pid}.log", "w") as f:
            f.write(str_stdout)
        self.logger.debug(str_stdout)

        str_stderr = proc.stderr.read().decode('utf-8')
        if len(str_stderr) > 0:
            with open(f"{self.logs_dir}/{task_name}_{pid}_stderr.log", "w") as f:
                f.write(str_stderr)
            self.logger.warning(str_stderr)

    def schedule(
        self,
        cmd_command: Union[List[str], str],
        min_mem_in_gib: Optional[float] = None,
        task_name: Optional[str] = None,
        estimated_memory_usage_in_gib: Optional[float] = None,
    ):
        """Add a task to the scheduler."""
        if isinstance(cmd_command, str):
            cmd_command = cmd_command.split(" ")
        if min_mem_in_gib is None:
            min_mem_in_gib = self.global_min_mem_in_gib
        min_mem = min_mem_in_gib * (1 << 30)
        if estimated_memory_usage_in_gib is None:
            estimated_memory_usage = min_mem
        else:
            estimated_memory_usage = estimated_memory_usage_in_gib * (1 << 30)
        self.tasks.append((cmd_command, min_mem, task_name, estimated_memory_usage))

    @staticmethod
    def popen_with_callback(on_start, on_exit, popen_kwargs):
        """
        Runs the given args in a subprocess.Popen, and then calls the function
        on_exit when the subprocess completes.
        on_exit is a callable object, and popen_args is a list/tuple of args that
        would give to subprocess.Popen.
        https://stackoverflow.com/questions/2581817/python-subprocess-callback-when-cmd-exits
        """
        def run_in_thread(on_start, on_exit, popen_kwargs):
            proc = subprocess.Popen(**popen_kwargs)
            on_start(proc)
            proc.wait()
            on_exit(proc)
            return
        thread = threading.Thread(target=run_in_thread, args=(on_start, on_exit, popen_kwargs))
        thread.start()

    def run(self):
        """Start to run all tasks."""
        for cmd_command, min_mem, task_name, estimated_memory_usage in self.tasks:
            devices_list = self._wait_devices(min_mem)

            cuda_visible_devices = ",".join(map(str, devices_list))
            if not task_name:
                task_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_C" + cuda_visible_devices + "_" + cmd_command[0].split("/")[-1]
            env = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}
            popen_kwargs = {"args": cmd_command, "env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

            def get_on_start(task_name, cmd_command, env, devices_list, estimated_memory_usage):
                def on_start(proc):
                    self.logger.info(f"STARTING {task_name}_{proc.pid} CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd_command)}")
                    self._set_task_gpu(proc, devices_list, task_name, estimated_memory_usage)
                return on_start

            def on_exit(proc):
                self._unset_task_gpu(proc)

            self.popen_with_callback(
                on_start=get_on_start(task_name, cmd_command, env, devices_list, estimated_memory_usage),
                on_exit=on_exit,
                popen_kwargs=popen_kwargs,
            )

        try:
            self._join_procs()
        except KeyboardInterrupt:
            self._wait_when_error()
        self.tasks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.tasks) > 0:
            self.run()


if __name__ == '__main__':

    with GPUScheduler(min_gpu_count=1, max_gpu_count=2) as g:
        g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
        g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
        g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)
        g.schedule(["/usr/bin/python3", "which_gpu_am_i.py"], 15)

