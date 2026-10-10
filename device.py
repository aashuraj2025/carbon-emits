"""Describe the machine you are running on and estimate its power draw while training."""
import os, platform, re, shutil, subprocess
from dataclasses import dataclass, asdict
from .core import Workload


@dataclass
class Device:
    cpu: str = ""
    cores: int = 0
    ram_gb: float = 0.0
    gpu: str = ""
    gpu_w: float = 0.0      # GPU power limit (W), 0 if none
    cpu_tdp: float = 45.0   # CPU package power class (W)
    os: str = ""


def guess_cpu_tdp(name: str) -> float:
    """Rough power class from the CPU model string. Override with --cpu-tdp if you know it."""
    n = name or ""
    if re.search(r"Apple M\d.*(Pro|Max|Ultra)", n): return 40
    if "Apple M" in n: return 20
    if re.search(r"\d{4,5}HX|HX\b", n): return 55
    if re.search(r"\d{4,5}(H|HS)\b|-H\b", n): return 45
    if re.search(r"\d{4,5}P\b", n): return 28
    if re.search(r"\d{4,5}U\b|-U\b|Ultra \d.*U\b", n): return 15
    if re.search(r"\d{4,5}K\w?\b|Threadripper|Ryzen 9 \d{4}X", n): return 125
    if re.search(r"\d{4,5}X\b|\d{4,5}F?\b.*Core|i[579]-\d{4,5}\b", n): return 65
    return 45


def _cpu_name() -> str:
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        return 0.0


def _gpu():
    if not shutil.which("nvidia-smi"):
        return "", 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader,nounits"],
            text=True, timeout=5).splitlines()[0]
        name, w = [x.strip() for x in out.split(",")]
        return name, float(w)
    except Exception:
        return "", 0.0


def detect() -> Device:
    cpu = _cpu_name()
    gpu, gw = _gpu()
    return Device(cpu=cpu, cores=os.cpu_count() or 0, ram_gb=_ram_gb(), gpu=gpu, gpu_w=gw,
                  cpu_tdp=guess_cpu_tdp(cpu), os=f"{platform.system()} {platform.release()}")


def device_watts(d: Device, cpu_load=0.5, gpu_load=0.0, charger=0.85, other=8.0) -> float:
    """Wall power (W): (CPU + GPU + RAM + screen/board) / charger efficiency."""
    return (d.cpu_tdp * cpu_load + d.gpu_w * gpu_load + d.ram_gb * 0.375 + other) / max(charger, 0.5)


def workload_from_device(d: Device, hours: float, grid: float = 700, use_gpu=None, **kw) -> Workload:
    """A Workload describing a run on this machine (local, so PUE = 1.0, no HPO)."""
    gpu = (d.gpu_w > 0) if use_gpu is None else use_gpu
    if gpu and d.gpu_w > 0:
        return Workload(gpu_tdp=d.gpu_w, gpus=1, hours=hours, util=0.8,
                        host_w=d.cpu_tdp * 0.3 + d.ram_gb * 0.375 + 8, pue=1.0, grid=grid, trials=0, **kw)
    return Workload(gpu_tdp=d.cpu_tdp, gpus=1, hours=hours, util=0.5,
                    host_w=d.ram_gb * 0.375 + 8, pue=1.0, grid=grid, trials=0, **kw)


def describe(d: Device) -> str:
    g = f", GPU {d.gpu} ({d.gpu_w:.0f} W)" if d.gpu else ", no NVIDIA GPU found"
    return f"{d.cpu} ({d.cores} threads, ~{d.cpu_tdp:.0f} W class), {d.ram_gb:.0f} GB RAM{g}, {d.os}"
