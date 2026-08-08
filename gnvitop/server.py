#!/usr/bin/env python3
"""GPU Monitor - Flask server that reads SSH config and queries remote GPUs."""

import getpass
import json
import os
import queue
import re
import shlex
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify, Response
import paramiko

from . import __version__
from .dashboard import DASHBOARD_HTML

app = Flask(__name__)

SSH_CONFIG_PATH = os.path.expanduser("~/.ssh/config")
SSH_TIMEOUT = 45

# ── nvidia-smi queries ────────────────────────────────────────────────────────
_GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null"
)
_PROC_QUERY = (
    r"nvidia-smi pmon -c 1 -s m 2>/dev/null | tail -n +3"
    r" | while read gpu pid type mem cmd; do"
    r' if [ "$pid" != "-" ]; then'
    r" user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ');"
    r" comm=$(ps -o comm= -p $pid 2>/dev/null | tr -d ' ');"
    r' [ "$mem" = "-" ] && mem=0;'
    r' printf "%s,%s,%s,%s,%s\n" "$pid" "$gpu" "$mem" "$user" "$comm";'
    r" fi; done"
)

# ── TPU queries (Google Cloud TPU) ───────────────────────────────────────────
_TPU_CHIP_QUERY = "ls /dev/accel* 2>/dev/null | wc -l"
_TPU_PROC_QUERY = (
    "ps -eo pid,user,comm 2>/dev/null"
    " | awk 'NR>1 && /python/ && !/awk/ {print $1\",0,0,\"$2\",\"$3}'"
    " | head -10"
)

# ── mx-smi queries (沐曦 MetaX GPUs) ─────────────────────────────────────────
_MX_PROC_QUERY = (
    r"mx-smi --show-all-process 2>/dev/null"
    r" | grep -E '^\|[[:space:]]+[0-9]'"
    r" | sed 's/|//g'"
    r" | awk '{print $1, $2, $NF}'"
    r" | while read gpu_idx pid mem; do"
    r' if [ -n "$pid" ]; then'
    r" user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ');"
    r' printf "%s,%s,%s,%s\n" "$gpu_idx" "$pid" "$mem" "$user";'
    r" fi; done"
)

# ── System memory query ──────────────────────────────────────────────────────
# Unified-memory GPUs (e.g. NVIDIA GB10 / DGX Spark) report [N/A] for all
# nvidia-smi memory fields; emit MemTotal + used (MemTotal - MemAvailable) in
# MiB so the parser can fall back to the unified system pool.
_SYS_MEM_QUERY = (
    "echo '---SYS---'"
    "; awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2}"
    " END{printf \"%d %d\", int(t/1024), int((t-a)/1024)}' /proc/meminfo"
)

# ── Auto-detect: try nvidia-smi first, fall back to mx-smi ───────────────────
# Output begins with "NVIDIA\n" or "MX\n" so the parser knows which format follows
COMBINED_CMD = (
    "if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then "
    "echo NVIDIA; " + _GPU_QUERY + "; echo '---SEP---'; " + _PROC_QUERY + "; " + _SYS_MEM_QUERY + "; "
    "elif command -v mx-smi >/dev/null 2>&1; then "
    "echo MX; mx-smi 2>/dev/null; echo '---SEP---'; " + _MX_PROC_QUERY + "; " + _SYS_MEM_QUERY + "; "
    "elif ls /dev/accel0 >/dev/null 2>&1; then "
    "echo TPU; " + _TPU_CHIP_QUERY + "; echo '---SEP---'; " + _TPU_PROC_QUERY + "; " + _SYS_MEM_QUERY + "; "
    "fi"
)

CURRENT_USER = getpass.getuser()

# System users to filter out from GPU process list
SYSTEM_USERS = frozenset({
    "root", "gdm", "lightdm", "sddm", "nvidia-persistenced",
    "Xorg", "gnome-shell",
})

cache = {"data": [], "last_update": 0}
cache_lock = threading.Lock()
CACHE_TTL = 30

# Background refresh state
_bg_refresh_running = False
_bg_refresh_lock = threading.Lock()


def parse_ssh_config(path):
    """Parse ~/.ssh/config and return a list of hosts."""
    hosts = []
    current = None

    if not os.path.exists(path):
        return hosts

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key_match = re.match(r"^(\w+)\s+(.+)$", line)
            if not key_match:
                continue

            key, value = key_match.group(1), key_match.group(2)

            if key.lower() == "host":
                if "*" in value or "?" in value:
                    current = None
                    continue
                current = {
                    "alias": value,
                    "hostname": None,
                    "user": None,
                    "port": 22,
                    "identity_file": None,
                    "proxy_jump": None,
                    "proxy_command": None,
                }
                hosts.append(current)
            elif current is not None:
                if key.lower() == "hostname":
                    current["hostname"] = value
                elif key.lower() == "user":
                    current["user"] = value
                elif key.lower() == "port":
                    current["port"] = int(value)
                elif key.lower() == "identityfile":
                    current["identity_file"] = os.path.expanduser(value)
                elif key.lower() == "proxyjump":
                    # Take only the first jump host (chained jumps not supported)
                    current["proxy_jump"] = value.split(",")[0].strip()
                elif key.lower() == "proxycommand":
                    current["proxy_command"] = value

    return hosts


def _parse_combined_output(output):
    """Split combined command output into (gpu_part, proc_part, sys_part, vendor).

    Output may begin with 'NVIDIA', 'MX', or 'TPU' to indicate accelerator vendor.
    Returns vendor as one of: 'nvidia', 'mx', 'tpu'.
    """
    vendor = "nvidia"
    lines = output.split("\n")
    if lines and lines[0].strip() == "MX":
        vendor = "mx"
        output = "\n".join(lines[1:])
    elif lines and lines[0].strip() == "NVIDIA":
        output = "\n".join(lines[1:])
    elif lines and lines[0].strip() == "TPU":
        vendor = "tpu"
        output = "\n".join(lines[1:])

    sys_part = ""
    if "---SYS---" in output:
        output, sys_part = output.split("---SYS---", 1)
    if "---SEP---" in output:
        gpu_part, proc_part = output.split("---SEP---", 1)
    else:
        gpu_part, proc_part = output, ""
    return gpu_part.strip(), proc_part.strip(), sys_part.strip(), vendor


def _safe_float(value, default=-1.0):
    """Parse a numeric CSV field, tolerating nvidia-smi's N/A placeholders.

    Returns default (a negative sentinel meaning 'unknown') when the field is
    empty, '[N/A]', '[Not Supported]', '-', or otherwise not a number.
    """
    v = value.strip().lower()
    if not v or v in ("n/a", "[n/a]", "na", "[not supported]", "[unknown]", "-", "--"):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _parse_sys_mem(sys_part):
    """Parse the '---SYS---' section into (total_mb, used_mb) or None."""
    try:
        vals = sys_part.split()
        if len(vals) >= 2 and float(vals[0]) > 0:
            return float(vals[0]), float(vals[1])
    except ValueError:
        pass
    return None


def _build_gpus(gpu_part, sys_mem_mb=None):
    """Parse GPU stats section into a list of GPU dicts.

    Unified-memory GPUs (GB10 / DGX Spark) report [N/A] for every memory field;
    when that happens, use the system memory figures passed in sys_mem_mb as the
    unified pool. Genuinely unknown values are -1 sentinels (same as TPU).
    """
    gpus = []
    for line in gpu_part.split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            mem_total = _safe_float(parts[2])
            mem_used = _safe_float(parts[3])
            mem_free = _safe_float(parts[4])
            utilization = _safe_float(parts[5])
            if mem_total < 0 and sys_mem_mb is not None:
                sys_total, sys_used = sys_mem_mb
                mem_total = sys_total
                mem_used = sys_used if sys_used >= 0 else -1
                mem_free = mem_total - mem_used if mem_used >= 0 else -1
            if mem_total > 0 and mem_used >= 0:
                mem_pct = round(mem_used / mem_total * 100, 1)
            else:
                mem_pct = -1
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": mem_total,
                "memory_used_mb": mem_used,
                "memory_free_mb": mem_free,
                "memory_usage_pct": mem_pct,
                "gpu_utilization_pct": utilization,
                "temperature_c": _safe_float(parts[6]),
                "processes": [],
            })
    return gpus


def _build_mx_gpus(output):
    """Parse default mx-smi table output into GPU dicts.

    Each GPU occupies two adjacent table rows, e.g.:
      | 0       MetaX C500  Off | 0000:0e:00.0 | 0%  Native |
      | 36C  56W / 350W  P0     | 858/65536 MiB | Available  |
    """
    import re
    gpus = []
    lines = output.split("\n")
    # Row 1: index, name, utilisation
    row1_re = re.compile(
        r"\|\s*(\d+)\s+([\w ]+?)\s+(?:Off|On)\s*\|[^|]+\|\s*(\d+)%"
    )
    # Row 2: temperature, mem_used / mem_total MiB
    row2_re = re.compile(
        r"\|\s*(\d+(?:\.\d+)?)C\s+[^|]+\|\s*(\d+)/(\d+)\s+MiB"
    )
    i = 0
    while i < len(lines):
        m1 = row1_re.search(lines[i])
        if m1 and i + 1 < len(lines):
            m2 = row2_re.search(lines[i + 1])
            if m2:
                mem_used = _safe_float(m2.group(2))
                mem_total = _safe_float(m2.group(3))
                gpus.append({
                    "index": int(m1.group(1)),
                    "name": m1.group(2).strip(),
                    "memory_total_mb": mem_total,
                    "memory_used_mb": mem_used,
                    "memory_free_mb": mem_total - mem_used if mem_total >= 0 and mem_used >= 0 else -1,
                    "memory_usage_pct": round(mem_used / mem_total * 100, 1) if mem_total > 0 and mem_used >= 0 else -1,
                    "gpu_utilization_pct": _safe_float(m1.group(3)),
                    "temperature_c": _safe_float(m2.group(1)),
                    "processes": [],
                })
                i += 2
                continue
        i += 1
    return gpus


def _attach_mx_processes(gpus, proc_output):
    """Parse mx-smi process output (gpu_idx,pid,mem,user,exp) and attach to GPUs."""
    gpu_by_index = {g["index"]: g for g in gpus}
    for line in proc_output.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            user = parts[3] if parts[3] else "unknown"
            if user in SYSTEM_USERS:
                continue
            gpu_idx = int(parts[0])
            proc = {
                "pid": int(parts[1]),
                "gpu_memory_mb": _safe_float(parts[2], default=0.0),
                "user": user,
                "command": "",
            }
            if gpu_idx in gpu_by_index:
                gpu_by_index[gpu_idx]["processes"].append(proc)


# ── TPU helpers ───────────────────────────────────────────────────────────────
_TPU_HBM_MB_PER_CHIP = {
    "v4": 32 * 1024,   # 32 GB HBM
    "v5e": 16 * 1024,  # 16 GB HBM
    "v6e": 32 * 1024,  # 32 GB HBM
}
_TPU_DEFAULT_HBM_MB = 32 * 1024


def _build_tpu_gpus(chip_count_output):
    """Build GPU-like dicts for each TPU chip.
    Memory is known spec; utilization is unknown until torch_xla is installed.
    Uses -1 as sentinel for 'unknown' in numeric fields.
    """
    try:
        num_chips = int(chip_count_output.strip().split()[0])
    except (ValueError, IndexError):
        num_chips = 4  # default for *-8 node
    hbm_mb = _TPU_DEFAULT_HBM_MB
    return [
        {
            "index": i,
            "name": "Google TPU v4",
            "memory_total_mb": hbm_mb,
            "memory_used_mb": -1,   # unknown without torch_xla
            "memory_free_mb": -1,
            "memory_usage_pct": -1,
            "gpu_utilization_pct": -1,
            "temperature_c": -1,
            "processes": [],
        }
        for i in range(num_chips)
    ]


def _attach_tpu_processes(gpus, proc_output):
    """Attach running Python processes to chip 0 (can't determine per-chip assignment)."""
    if not gpus or not proc_output.strip():
        return
    for line in proc_output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        user = parts[3] if len(parts) > 3 else "unknown"
        if user in SYSTEM_USERS:
            continue
        proc = {
            "pid": pid,
            "gpu_memory_mb": 0,
            "user": user,
            "command": parts[4] if len(parts) > 4 else "",
        }
        gpus[0]["processes"].append(proc)


def _make_ssh_client(hostname, port, user, identity_file, sock=None):
    """Create and connect a paramiko SSHClient."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = {
        "hostname": hostname,
        "port": port,
        "username": user,
        "timeout": SSH_TIMEOUT,
        "banner_timeout": SSH_TIMEOUT,
        "auth_timeout": SSH_TIMEOUT,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if identity_file:
        kwargs["key_filename"] = identity_file
    if sock is not None:
        kwargs["sock"] = sock
    client.connect(**kwargs)
    return client


def query_gpu(host_info, hosts_by_alias=None):
    """SSH into a host and query GPU information (single round trip).

    hosts_by_alias: dict of alias -> host_info for resolving ProxyJump targets.
    """
    alias = host_info["alias"]
    hostname = host_info["hostname"] or alias
    user = host_info["user"]
    port = host_info["port"]

    result = {
        "alias": alias,
        "hostname": hostname,
        "user": user or "unknown",
        "port": port,
        "status": "error",
        "error": None,
        "gpus": [],
    }

    jump_client = None
    try:
        sock = None
        proxy_alias = host_info.get("proxy_jump")
        proxy_cmd = host_info.get("proxy_command")
        if proxy_alias:
            # Resolve jump host info
            jump_info = (hosts_by_alias or {}).get(proxy_alias) or {
                "alias": proxy_alias,
                "hostname": proxy_alias,
                "user": None,
                "port": 22,
                "identity_file": None,
            }
            jump_host = jump_info.get("hostname") or proxy_alias
            jump_port = jump_info.get("port", 22)
            jump_user = jump_info.get("user")
            jump_key = jump_info.get("identity_file")
            jump_client = _make_ssh_client(jump_host, jump_port, jump_user, jump_key)
            sock = jump_client.get_transport().open_channel(
                "direct-tcpip", (hostname, port), ("", 0)
            )
        elif proxy_cmd:
            # ProxyCommand: execute the command and use its stdio as socket
            cmd = proxy_cmd.replace("%h", hostname).replace("%p", str(port))
            sock = paramiko.ProxyCommand(cmd)

        client = _make_ssh_client(hostname, port, user, host_info.get("identity_file"), sock=sock)

        # Single exec_command for both GPU stats and process info
        # Wrap in bash -c to avoid issues with non-bash login shells (e.g. fish)
        _, stdout, _ = client.exec_command("bash -c " + shlex.quote(COMBINED_CMD), timeout=SSH_TIMEOUT)
        output = stdout.read().decode("utf-8").strip()
        client.close()

        gpu_part, proc_part, sys_part, vendor = _parse_combined_output(output)
        sys_mem = _parse_sys_mem(sys_part)

        if not gpu_part:
            result["status"] = "no_gpu"
            result["error"] = "No supported GPU found (tried nvidia-smi, mx-smi, and TPU)"
        else:
            if vendor == "mx":
                gpus = _build_mx_gpus(gpu_part)
                if proc_part:
                    _attach_mx_processes(gpus, proc_part)
            elif vendor == "tpu":
                gpus = _build_tpu_gpus(gpu_part)
                if proc_part:
                    _attach_tpu_processes(gpus, proc_part)
                result["is_tpu"] = True
            else:
                gpus = _build_gpus(gpu_part, sys_mem)
                if proc_part:
                    _attach_processes(gpus, proc_part)
            result["gpus"] = gpus
            if gpus:
                result["status"] = "ok"
            else:
                result["status"] = "no_gpu"
                result["error"] = "No valid accelerator data returned"

    except paramiko.AuthenticationException:
        result["error"] = "Authentication failed"
    except paramiko.SSHException as e:
        result["error"] = f"SSH error: {e}"
    except TimeoutError:
        result["error"] = "Connection timed out"
    except OSError as e:
        result["error"] = f"Connection failed: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if jump_client:
            try:
                jump_client.close()
            except Exception:
                pass

    return result


def _attach_processes(gpus, proc_output):
    """Parse process output and attach to matching GPUs (skip system users)."""
    gpu_by_index = {g["index"]: g for g in gpus}
    for line in proc_output.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            user = parts[3] if parts[3] else "unknown"
            if user in SYSTEM_USERS:
                continue
            gpu_idx = int(parts[1])
            proc = {
                "pid": int(parts[0]),
                "gpu_memory_mb": _safe_float(parts[2], default=0.0),
                "user": user,
                "command": parts[4] if len(parts) >= 5 else "",
            }
            if gpu_idx in gpu_by_index:
                gpu_by_index[gpu_idx]["processes"].append(proc)


def query_local_gpu():
    """Query the local machine for GPU information (single subprocess call)."""
    import socket

    hostname = socket.gethostname()
    result = {
        "alias": "localhost",
        "hostname": hostname,
        "user": CURRENT_USER,
        "port": 0,
        "status": "error",
        "error": None,
        "gpus": [],
        "is_local": True,
    }

    try:
        output = subprocess.run(
            COMBINED_CMD, shell=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()

        gpu_part, proc_part, sys_part, vendor = _parse_combined_output(output)
        sys_mem = _parse_sys_mem(sys_part)

        if not gpu_part:
            result["status"] = "no_gpu"
            result["error"] = "No supported GPU found (tried nvidia-smi, mx-smi, and TPU)"
        else:
            if vendor == "mx":
                gpus = _build_mx_gpus(gpu_part)
                if proc_part:
                    _attach_mx_processes(gpus, proc_part)
            elif vendor == "tpu":
                gpus = _build_tpu_gpus(gpu_part)
                if proc_part:
                    _attach_tpu_processes(gpus, proc_part)
                result["is_tpu"] = True
            else:
                gpus = _build_gpus(gpu_part, sys_mem)
                if proc_part:
                    _attach_processes(gpus, proc_part)
            result["gpus"] = gpus
            if gpus:
                result["status"] = "ok"
            else:
                result["status"] = "no_gpu"
                result["error"] = "No valid accelerator data returned"

    except FileNotFoundError:
        result["status"] = "no_gpu"
        result["error"] = "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        result["error"] = "nvidia-smi timed out"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _sort_results(results):
    order = {"ok": 0, "no_gpu": 1, "error": 2}
    results.sort(key=lambda x: (
        0 if x.get("is_local") else 1,
        order.get(x["status"], 3),
        -len(x.get("gpus", [])),
        x["alias"],
    ))
    return results


def discover_gadi_nodes(hosts_by_alias):
    """SSH to each gadi-like login node and discover allocated GPU compute nodes via qstat.

    Returns a list of host_info dicts for any active job nodes found.
    A host is treated as a Gadi login node if its HostName ends with '.nci.org.au'
    and it has no ProxyJump (i.e. it is itself the jump host).
    """
    discovered = []
    for alias, info in hosts_by_alias.items():
        hostname = info.get("hostname") or alias
        if not hostname.endswith(".nci.org.au"):
            continue
        if info.get("proxy_jump"):
            continue  # skip compute nodes, only query login nodes
        # SSH to login node and run qstat to find allocated nodes
        try:
            client = _make_ssh_client(
                hostname, info.get("port", 22),
                info.get("user"), info.get("identity_file"),
            )
            _, stdout, _ = client.exec_command(
                "qstat -u $(whoami) -n 2>/dev/null | grep -oE 'gadi-gpu-[a-z0-9-]+' | sort -u",
                timeout=SSH_TIMEOUT,
            )
            node_names = [n.strip() for n in stdout.read().decode().splitlines() if n.strip()]
            client.close()
        except Exception:
            continue

        for node in node_names:
            node_hostname = f"{node}.gadi.nci.org.au"
            discovered.append({
                "alias": node,
                "hostname": node_hostname,
                "user": info.get("user"),
                "port": 22,
                "identity_file": info.get("identity_file"),
                "proxy_jump": alias,
            })

    return discovered


def fetch_all_gpu_info():
    """Query all hosts (local + remote) concurrently and return sorted results."""
    hosts = parse_ssh_config(SSH_CONFIG_PATH)
    hosts_by_alias = {h["alias"]: h for h in hosts}

    # Discover dynamically allocated Gadi compute nodes
    dynamic = discover_gadi_nodes(hosts_by_alias)
    for h in dynamic:
        if h["alias"] not in hosts_by_alias:
            hosts.append(h)
            hosts_by_alias[h["alias"]] = h

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(query_local_gpu): None}
        for h in hosts:
            futures[executor.submit(query_gpu, h, hosts_by_alias)] = h
        for future in as_completed(futures):
            results.append(future.result())

    return _sort_results(results)


def _do_background_refresh():
    """Run fetch_all_gpu_info and update cache; reset _bg_refresh_running flag when done."""
    global _bg_refresh_running
    try:
        data = fetch_all_gpu_info()
        with cache_lock:
            cache["data"] = data
            cache["last_update"] = time.time()
    finally:
        with _bg_refresh_lock:
            _bg_refresh_running = False


def _trigger_background_refresh():
    """Spawn a background refresh thread if one isn't already running."""
    global _bg_refresh_running
    with _bg_refresh_lock:
        if _bg_refresh_running:
            return
        _bg_refresh_running = True
    t = threading.Thread(target=_do_background_refresh, daemon=True)
    t.start()


def _start_background_warmer():
    """Background thread that keeps cache warm by refreshing every CACHE_TTL seconds."""
    def _warmer():
        # Initial warm-up: start immediately so first page load hits cached data
        _do_background_refresh()
        while True:
            time.sleep(CACHE_TTL)
            _do_background_refresh()

    t = threading.Thread(target=_warmer, daemon=True)
    t.start()


@app.route("/")
def index():
    import socket
    host_info = f"{getpass.getuser()}@{socket.gethostname()}"
    html = DASHBOARD_HTML.replace("{{GNVITOP_HOST_INFO}}", host_info).replace("{{GNVITOP_VERSION}}", __version__)
    return Response(html, mimetype="text/html")


@app.route("/api/gpus")
def api_gpus():
    """Return cached data immediately; trigger background refresh if cache is stale."""
    now = time.time()
    with cache_lock:
        data = cache["data"]
        last_update = cache["last_update"]

    if now - last_update > CACHE_TTL:
        _trigger_background_refresh()

    return jsonify({"hosts": data, "updated_at": last_update})


@app.route("/api/refresh")
def api_refresh():
    """Force a synchronous refresh and return fresh data."""
    with cache_lock:
        cache["data"] = fetch_all_gpu_info()
        cache["last_update"] = time.time()
        return jsonify({"hosts": cache["data"], "updated_at": cache["last_update"]})


@app.route("/api/stream")
def api_stream():
    """SSE endpoint: streams each host result as it arrives, then a 'done' event."""
    def generate():
        hosts = parse_ssh_config(SSH_CONFIG_PATH)
        hosts_by_alias = {h["alias"]: h for h in hosts}
        results = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(query_local_gpu): None}
            for h in hosts:
                futures[executor.submit(query_gpu, h, hosts_by_alias)] = h

            for future in as_completed(futures):
                host_result = future.result()
                results.append(host_result)
                payload = json.dumps({"host": host_result})
                yield f"data: {payload}\n\n"

        # Update cache with fresh streamed data
        sorted_results = _sort_results(results)
        with cache_lock:
            cache["data"] = sorted_results
            cache["last_update"] = time.time()

        yield f"data: {json.dumps({'done': True, 'updated_at': cache['last_update']})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
