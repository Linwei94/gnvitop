"""Terminal UI for gnvitop using curses."""

import curses
import threading
import time


def _fmt_mb(mb):
    if mb >= 1024:
        return f"{mb / 1024:.1f}G"
    return f"{int(mb)}M"


def _bar(pct, width):
    filled = int(pct / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _usage_pair(pct):
    """Return color pair id based on usage percentage."""
    if pct < 50:
        return 1  # green
    if pct < 80:
        return 2  # yellow
    return 3       # red


def _temp_pair(temp):
    if temp < 50:
        return 1
    if temp < 75:
        return 2
    return 3


def _safe_addstr(stdscr, row, col, text, attr=0):
    """Add string without raising on boundary overflow."""
    h, w = stdscr.getmaxyx()
    if row >= h - 1 or col >= w:
        return
    available = w - col - 1
    if available <= 0:
        return
    try:
        stdscr.addstr(row, col, text[:available], attr)
    except curses.error:
        pass


def _draw(stdscr, data, loading, last_update, next_refresh, refresh_interval):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    GREEN  = curses.color_pair(1)
    YELLOW = curses.color_pair(2)
    RED    = curses.color_pair(3)
    CYAN   = curses.color_pair(4)
    WHITE  = curses.color_pair(5)
    BLUE   = curses.color_pair(6)
    MAGENTA = curses.color_pair(7)
    BOLD   = curses.A_BOLD

    # ── Title bar ──────────────────────────────────────────────────────────
    title = f" gnvitop \u2014 GPU Monitor (TUI)  [r]efresh  [q]uit "
    _safe_addstr(stdscr, 0, 0, title.ljust(w), CYAN | BOLD)

    # ── Status line ────────────────────────────────────────────────────────
    if loading:
        _safe_addstr(stdscr, 1, 0, " Fetching data...", YELLOW)
    else:
        ts = time.strftime("%H:%M:%S", time.localtime(last_update))
        remaining = max(0, int(next_refresh - time.time()))
        status = f" Updated: {ts}  |  Refresh in {remaining}s (interval: {refresh_interval}s)"
        _safe_addstr(stdscr, 1, 0, status, WHITE)

    # ── Separator ──────────────────────────────────────────────────────────
    _safe_addstr(stdscr, 2, 0, "\u2500" * (w - 1), curses.color_pair(8))

    row = 3

    if not data:
        _safe_addstr(stdscr, row, 2, "Waiting for data...", YELLOW)
        stdscr.refresh()
        return

    bar_w = max(10, min(24, (w - 72) // 2)) if w > 72 else 12

    for host in data:
        if row >= h - 1:
            break

        alias = host["alias"]
        status_str = host["status"]

        if host.get("is_local"):
            badge_attr = BLUE | BOLD
            badge = "[Local]"
        elif status_str == "ok":
            badge_attr = GREEN | BOLD
            badge = "[Online]"
        elif status_str == "no_gpu":
            badge_attr = YELLOW | BOLD
            badge = "[No GPU]"
        else:
            badge_attr = RED | BOLD
            badge = "[Offline]"

        # Host line
        host_label = f"  \u25b6 {alias}"
        _safe_addstr(stdscr, row, 0, host_label, WHITE | BOLD)
        badge_col = len(host_label) + 1
        _safe_addstr(stdscr, row, badge_col, badge, badge_attr)
        row += 1

        if status_str == "error":
            _safe_addstr(stdscr, row, 6, f"Error: {host.get('error', 'unknown')}", RED)
            row += 1
        elif status_str == "no_gpu":
            _safe_addstr(stdscr, row, 6, "No NVIDIA GPU detected", YELLOW)
            row += 1
        else:
            for gpu in host.get("gpus", []):
                if row >= h - 1:
                    break

                mem_pct = gpu["memory_usage_pct"]
                gpu_pct = gpu["gpu_utilization_pct"]
                temp    = gpu["temperature_c"]
                # Negative sentinel values mean "unknown" (e.g. TPUs, or GPUs
                # reporting [N/A] for a metric); show N/A instead of a bar.
                mem_known  = mem_pct >= 0
                gpu_known  = gpu_pct >= 0
                temp_known = temp >= 0
                mem_used  = _fmt_mb(gpu["memory_used_mb"])
                mem_total = _fmt_mb(gpu["memory_total_mb"])
                gpu_name  = gpu["name"]

                col = 0
                prefix = f"    GPU{gpu['index']} {gpu_name[:16]:<16} "
                _safe_addstr(stdscr, row, col, prefix, WHITE)
                col += len(prefix)

                # Memory bar
                _safe_addstr(stdscr, row, col, "MEM ", WHITE)
                col += 4
                if mem_known:
                    _safe_addstr(stdscr, row, col, _bar(mem_pct, bar_w),
                                 curses.color_pair(_usage_pair(mem_pct)))
                    col += bar_w
                    mem_str = f" {mem_used}/{mem_total} "
                    _safe_addstr(stdscr, row, col, mem_str, WHITE)
                    col += len(mem_str)
                else:
                    _safe_addstr(stdscr, row, col, "N/A".ljust(bar_w), YELLOW)
                    col += bar_w

                # GPU utilization bar
                _safe_addstr(stdscr, row, col, "GPU ", WHITE)
                col += 4
                if gpu_known:
                    _safe_addstr(stdscr, row, col, _bar(gpu_pct, bar_w),
                                 curses.color_pair(_usage_pair(gpu_pct)))
                    col += bar_w
                    gpu_str = f" {gpu_pct:3.0f}% "
                    _safe_addstr(stdscr, row, col, gpu_str,
                                 curses.color_pair(_usage_pair(gpu_pct)))
                    col += len(gpu_str)
                else:
                    _safe_addstr(stdscr, row, col, "N/A".ljust(bar_w), YELLOW)
                    col += bar_w

                # Temperature
                if temp_known:
                    temp_str = f"{int(temp)}\u00b0C"
                    _safe_addstr(stdscr, row, col, temp_str,
                                 curses.color_pair(_temp_pair(temp)))
                else:
                    _safe_addstr(stdscr, row, col, "N/A", YELLOW)

                row += 1

                # Users
                procs = gpu.get("processes", [])
                if procs and row < h - 1:
                    user_mem = {}
                    for p in procs:
                        u = p.get("user", "?")
                        user_mem[u] = user_mem.get(u, 0) + p.get("gpu_memory_mb", 0)
                    user_parts = [f"{u}({_fmt_mb(m)})" for u, m in user_mem.items()]
                    _safe_addstr(stdscr, row, 8, "  ".join(user_parts), MAGENTA)
                    row += 1

        # Blank line between hosts
        row += 1

    stdscr.refresh()


def _tui_main(stdscr, fetch_fn, refresh_interval):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_GREEN,   -1)
    curses.init_pair(2, curses.COLOR_YELLOW,  -1)
    curses.init_pair(3, curses.COLOR_RED,     -1)
    curses.init_pair(4, curses.COLOR_CYAN,    -1)
    curses.init_pair(5, curses.COLOR_WHITE,   -1)
    curses.init_pair(6, curses.COLOR_BLUE,    -1)
    curses.init_pair(7, curses.COLOR_MAGENTA, -1)
    curses.init_pair(8, curses.COLOR_BLACK + 8, -1)  # bright black = dark gray

    data = []
    last_update = 0.0
    loading = True
    _lock = threading.Lock()

    def do_fetch():
        nonlocal data, last_update, loading
        with _lock:
            loading = True
        result = fetch_fn()
        with _lock:
            data = result
            last_update = time.time()
            loading = False

    # Kick off initial fetch
    t = threading.Thread(target=do_fetch, daemon=True)
    t.start()

    next_refresh = time.time() + refresh_interval
    stdscr.timeout(500)  # poll every 0.5s for smooth countdown

    while True:
        key = stdscr.getch()

        if key in (ord('q'), ord('Q'), 27):  # q / Q / Esc
            break

        if key in (ord('r'), ord('R')):
            with _lock:
                if not loading:
                    t = threading.Thread(target=do_fetch, daemon=True)
                    t.start()
                    next_refresh = time.time() + refresh_interval

        # Auto-refresh
        with _lock:
            cur_loading = loading
        if time.time() >= next_refresh and not cur_loading:
            t = threading.Thread(target=do_fetch, daemon=True)
            t.start()
            next_refresh = time.time() + refresh_interval

        with _lock:
            snap_data     = list(data)
            snap_loading  = loading
            snap_last     = last_update

        _draw(stdscr, snap_data, snap_loading, snap_last, next_refresh, refresh_interval)


def run_tui(ssh_config_path=None, refresh_interval=30):
    """Launch the curses TUI dashboard."""
    import gnvitop.server as srv
    if ssh_config_path:
        srv.SSH_CONFIG_PATH = ssh_config_path

    # Import fetch function after potentially updating SSH_CONFIG_PATH
    from .server import fetch_all_gpu_info

    try:
        curses.wrapper(_tui_main, fetch_all_gpu_info, refresh_interval)
    except KeyboardInterrupt:
        pass
