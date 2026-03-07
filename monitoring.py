import threading
import psutil
import tracemalloc
import time
import os
import sys
import traceback
import signal
import logging

log = logging.getLogger()
THRESHOLD_MB = 55000  # raise/kill if RSS > this (MB)
CHECK_INTERVAL = 1.0  # seconds between checks
TOP_N = 10  # how many top memory "hog" traces to show


def bytes_to_mb(b):
    return b / 1024 / 1024


def report_top_traces(n=TOP_N):
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")
    log.error(f"Top {n} memory allocations (by traceback + line):")
    for i, stat in enumerate(stats[:n], 1):
        frame = stat.traceback[0]
        log.error(
            f"{i}. {frame.filename}:{frame.lineno} — {bytes_to_mb(stat.size):.2f} MB in {stat.count} blocks"
        )
        for line in stat.traceback.format()[:3]:
            log.error("    " + line)
    total = sum(s.size for s in stats)
    log.error(f"Snapshot total tracked by tracemalloc: {bytes_to_mb(total):.2f} MB")


def monitor_thread(threshold_mb=THRESHOLD_MB, check_interval=CHECK_INTERVAL):
    proc = psutil.Process(os.getpid())
    tracemalloc.start(25)
    try:
        while True:
            rss = proc.memory_info().rss
            rss_mb = bytes_to_mb(rss)
            if rss_mb > threshold_mb:
                log.error(
                    f"Memory threshold exceeded: {rss_mb:.2f} MB > {threshold_mb} MB"
                )
                try:
                    report_top_traces()
                except Exception:
                    log.error("Failed to collect tracemalloc snapshot:")
                    traceback.print_exc(file=sys.stderr)
                # attempt graceful termination signals first
                try:
                    # send SIGTERM to process group if available
                    os.killpg(os.getpid(), signal.SIGTERM)
                except Exception:
                    pass
                # fallback to immediate exit without cleanup (stops all threads)
                os._exit(1)
            time.sleep(check_interval)
    finally:
        tracemalloc.stop()


def start_monitor_in_background():
    t = threading.Thread(target=monitor_thread, name="memory-monitor", daemon=True)
    t.start()
    return t
