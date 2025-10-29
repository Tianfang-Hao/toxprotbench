"""
Compatibility shim for static analysis and tools.
Re-exports the real metrics helpers from src.metrics when available.
If src.metrics cannot be imported (e.g., in some analysis environments), this file
provides no-op fallbacks so scripts won't fail at import-time.
"""
from contextlib import contextmanager
import json
import time
import os

try:
    # Prefer the packaged implementation under src for runtime
    from src.metrics import write_metric, metric_timer  # type: ignore
except Exception:
    # Fallback no-op implementations for environments where src isn't on sys.path
    def write_metric(metric_file, step, data):
        try:
            if not metric_file:
                return
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(metric_file), exist_ok=True)
            record = {
                "step": step,
                "timestamp": int(time.time()),
                "data": data
            }
            with open(metric_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Silently ignore metric write failures
            return

    @contextmanager
    def metric_timer(metric_file, step, extra=None):
        start = time.time()
        try:
            yield
        finally:
            try:
                duration = time.time() - start
                write_metric(metric_file, step, {"duration_sec": float(duration), **(extra or {})})
            except Exception:
                pass

__all__ = ["write_metric", "metric_timer"]
