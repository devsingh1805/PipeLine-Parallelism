"""
tracker.py — Sends training metrics to the dashboard server

Each worker calls tracker.send() to report what it's doing.
The dashboard receives these and shows them in real time.

If the dashboard server is not running, this silently does nothing —
training continues normally without any crash.
"""

import json
import time
import urllib.request


class Tracker:
    def __init__(self, rank: int, world_size: int,
                 server_url: str = "http://localhost:5005"):
        self.rank       = rank
        self.world_size = world_size
        self.server_url = server_url

    def send(self, step: int, loss: float = None,
             phase: str = "step", extra: dict = None):
        payload = {
            "rank"      : self.rank,
            "world_size": self.world_size,
            "step"      : step,
            "loss"      : round(float(loss), 6) if loss is not None else None,
            "phase"     : phase,
            "timestamp" : round(time.time(), 3),
            "extra"     : extra or {},
        }
        self._post("/event", payload)

    def send_schedule(self, schedule_name, chunks, total_steps,
                      hidden_dim, total_layers):
        self._post("/config", {
            "schedule"    : schedule_name,
            "chunks"      : chunks,
            "total_steps" : total_steps,
            "hidden_dim"  : hidden_dim,
            "total_layers": total_layers,
            "world_size"  : self.world_size,
        })

    def send_timing(self, step: int, elapsed_ms: float):
        self._post("/timing", {
            "rank": self.rank, "step": step,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    def _post(self, path: str, payload: dict):
        try:
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                self.server_url + path, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(req, timeout=0.5)
        except Exception:
            pass   # Never crash training because of the tracker