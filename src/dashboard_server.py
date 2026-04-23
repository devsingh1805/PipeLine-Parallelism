"""
dashboard_server.py — Receives metrics from workers, streams to browser

Run this BEFORE training:
    python dashboard_server.py

Then open: http://localhost:5005
Then run training: torchrun --nproc-per-node=4 src/main.py
"""

import json, os, queue, threading, time
from collections import defaultdict
from flask import Flask, Response, request, jsonify, send_from_directory

app = Flask(__name__)
lock        = threading.Lock()
subscribers = []

state = {
    "config"      : {},
    "events"      : [],
    "loss_history": [],
    "rank_phases" : defaultdict(list),
    "step_times"  : [],
    "last_step"   : 0,
    "started_at"  : None,
    "finished"    : False,
}

@app.route("/config", methods=["POST"])
def receive_config():
    data = request.get_json()
    with lock:
        state["config"]     = data
        state["started_at"] = time.time()
    _push("config", data)
    return jsonify(ok=True)

@app.route("/event", methods=["POST"])
def receive_event():
    data = request.get_json()
    with lock:
        state["events"].append(data)
        if data.get("loss") is not None:
            state["loss_history"].append({"step": data["step"], "loss": data["loss"]})
        state["rank_phases"][str(data["rank"])].append({
            "step": data["step"], "phase": data["phase"], "ts": data["timestamp"]
        })
        state["last_step"] = max(state["last_step"], data["step"])
        if data.get("phase") == "done":
            state["finished"] = True
    _push("event", data)
    return jsonify(ok=True)

@app.route("/timing", methods=["POST"])
def receive_timing():
    data = request.get_json()
    with lock:
        state["step_times"].append(data)
    _push("timing", data)
    return jsonify(ok=True)

@app.route("/reset", methods=["POST"])
def reset():
    with lock:
        state.update({
            "config": {}, "events": [], "loss_history": [],
            "rank_phases": defaultdict(list), "step_times": [],
            "last_step": 0, "started_at": None, "finished": False,
        })
    _push("reset", {})
    return jsonify(ok=True)

@app.route("/snapshot")
def snapshot():
    with lock:
        return jsonify({
            "config"      : state["config"],
            "loss_history": state["loss_history"],
            "rank_phases" : dict(state["rank_phases"]),
            "step_times"  : state["step_times"],
            "last_step"   : state["last_step"],
            "finished"    : state["finished"],
            "event_count" : len(state["events"]),
        })

@app.route("/stream")
def stream():
    def generate():
        yield 'data: {"type":"connected"}\n\n'
        sub = queue.Queue()
        subscribers.append(sub)
        try:
            while True:
                try:
                    yield f"data: {sub.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield 'data: {"type":"heartbeat"}\n\n'
        except GeneratorExit:
            try: subscribers.remove(sub)
            except ValueError: pass
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "dashboard.html")

def _push(event_type, data):
    msg = json.dumps({"type": event_type, "data": data})
    for sub in subscribers[:]:
        try: sub.put_nowait(msg)
        except: pass

if __name__ == "__main__":
    print("\n  micropp Dashboard → http://localhost:5005")
    print("  Run training:  torchrun --nproc-per-node=4 src/main.py\n")
    app.run(host="0.0.0.0", port=5005, threaded=True, debug=False)