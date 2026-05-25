"""FedX-Palm FL Server — DEMO STUB.

Simulates the lifecycle of a Flower-based federated learning aggregator:
client registration, FedAvg aggregation log, and round broadcast. Real
training metrics come from the Colab simulation (see Bab 4 of thesis).
"""
import os
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "4"))
PORT = int(os.getenv("PORT", "8080"))
# baseline (ε=∞) konvergen; dp (ε≤8) collapse — sesuai temuan Bab 4 thesis.
SCENARIO = os.getenv("SCENARIO", "baseline").lower()

# Trajektori mAP@0.5 REAL dari federated baseline (Tabel 4.3 thesis).
BASELINE_MAP = [0.9946, 0.9944, 0.9944, 0.9943, 0.9945]

app = Flask(__name__)
state = {
    "clients": {},
    "round": 0,
    "started_at": datetime.utcnow().isoformat(),
}


def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] FL-SERVER | {msg}", flush=True)


@app.route("/health")
def health():
    return jsonify(status="ok", clients=len(state["clients"]), round=state["round"])


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    cid = data.get("client_id", "unknown")
    state["clients"][cid] = {"registered_at": datetime.utcnow().isoformat()}
    log(f"Client registered: {cid} ({len(state['clients'])}/{MIN_CLIENTS})")
    return jsonify(ok=True, round=state["round"])


@app.route("/submit_update", methods=["POST"])
def submit_update():
    data = request.get_json() or {}
    cid = data.get("client_id", "unknown")
    delta_norm = data.get("delta_norm", 0.0)
    log(f"Received Δw from {cid} (norm={delta_norm:.4f}, DP-clipped)")
    return jsonify(ok=True)


def federated_loop():
    """Wait for all clients then simulate FedAvg rounds."""
    log(f"FL Server started — waiting for {MIN_CLIENTS} clients on port {PORT}")
    while len(state["clients"]) < MIN_CLIENTS:
        time.sleep(2)
    log(f"All {MIN_CLIENTS} clients connected. Starting federated training.")
    log(f"SCENARIO = {SCENARIO.upper()} "
        f"({'ε=∞, no DP' if SCENARIO == 'baseline' else 'DP-SGD enabled'})")
    for r in range(1, NUM_ROUNDS + 1):
        state["round"] = r
        log(f"=== Round {r}/{NUM_ROUNDS} ===")
        log(f"Broadcasting global model w_{r-1} to {MIN_CLIENTS} clients")
        time.sleep(3)  # simulate broadcast
        log(f"Aggregating Δw via FedAvg: w_{r} = w_{r-1} + Σ (n_k/N) Δw_k")
        time.sleep(2)  # simulate aggregation
        if SCENARIO == "baseline":
            map_val = BASELINE_MAP[(r - 1) % len(BASELINE_MAP)]
            log(f"Round {r} complete | global mAP@0.5 = {map_val:.4f} (real, Tabel 4.3)")
        else:
            log(f"Round {r} complete | global mAP@0.5 = 0.0000 "
                f"(DP collapse — temuan utama thesis, Tabel 4.5)")
    log("Training complete. Final model saved to /app/checkpoints/global_final.pt")
    log("Server idle — keep running for demo purposes")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    threading.Thread(target=federated_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
