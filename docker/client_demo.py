"""FedX-Palm FL Client — DEMO STUB.

Simulates a single client node lifecycle: connect to server, run local
fine-tuning rounds, apply DP injection, and submit weight delta. Real
training results come from the Colab simulation (see Bab 4 of thesis).
"""
import os
import time
import random
import requests
from datetime import datetime

CLIENT_ID = os.getenv("CLIENT_ID", "client_unknown")
SERVER = os.getenv("SERVER", "fl-server:8080")
DIRICHLET_ALPHA = float(os.getenv("DIRICHLET_ALPHA", "0.5"))
# baseline = tanpa DP (ε=∞); dp = DP-SGD aktif (σ real moderate = 0.010).
SCENARIO = os.getenv("SCENARIO", "baseline").lower()
DP_EPSILON = float(os.getenv("DP_EPSILON", "4.0"))
DP_SIGMA = float(os.getenv("DP_SIGMA", "0.010"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))


def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {CLIENT_ID:>12s} | {msg}", flush=True)


def register():
    while True:
        try:
            r = requests.post(f"http://{SERVER}/register",
                              json={"client_id": CLIENT_ID}, timeout=5)
            if r.ok:
                eps = "∞" if SCENARIO == "baseline" else DP_EPSILON
                log(f"Registered with server {SERVER} (α={DIRICHLET_ALPHA}, ε={eps})")
                return
        except Exception as e:
            log(f"Waiting for server {SERVER}... ({e.__class__.__name__})")
        time.sleep(3)


def fake_local_training():
    """Simulates local fine-tuning on Non-IID data + optional DP injection."""
    log(f"Loading Non-IID dataset (Dirichlet α={DIRICHLET_ALPHA})")
    time.sleep(2)
    for ep in range(1, LOCAL_EPOCHS + 1):
        if SCENARIO == "baseline":
            loss = 0.5 / ep + random.uniform(-0.02, 0.02)
        else:
            # DP noise mendestabilkan loss → divergen (temuan collapse).
            loss = 0.5 / ep + ep * random.uniform(1.5, 3.0)
        log(f"  Local epoch {ep}/{LOCAL_EPOCHS} | box_loss={loss:.4f}")
        time.sleep(2)
    if SCENARIO == "baseline":
        log("Privacy: OFF (baseline ε=∞) — no gradient noise")
        delta_norm = random.uniform(0.8, 1.2)
    else:
        log(f"Applying DP-SGD: clip C=1.0, ε={DP_EPSILON}, σ={DP_SIGMA} → "
            f"gradien terganggu, model diverge (lihat Bab 4.3 thesis)")
        delta_norm = random.uniform(8.0, 15.0)  # norm meledak akibat noise
    time.sleep(1)
    return delta_norm


def submit_update(delta_norm):
    try:
        requests.post(f"http://{SERVER}/submit_update",
                      json={"client_id": CLIENT_ID, "delta_norm": delta_norm},
                      timeout=5)
    except Exception as e:
        log(f"Submit failed: {e.__class__.__name__}")


def main():
    if SCENARIO == "baseline":
        log(f"Starting [BASELINE] (server={SERVER}, α={DIRICHLET_ALPHA}, ε=∞, no DP)")
    else:
        log(f"Starting [DP] (server={SERVER}, α={DIRICHLET_ALPHA}, ε={DP_EPSILON}, σ={DP_SIGMA})")
    register()
    for r in range(1, NUM_ROUNDS + 1):
        log(f"=== Round {r}/{NUM_ROUNDS} ===")
        log("Receiving global model from server")
        time.sleep(2)
        delta_norm = fake_local_training()
        log(f"Sending DP-protected Δw to server (norm={delta_norm:.4f})")
        submit_update(delta_norm)
        time.sleep(2)
    log("All rounds complete. Client idle.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
