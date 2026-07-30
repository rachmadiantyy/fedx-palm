#!/usr/bin/env python3
"""Computational + communication cost report for B1 / B2 / E1 / E2 -- reads
ONLY already-completed runs' own logs (results.csv, history.json, locked
result JSONs). No training, no re-evaluation, nothing recomputed from the
model itself.

Wall-clock time source per experiment (whatever that experiment's own
runner already recorded -- nothing timed here that wasn't already timed
there):
  B1  (centralized, scripts/05):      Ultralytics' own results.csv "time" column
  B2  (FedAvg no-DP, scripts/06):     result JSON's total_runtime_sec/mean_round_sec
                                      (fedxpalm.federated.server.run_federated_training)
  E1/E2 (DP-SGD, scripts/38):        summed from history.json's per-round elapsed_sec
                                      (scripts/38 itself never saved the summary fields
                                      run_federated_training returns -- recovered here
                                      from the same per-round log instead of re-running)

Communication payload is a THEORETICAL calculation (this is a single-GPU
simulated federated run -- fedxpalm.federated.server.run_federated_training
never actually transmits bytes over a network), not a measurement:
  per-round per-client payload = total_params * 4 bytes (float32) * 2 (the
  client downloads the current global model, then uploads its local
  update)
  total = payload * num_clients * communication_rounds

IMPORTANT, verified directly from source (fedxpalm/federated/fedavg.py):
fedavg() iterates every key in the state_dict with no filtering by
requires_grad, so E2's frozen 1,661,488 parameters are STILL fully
transmitted/aggregated every round in the current implementation. E2's
partial-DP design saves compute/GPU-memory (Opacus never needs to
materialize per-sample gradients for frozen parameters), NOT communication
bandwidth, under the code as it exists today. This script reports both the
as-implemented (identical total_params for E1/E2) figure and a clearly
separate, explicitly-labeled hypothetical "trainable-only" figure for E2,
which is NOT what actually happened.

Usage:
    python scripts/52_report_computation_communication_cost.py

Output (refuses to overwrite):
    results/cost_report/computation_communication_cost.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "results/cost_report/computation_communication_cost.json"

BYTES_PER_PARAM = 4  # float32
TOTAL_PARAMS = 2_591_010  # YOLO11n, nc=6 -- identical architecture for all 4 experiments
E1_TRAINABLE_PARAMS = 2_590_994
E2_TRAINABLE_PARAMS = 929_522
K_CLIENTS = 4

B1_RESULTS_CSV = REPO_ROOT / "runs/b1_centralized_leakagefree/train/results.csv"
B2_RESULTS_JSON = REPO_ROOT / "results/b2_k4_seed42_leakagefree_stageA_seedfix.json"
E1_HISTORY = REPO_ROOT / "runs/final_dp_canonical/e1_full/seed42_20r/history.json"
E2_HISTORY = REPO_ROOT / "runs/final_dp_canonical/e2_partial_p2/seed42_20r/history.json"
E1_RESULT_JSON = REPO_ROOT / "results/final_dp_canonical/e1_full_seed42_20r.json"
E2_RESULT_JSON = REPO_ROOT / "results/final_dp_canonical/e2_partial_p2_seed42_20r.json"


def read_b1_time(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: no rows")
    last = rows[-1]
    time_key = next((k for k in last if k.strip().lower() == "time"), None)
    if time_key is None:
        raise ValueError(f"{csv_path}: no 'time' column found -- columns were {list(last.keys())}")
    return {
        "n_epochs": len(rows),
        "total_wall_clock_sec": float(last[time_key]),
        "source": "results.csv 'time' column, last row (Ultralytics' own cumulative training time)",
    }


def read_federated_history_time(history_path: Path) -> dict:
    with open(history_path) as f:
        history = json.load(f)
    elapsed = [r["elapsed_sec"] for r in history if "elapsed_sec" in r]
    if not elapsed:
        raise ValueError(f"{history_path}: no round has 'elapsed_sec'")
    return {
        "n_rounds": len(elapsed),
        "total_wall_clock_sec": sum(elapsed),
        "mean_round_sec": sum(elapsed) / len(elapsed),
        "source": "history.json per-round 'elapsed_sec', summed (server.py's own timer)",
    }


def comm_payload_bytes(total_params: int, k_clients: int, rounds: int) -> dict:
    per_client_per_round = total_params * BYTES_PER_PARAM * 2  # download + upload
    total = per_client_per_round * k_clients * rounds
    return {
        "params_communicated": total_params,
        "bytes_per_client_per_round": per_client_per_round,
        "total_bytes_all_rounds_all_clients": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def main() -> int:
    if OUT_JSON.exists():
        print(f"FAIL: {OUT_JSON} already exists -- refusing to overwrite")
        return 1

    # B1 is reported best-effort: its runs/ directory has been observed to move
    # between machines/sessions (e.g. b1_centralized -> b1_centralized_leakagefree),
    # and its wall-clock time is a nice-to-have, not something B2/E1/E2's own
    # report should be blocked on if it can't be found.
    missing_required = [p for p in (B2_RESULTS_JSON, E1_HISTORY, E2_HISTORY,
                                    E1_RESULT_JSON, E2_RESULT_JSON) if not p.exists()]
    if missing_required:
        print("FAIL: required file(s) not found:")
        for p in missing_required:
            print(f"  {p}")
        return 1

    if B1_RESULTS_CSV.exists():
        b1_time = read_b1_time(B1_RESULTS_CSV)
    else:
        print(f"[!] {B1_RESULTS_CSV} not found -- B1 wall-clock time omitted from this report "
              f"(B2/E1/E2 are unaffected)")
        b1_time = None

    with open(B2_RESULTS_JSON) as f:
        b2 = json.load(f)
    b2_time = {
        "n_rounds": b2["communication_rounds"],
        "total_wall_clock_sec": b2["total_runtime_sec"],
        "mean_round_sec": b2["mean_round_sec"],
        "source": "results JSON's total_runtime_sec/mean_round_sec (run_federated_training's own timer)",
    }
    b2_comm = comm_payload_bytes(TOTAL_PARAMS, K_CLIENTS, b2["communication_rounds"])

    e1_time = read_federated_history_time(E1_HISTORY)
    e2_time = read_federated_history_time(E2_HISTORY)
    with open(E1_RESULT_JSON) as f:
        e1 = json.load(f)
    with open(E2_RESULT_JSON) as f:
        e2 = json.load(f)

    e1_comm = comm_payload_bytes(TOTAL_PARAMS, K_CLIENTS, e1_time["n_rounds"])
    e2_comm_as_implemented = comm_payload_bytes(TOTAL_PARAMS, K_CLIENTS, e2_time["n_rounds"])
    e2_comm_hypothetical_trainable_only = comm_payload_bytes(
        E2_TRAINABLE_PARAMS, K_CLIENTS, e2_time["n_rounds"])

    e1_final_steps = e1["private_steps_per_client_at_final_round"]
    e2_final_steps = e2["private_steps_per_client_at_final_round"]

    report = {
        "report_kind": "computation_communication_cost",
        "note": "Wall-clock times are each experiment's OWN already-recorded timer, read from "
                "its own log file -- nothing was re-run or re-timed here. Communication payload "
                "is a theoretical calculation (this is a single-GPU simulated federated run, no "
                "bytes were actually transmitted over any network).",
        "architecture_note": f"Same YOLO11n (nc=6) architecture for all 4 experiments -- "
                             f"{TOTAL_PARAMS} total params, 6.3 GFLOPs/image at inference "
                             f"(imgsz=960, per Ultralytics' own model summary) -- FLOPs/image is "
                             f"architecture-constant and does not by itself capture Opacus's "
                             f"per-sample-gradient training overhead in E1/E2; wall-clock time is "
                             f"used as the practical proxy for that instead, since no dedicated "
                             f"training-time FLOPs profiler was instrumented in this codebase.",
        "fedavg_communication_note": "Verified from fedxpalm/federated/fedavg.py: fedavg() "
                                     "aggregates every state_dict key with no requires_grad "
                                     "filtering, so E2's 1,661,488 frozen parameters are STILL "
                                     "fully transmitted/aggregated every round in the current "
                                     "implementation. E2's benefit under this code is "
                                     "compute/GPU-memory (Opacus skips per-sample gradients for "
                                     "frozen params), NOT communication bandwidth. The "
                                     "'hypothetical_trainable_only' figure below for E2 is NOT "
                                     "what actually happened -- it is a separate, clearly-labeled "
                                     "illustration of the bandwidth reduction possible if a "
                                     "future implementation only communicated trainable params.",
        "B1_centralized": {
            "description": "no federation, no DP, single machine, single model",
            "wall_clock": b1_time,
            "communication": None,
        },
        "B2_fedavg_nodp": {
            "description": f"FedAvg, K={K_CLIENTS}, no DP, seed=42",
            "wall_clock": b2_time,
            "total_optimizer_steps": b2.get("total_optimizer_steps"),
            "total_sample_exposure": b2.get("total_sample_exposure"),
            "communication_as_implemented": b2_comm,
        },
        "E1_full_dp": {
            "description": f"FedAvg + DP-SGD, K={K_CLIENTS}, ALL params trainable, sigma=0.75, C=1, seed=42",
            "wall_clock": e1_time,
            "trainable_params": E1_TRAINABLE_PARAMS,
            "cumulative_private_steps_per_client_at_final_round": e1_final_steps,
            "communication_as_implemented": e1_comm,
        },
        "E2_partial_dp_P2": {
            "description": f"FedAvg + DP-SGD, K={K_CLIENTS}, trainable stages 16/19/22/23 only, "
                           f"sigma=0.75, C=1, seed=42",
            "wall_clock": e2_time,
            "trainable_params": E2_TRAINABLE_PARAMS,
            "cumulative_private_steps_per_client_at_final_round": e2_final_steps,
            "communication_as_implemented": e2_comm_as_implemented,
            "communication_hypothetical_trainable_only_NOT_IMPLEMENTED": e2_comm_hypothetical_trainable_only,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'experiment':>14}{'rounds/epochs':>15}{'total_sec':>12}{'sec/round':>12}{'params_comm':>13}")
    if b1_time is not None:
        print(f"{'B1':>14}{b1_time['n_epochs']:>15}{b1_time['total_wall_clock_sec']:>12.1f}{'n/a':>12}{'n/a':>13}")
    else:
        print(f"{'B1':>14}{'n/a (not found)':>15}")
    print(f"{'B2':>14}{b2_time['n_rounds']:>15}{b2_time['total_wall_clock_sec']:>12.1f}"
          f"{b2_time['mean_round_sec']:>12.1f}{TOTAL_PARAMS:>13}")
    print(f"{'E1':>14}{e1_time['n_rounds']:>15}{e1_time['total_wall_clock_sec']:>12.1f}"
          f"{e1_time['mean_round_sec']:>12.1f}{TOTAL_PARAMS:>13}")
    print(f"{'E2':>14}{e2_time['n_rounds']:>15}{e2_time['total_wall_clock_sec']:>12.1f}"
          f"{e2_time['mean_round_sec']:>12.1f}{TOTAL_PARAMS:>13} (as-implemented)")
    print(f"\nCommunication (as-implemented, theoretical): B2={b2_comm['total_gb']:.3f} GB  "
          f"E1={e1_comm['total_gb']:.3f} GB  E2={e2_comm_as_implemented['total_gb']:.3f} GB")
    print(f"E2 hypothetical trainable-only (NOT what was actually implemented): "
          f"{e2_comm_hypothetical_trainable_only['total_gb']:.3f} GB")
    print(f"\nSaved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
