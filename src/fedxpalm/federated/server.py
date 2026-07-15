"""Server-aggregator loop: orchestrates K clients through FedAvg rounds.

Works for both the no-DP path (federated/client.py, block B2) and the
DP-SGD path (privacy/dp_sgd.py, blocks E1/E2) via the `client_round_fn`
callback -- the server doesn't need to know which one it's driving.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from fedxpalm.federated.fedavg import fedavg


def run_federated_training(
    client_round_fn,
    client_data_yamls: dict[str, str],
    client_sample_counts: dict[str, int],
    init_weights_path: str,
    rounds: int,
    out_dir: str,
    round_extra_log: dict | None = None,
    eval_fn=None,
    eval_every: int = 1,
) -> dict:
    """Runs `rounds` of FedAvg. `client_round_fn(client_id, data_yaml, global_weights_path,
    round_idx, out_dir) -> (state_dict, extra_info_dict)`.

    Writes `global_round_{t}.pt` checkpoints and a `history.json` log of
    per-round metrics (whatever `extra_info_dict` each client returns, e.g.
    epsilon for DP runs) to `out_dir`.

    `eval_fn(weights_path) -> metrics dict` (expected keys at least
    "map50"/"map50_95") is called on the aggregated global model every
    `eval_every` rounds against the *validation* split. The best round by
    val map50 is tracked and its checkpoint copied to `best_global.pt`;
    the last round's checkpoint is copied to `final_global.pt`. Final test
    evaluation stays the caller's job (val is for selection only, so the
    test set never influences which checkpoint gets picked).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_weights_path = init_weights_path
    history = []
    best = {"round": None, "map50": -1.0, "weights": None}

    for t in range(rounds):
        round_start = time.time()
        state_dicts, sample_counts, client_infos = [], [], {}

        for client_id, data_yaml in client_data_yamls.items():
            sd, info = client_round_fn(client_id, data_yaml, global_weights_path, t, str(out_dir))
            state_dicts.append(sd)
            sample_counts.append(client_sample_counts[client_id])
            client_infos[client_id] = info

        aggregated = fedavg(state_dicts, sample_counts)

        # save aggregated weights back into a loadable YOLO checkpoint by
        # re-using the previous checkpoint's non-tensor metadata
        ckpt = torch.load(global_weights_path, map_location="cpu", weights_only=False)
        ckpt["model"].load_state_dict(aggregated)
        global_weights_path = str(out_dir / f"global_round_{t}.pt")
        torch.save(ckpt, global_weights_path)

        round_record = {
            "round": t,
            "elapsed_sec": time.time() - round_start,
            "checkpoint": global_weights_path,
            "clients": client_infos,
        }

        if eval_fn is not None and (t % eval_every == 0 or t == rounds - 1):
            val_metrics = eval_fn(global_weights_path)
            round_record["val"] = val_metrics
            if float(val_metrics.get("map50", -1.0)) > best["map50"]:
                best = {"round": t, "map50": float(val_metrics["map50"]),
                        "weights": str(out_dir / "best_global.pt")}
                shutil.copy2(global_weights_path, best["weights"])
            print(f"[round {t + 1}/{rounds}] val mAP@0.5={val_metrics.get('map50', float('nan')):.3f} "
                  f"(best so far: {best['map50']:.3f} @ round {best['round']})")

        if round_extra_log:
            round_record.update(round_extra_log)
        history.append(round_record)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"[round {t + 1}/{rounds}] aggregated {len(state_dicts)} clients "
              f"in {round_record['elapsed_sec']:.1f}s -> {global_weights_path}")

    final_copy = str(out_dir / "final_global.pt")
    shutil.copy2(global_weights_path, final_copy)

    return {
        "final_weights": final_copy,
        "best_weights": best["weights"] or final_copy,  # no eval_fn -> fall back to final
        "best_round": best["round"] if best["round"] is not None else rounds - 1,
        "best_val_map50": best["map50"] if best["map50"] >= 0 else None,
        "history": history,
    }
