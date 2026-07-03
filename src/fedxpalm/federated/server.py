"""Server-aggregator loop: orchestrates K clients through FedAvg rounds.

Works for both the no-DP path (federated/client.py, block B2) and the
DP-SGD path (privacy/dp_sgd.py, blocks E1/E2) via the `client_round_fn`
callback -- the server doesn't need to know which one it's driving.
"""
from __future__ import annotations

import json
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
) -> dict:
    """Runs `rounds` of FedAvg. `client_round_fn(client_id, data_yaml, global_weights_path,
    round_idx, out_dir) -> (state_dict, extra_info_dict)`.

    Writes `global_round_{t}.pt` checkpoints and a `history.json` log of
    per-round metrics (whatever `extra_info_dict` each client returns, e.g.
    epsilon for DP runs) to `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_weights_path = init_weights_path
    history = []

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
            "clients": client_infos,
        }
        if round_extra_log:
            round_record.update(round_extra_log)
        history.append(round_record)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"[round {t + 1}/{rounds}] aggregated {len(state_dicts)} clients "
              f"in {round_record['elapsed_sec']:.1f}s -> {global_weights_path}")

    return {"final_weights": global_weights_path, "history": history}
