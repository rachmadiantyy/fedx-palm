#!/usr/bin/env python3
"""B2 -- federated baseline (FedAvg, no DP), swept over every K in
configs/fl_config.yaml's clients.k_values. Mirrors thesis Table 4.2."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.client import train_client_round  # noqa: E402
from fedxpalm.federated.server import run_federated_training  # noqa: E402


def make_client_round_fn(hyp, device):
    def client_round_fn(client_id, data_yaml, global_weights_path, round_idx, out_dir):
        import torch

        weights_path = train_client_round(global_weights_path, data_yaml, hyp, round_idx, client_id, out_dir, device=device)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)["model"].state_dict()
        return state_dict, {}

    return client_round_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--k", type=int, default=None, help="run only this K (default: sweep all k_values)")
    parser.add_argument("--rounds", type=int, default=None, help="override configs/fl_config.yaml federated.rounds")
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    hyp = dict(fl_cfg["local_training"], imgsz=fl_cfg["model"]["imgsz"])
    rounds = args.rounds or fl_cfg["federated"]["rounds"]
    k_values = [args.k] if args.k is not None else fl_cfg["clients"]["k_values"]

    Path("results").mkdir(exist_ok=True)
    client_round_fn = make_client_round_fn(hyp, args.device)

    for k in k_values:
        manifest_path = splits_dir / "federated_partitions" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)[str(k)]
        clients_dir = splits_dir / "federated_partitions" / f"k{k}_clients"

        client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
        client_sample_counts = {cid: n for cid, n in manifest["sizes"].items()}

        out_dir = f"runs/b2_federated/k{k}"
        result = run_federated_training(
            client_round_fn, client_data_yamls, client_sample_counts,
            init_weights_path="models/base_groupnorm.pt", rounds=rounds, out_dir=out_dir,
        )

        metrics = evaluate_detector(result["final_weights"], data_yaml, split="test",
                                     imgsz=fl_cfg["model"]["imgsz"], device=args.device)
        with open(f"results/b2_k{k}.json", "w") as f:
            json.dump({"k": k, "rounds": rounds, "weights": result["final_weights"], "metrics": metrics}, f, indent=2)
        print(f"B2 K={k}: mAP@0.5={metrics['map50']:.3f}  mAP@0.5:0.95={metrics['map50_95']:.3f}")

    print("Saved results/b2_k*.json")
