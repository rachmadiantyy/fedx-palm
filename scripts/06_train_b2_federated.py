#!/usr/bin/env python3
"""B2 -- federated baseline (FedAvg, no DP), swept over every K in
configs/fl_config.yaml's clients.k_values. Mirrors thesis Table 4.2.

Checkpoint selection: the aggregated global model is evaluated on the
*validation* split every round; the best round's weights are saved as
best_global.pt and the final test-set numbers are reported for that
checkpoint (plus final_global.pt for reference). The test set is never
used to pick a checkpoint or tune anything.

Results go to results/b2_k{K}_seed{S}.json -- old results/b2_k{K}.json
files from before this scheme are left untouched.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.client import read_local_train_log, train_client_round  # noqa: E402
from fedxpalm.federated.server import run_federated_training  # noqa: E402


def make_client_round_fn(hyp, device, rounds, lr_round_decay=False):
    def client_round_fn(client_id, data_yaml, global_weights_path, round_idx, out_dir):
        import torch

        round_hyp = dict(hyp)
        if lr_round_decay and rounds > 1:
            # cosine decay of the *round's* starting LR: lr0 -> 0.1*lr0 across rounds
            frac = 0.5 * (1 + math.cos(math.pi * round_idx / (rounds - 1)))
            round_hyp["lr0"] = hyp["lr0"] * (0.1 + 0.9 * frac)

        weights_path = train_client_round(global_weights_path, data_yaml, round_hyp,
                                          round_idx, client_id, out_dir, device=device)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)["model"].state_dict()
        info = read_local_train_log(Path(weights_path).parent.parent)
        info["lr0_this_round"] = round_hyp["lr0"]
        return state_dict, info

    return client_round_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--k", type=int, default=None, help="run only this K (default: sweep all k_values)")
    parser.add_argument("--rounds", type=int, default=None, help="override configs/fl_config.yaml federated.rounds")
    parser.add_argument("--imgsz", type=int, default=None, help="override configs/fl_config.yaml model.imgsz")
    parser.add_argument("--batch", type=int, default=None, help="override local_training.batch_size (lower if OOM)")
    parser.add_argument("--seed", type=int, default=42, help="training seed (Ultralytics trainer)")
    parser.add_argument("--partition-seed", type=int, default=None,
                        help="use partitions from federated_partitions_seed{S}/ (default: primary "
                             "federated_partitions/, i.e. fl_config clients.partition_seed)")
    parser.add_argument("--warmup-epochs", type=float, default=None,
                        help="override Ultralytics warmup_epochs per local run (default 3.0 = prior behavior; "
                             "with 2 local epochs the whole round trains inside warmup -- try 0)")
    parser.add_argument("--lr-round-decay", action="store_true",
                        help="cosine-decay each round's starting lr0 down to 0.1*lr0 across rounds")
    parser.add_argument("--eval-every", type=int, default=1, help="validate the global model every N rounds")
    parser.add_argument("--tag", default=None, help="extra suffix for results/run dirs (e.g. ablation name)")
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, seed=args.seed)
    if args.batch:
        hyp["batch_size"] = args.batch
    if args.warmup_epochs is not None:
        hyp["warmup_epochs"] = args.warmup_epochs
    rounds = args.rounds or fl_cfg["federated"]["rounds"]
    k_values = [args.k] if args.k is not None else fl_cfg["clients"]["k_values"]

    cfg_partition_seed = fl_cfg["clients"]["partition_seed"]
    partition_seed = args.partition_seed if args.partition_seed is not None else cfg_partition_seed
    partition_root = ("federated_partitions" if partition_seed == cfg_partition_seed
                      else f"federated_partitions_seed{partition_seed}")

    Path("results").mkdir(exist_ok=True)
    client_round_fn = make_client_round_fn(hyp, args.device, rounds, lr_round_decay=args.lr_round_decay)

    def eval_fn(weights_path):
        # validation split only -- used solely for best-checkpoint selection
        return evaluate_detector(weights_path, data_yaml, split="val", imgsz=imgsz, device=args.device)

    for k in k_values:
        manifest_path = splits_dir / partition_root / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)[str(k)]
        clients_dir = splits_dir / partition_root / f"k{k}_clients"

        client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
        client_sample_counts = {cid: n for cid, n in manifest["sizes"].items()}

        tag = f"_{args.tag}" if args.tag else ""
        out_dir = f"runs/b2_federated/k{k}_seed{args.seed}{tag}"
        result = run_federated_training(
            client_round_fn, client_data_yamls, client_sample_counts,
            init_weights_path="models/base_groupnorm.pt", rounds=rounds, out_dir=out_dir,
            eval_fn=eval_fn, eval_every=args.eval_every,
        )

        # test set: evaluated once, on the val-selected best checkpoint
        metrics = evaluate_detector(result["best_weights"], data_yaml, split="test",
                                    imgsz=imgsz, device=args.device)
        final_metrics = evaluate_detector(result["final_weights"], data_yaml, split="test",
                                          imgsz=imgsz, device=args.device)

        record = {
            "experiment": "B2",
            "num_clients": k,
            "seed": args.seed,
            "partition_seed": partition_seed,
            "dirichlet_alpha": manifest.get("dirichlet_alpha", fl_cfg["clients"]["dirichlet_alpha"]),
            "communication_rounds": rounds,
            "local_epochs": hyp["epochs_per_round"],
            "learning_rate": hyp["lr0"],
            "lr_round_decay": args.lr_round_decay,
            "warmup_epochs": hyp.get("warmup_epochs", 3.0),
            "batch_size": hyp["batch_size"],
            "imgsz": imgsz,
            "optimizer": hyp.get("optimizer", "SGD"),
            "normalization": "GroupNorm",
            "best_round": result["best_round"],
            "best_val_map50": result["best_val_map50"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "per_class": metrics["per_class"],
            "final_round_test_map50": final_metrics["map50"],
            "checkpoint": result["best_weights"],
            "final_checkpoint": result["final_weights"],
            "split_manifest": str(splits_dir / partition_root / "manifest.json"),
            "leakage_audit_passed": True,  # asserted by data/split.py at split time; see scripts/11_audit_split.py
        }
        out_json = f"results/b2_k{k}_seed{args.seed}{tag}.json"
        with open(out_json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"B2 K={k} seed={args.seed}: test mAP@0.5={metrics['map50']:.3f} "
              f"(best round {result['best_round']}, val {result['best_val_map50']}) "
              f"| final-round test mAP@0.5={final_metrics['map50']:.3f}")
        print(f"Saved {out_json}")
