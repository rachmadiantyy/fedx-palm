#!/usr/bin/env python3
"""B2 -- federated baseline (FedAvg, no DP), swept over every K in
configs/fl_config.yaml's clients.k_values. Mirrors thesis Table 4.2.

Checkpoint selection: the aggregated global model is evaluated on the
*validation* split every round; the best round's weights are saved as
best_global.pt. The test set is NEVER touched during training or tuning --
it is only evaluated when --eval-test is passed explicitly, i.e. after the
configuration has been locked. Tuning runs therefore compare val mAP.

Seeds: the Dirichlet client partition follows --seed by default (seed 42 ->
the primary federated_partitions/, other seeds -> federated_partitions_seed{S}/,
auto-generated if missing), so a different training seed never silently
re-uses seed 42's partition. Override with --partition-seed if you need to
decouple them deliberately.

Config inheritance: --from-json results/b2_kX_seedY[_tag].json re-uses that
run's rounds / lr0 / warmup / lr-round-decay / batch / imgsz, so staged
experiments always continue from the previous winner and the three-seed
validation runs are guaranteed hyperparameter-identical. Explicit CLI flags
still override inherited values.

Results go to results/b2_k{K}_seed{S}[_tag].json -- old results/b2_k{K}.json
files from before this scheme are left untouched.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.data.partition import partition_and_write  # noqa: E402
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.client import read_local_train_log, train_client_round  # noqa: E402
from fedxpalm.federated.client_data import materialize_clients  # noqa: E402
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


def ensure_partition(splits_dir: Path, partition_root: str, k: int, partition_seed: int,
                     nc: int, names: list[str]) -> None:
    """Auto-generate + materialize the K-client partition for this seed if missing,
    so `--seed 123` can never silently fall back to seed 42's clients."""
    manifest_path = splits_dir / partition_root / "manifest.json"
    have = False
    if manifest_path.exists():
        with open(manifest_path) as f:
            have = str(k) in json.load(f)
    if not have:
        print(f"[partition] {partition_root}/manifest.json has no K={k} entry -- "
              f"generating with partition_seed={partition_seed}")
        partition_and_write("configs/dataset.yaml", "configs/fl_config.yaml",
                            k=k, seed=partition_seed)
    clients_dir = splits_dir / partition_root / f"k{k}_clients"
    if not clients_dir.exists():
        materialize_clients(str(splits_dir / partition_root / f"k{k}.json"),
                            str(splits_dir), str(clients_dir), nc, names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--k", type=int, default=None, help="run only this K (default: sweep all k_values)")
    parser.add_argument("--rounds", type=int, default=None, help="override configs/fl_config.yaml federated.rounds")
    parser.add_argument("--imgsz", type=int, default=None, help="override configs/fl_config.yaml model.imgsz")
    parser.add_argument("--batch", type=int, default=None, help="override local_training.batch_size (lower if OOM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="experiment seed: sets the Ultralytics training seed AND (unless "
                             "--partition-seed is given) the Dirichlet partition seed")
    parser.add_argument("--partition-seed", type=int, default=None,
                        help="decouple the partition seed from --seed (rarely needed)")
    parser.add_argument("--warmup-epochs", type=float, default=None,
                        help="override Ultralytics warmup_epochs per local run (default 3.0 = prior behavior; "
                             "with 2 local epochs the whole round trains inside warmup -- try 0)")
    parser.add_argument("--lr-round-decay", action="store_true", default=None,
                        help="cosine-decay each round's starting lr0 down to 0.1*lr0 across rounds")
    parser.add_argument("--eval-every", type=int, default=1, help="validate the global model every N rounds")
    parser.add_argument("--eval-test", action="store_true",
                        help="ALSO evaluate the selected best checkpoint on the test split. Only pass "
                             "this once the configuration is final -- tuning runs must compare val mAP")
    parser.add_argument("--from-json", default=None,
                        help="inherit rounds/lr0/warmup/lr-round-decay/batch/imgsz from a previous "
                             "results/b2_*.json (explicit CLI flags still win)")
    parser.add_argument("--tag", default=None, help="extra suffix for results/run dirs (e.g. ablation name)")
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    inherited = {}
    if args.from_json:
        with open(args.from_json) as f:
            prev = json.load(f)
        inherited = {
            "rounds": prev.get("communication_rounds"),
            "imgsz": prev.get("imgsz"),
            "batch": prev.get("batch_size"),
            "lr0": prev.get("learning_rate"),
            "warmup_epochs": prev.get("warmup_epochs"),
            "lr_round_decay": prev.get("lr_round_decay"),
        }
        print(f"[inherit] {args.from_json} -> {inherited}")

    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    imgsz = args.imgsz or inherited.get("imgsz") or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, seed=args.seed)
    if inherited.get("lr0") is not None:
        hyp["lr0"] = inherited["lr0"]
    batch = args.batch or inherited.get("batch")
    if batch:
        hyp["batch_size"] = batch
    warmup = args.warmup_epochs if args.warmup_epochs is not None else inherited.get("warmup_epochs")
    if warmup is not None:
        hyp["warmup_epochs"] = warmup
    lr_round_decay = args.lr_round_decay if args.lr_round_decay is not None \
        else bool(inherited.get("lr_round_decay", False))
    rounds = args.rounds or inherited.get("rounds") or fl_cfg["federated"]["rounds"]
    k_values = [args.k] if args.k is not None else fl_cfg["clients"]["k_values"]

    # partition follows the experiment seed unless deliberately decoupled
    partition_seed = args.partition_seed if args.partition_seed is not None else args.seed
    cfg_partition_seed = fl_cfg["clients"]["partition_seed"]
    partition_root = ("federated_partitions" if partition_seed == cfg_partition_seed
                      else f"federated_partitions_seed{partition_seed}")

    Path("results").mkdir(exist_ok=True)
    client_round_fn = make_client_round_fn(hyp, args.device, rounds, lr_round_decay=lr_round_decay)

    def eval_fn(weights_path):
        # validation split only -- used solely for best-checkpoint selection
        return evaluate_detector(weights_path, data_yaml, split="val", imgsz=imgsz, device=args.device)

    for k in k_values:
        ensure_partition(splits_dir, partition_root, k, partition_seed, ds_cfg["nc"], ds_cfg["names"])
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

        # training-budget bookkeeping: report actual sample exposure / optimizer
        # steps instead of hand-waving "R rounds x E local epochs ~ RxE epochs"
        # (each round only touches each client's own shard once per local epoch,
        # so exposure = sum_k n_k * E * R = |train| * E * R, while a centralized
        # epoch touches |train| once -- comparable only via these numbers).
        epochs_local = hyp["epochs_per_round"]
        bsz = hyp["batch_size"]
        exposure = sum(client_sample_counts.values()) * epochs_local * rounds
        opt_steps = sum(math.ceil(n / bsz) * epochs_local * rounds for n in client_sample_counts.values())

        record = {
            "experiment": "B2",
            "num_clients": k,
            "seed": args.seed,
            "partition_seed": partition_seed,
            "dirichlet_alpha": manifest.get("dirichlet_alpha", fl_cfg["clients"]["dirichlet_alpha"]),
            "communication_rounds": rounds,
            "local_epochs": epochs_local,
            "learning_rate": hyp["lr0"],
            "lr_round_decay": lr_round_decay,
            "warmup_epochs": hyp.get("warmup_epochs", 3.0),
            "batch_size": bsz,
            "imgsz": imgsz,
            "optimizer": hyp.get("optimizer", "SGD"),
            "normalization": "GroupNorm",
            "total_sample_exposure": exposure,
            "total_optimizer_steps": opt_steps,
            "best_round": result["best_round"],
            "best_val_map50": result["best_val_map50"],
            # test fields stay null until --eval-test: tuning compares val only
            "map50": None,
            "map50_95": None,
            "precision": None,
            "recall": None,
            "per_class": None,
            "checkpoint": result["best_weights"],
            "final_checkpoint": result["final_weights"],
            "split_manifest": str(manifest_path),
            "leakage_audit_passed": True,  # asserted by data/split.py at split time; see scripts/11_audit_split.py
        }

        if args.eval_test:
            metrics = evaluate_detector(result["best_weights"], data_yaml, split="test",
                                        imgsz=imgsz, device=args.device)
            record.update({
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "per_class": metrics["per_class"],
            })

        out_json = f"results/b2_k{k}_seed{args.seed}{tag}.json"
        with open(out_json, "w") as f:
            json.dump(record, f, indent=2)

        line = (f"B2 K={k} seed={args.seed}: best round {result['best_round']} "
                f"val mAP@0.5={result['best_val_map50']}")
        if args.eval_test:
            line += f" | TEST mAP@0.5={record['map50']:.3f} mAP@0.5:0.95={record['map50_95']:.3f}"
        print(line)
        print(f"Saved {out_json}")
