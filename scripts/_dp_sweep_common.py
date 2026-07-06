"""Shared sweep logic for E1 (full DP-SGD) and E2 (partial DP-SGD, backbone
frozen) -- both are the same K x sigma grid loop, differing only in
`freeze_stages` and where results/checkpoints land. Imported by
07_train_e1_dp_full.py and 08_train_e2_dp_partial.py rather than duplicated.
"""
import json
from pathlib import Path

import yaml

import fedxpalm  # noqa: F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector
from fedxpalm.federated.server import run_federated_training
from fedxpalm.privacy.dp_sgd import train_client_round_dp


def make_dp_client_round_fn(hyp, dp_hyp, device, freeze_stages):
    def client_round_fn(client_id, data_yaml, global_weights_path, round_idx, out_dir):
        state_dict, info = train_client_round_dp(
            global_weights_path, data_yaml, hyp, dp_hyp, round_idx, client_id, out_dir,
            device=device, freeze_stages=freeze_stages,
        )
        return state_dict, info

    return client_round_fn


def run_dp_sweep(variant: str, device: str = "0", k_override: int | None = None,
                  sigma_override: float | None = None, rounds_override: int | None = None,
                  imgsz_override: int | None = None, batch_override: int | None = None):
    """variant: 'full' (E1) or 'partial' (E2), matching configs/dp_config.yaml's `variants` keys."""
    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    imgsz = imgsz_override or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz)
    if batch_override:
        hyp["batch_size"] = batch_override
    rounds = rounds_override or fl_cfg["federated"]["rounds"]
    k_values = [k_override] if k_override is not None else fl_cfg["clients"]["k_values"]
    sigma_values = [sigma_override] if sigma_override is not None else dp_cfg["dp_sgd"]["noise_multiplier_values"]
    freeze_stages = dp_cfg["variants"][variant]["freeze_stages"]
    tag = "e1_dp_full" if variant == "full" else "e2_dp_partial"

    Path("results").mkdir(exist_ok=True)

    for k in k_values:
        manifest_path = splits_dir / "federated_partitions" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)[str(k)]
        clients_dir = splits_dir / "federated_partitions" / f"k{k}_clients"
        client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
        client_sample_counts = dict(manifest["sizes"])

        for sigma in sigma_values:
            dp_hyp = dict(dp_cfg["dp_sgd"], sigma=sigma)
            dp_hyp.pop("noise_multiplier_values", None)
            client_round_fn = make_dp_client_round_fn(hyp, dp_hyp, device, freeze_stages)

            out_dir = f"runs/{tag}/k{k}_sigma{sigma}"
            result = run_federated_training(
                client_round_fn, client_data_yamls, client_sample_counts,
                init_weights_path="models/base_groupnorm.pt", rounds=rounds, out_dir=out_dir,
            )

            final_round = result["history"][-1]
            epsilons = [info["epsilon"] for info in final_round["clients"].values()]
            epsilon_max = max(epsilons)

            metrics = evaluate_detector(result["final_weights"], data_yaml, split="test",
                                         imgsz=imgsz, device=device)

            out_path = Path("results") / f"{tag}_k{k}_sigma{sigma}.json"
            with open(out_path, "w") as f:
                json.dump({
                    "variant": variant, "k": k, "sigma": sigma, "rounds": rounds,
                    "epsilon_max_over_clients": epsilon_max,
                    "weights": result["final_weights"], "metrics": metrics,
                }, f, indent=2)
            print(f"{tag} K={k} sigma={sigma}: mAP@0.5={metrics['map50']:.3f} "
                  f"epsilon={epsilon_max:.3f}  -> {out_path}")
