"""Shared sweep logic for E1 (full DP-SGD) and E2 (partial DP-SGD, backbone
frozen) -- both are the same K x sigma grid loop, differing only in
`freeze_stages` and where results/checkpoints land. Imported by
07_train_e1_dp_full.py and 08_train_e2_dp_partial.py rather than duplicated.

Test hygiene: this sweep evaluates the VALIDATION split ONLY (for
best-checkpoint selection). The test split is never read here. A single test
evaluation of the locked (K, sigma) best checkpoint is a separate, explicit
step run only after the configuration is locked and on explicit instruction.

Privacy accounting is persistent per client across rounds: each client's
accountant state is threaded round-to-round via a closure dict, so the
epsilon reported for a client is cumulative over all its local optimizer
steps. Epsilons are never summed across clients (each image lives on exactly
one client); the summary is the per-client epsilon and the maximum.
"""
import json
from pathlib import Path

import yaml

import fedxpalm  # noqa: F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector
from fedxpalm.federated.server import run_federated_training
from fedxpalm.privacy.dp_sgd import train_client_round_dp


def make_dp_client_round_fn(hyp, dp_hyp, device, freeze_stages, accountant_states):
    """`accountant_states` is a per-client dict (client_id -> saved accountant
    state) mutated in place across rounds so each client's privacy budget
    accumulates. The raw accountant state is threaded here and kept OUT of the
    history log to avoid bloating history.json."""
    def client_round_fn(client_id, data_yaml, global_weights_path, round_idx, out_dir):
        state_dict, info = train_client_round_dp(
            global_weights_path, data_yaml, hyp, dp_hyp, round_idx, client_id, out_dir,
            device=device, freeze_stages=freeze_stages,
            accountant_state=accountant_states.get(client_id),
        )
        accountant_states[client_id] = info.pop("accountant_state")
        return state_dict, info

    return client_round_fn


def run_dp_sweep(variant: str, device: str = "0", k_override: int | None = None,
                  sigma_override: float | None = None, rounds_override: int | None = None,
                  imgsz_override: int | None = None, batch_override: int | None = None,
                  tag_suffix: str = "", eval_every: int = 1):
    """variant: 'full' (E1) or 'partial' (E2), matching configs/dp_config.yaml's `variants` keys.

    Returns the list of per-cell result dicts (also written to results/).
    """
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
    tag = ("e1_dp_full" if variant == "full" else "e2_dp_partial") + (f"_{tag_suffix}" if tag_suffix else "")

    Path("results").mkdir(exist_ok=True)

    def eval_fn(weights_path):
        # VALIDATION split only -- best-checkpoint selection. Test is never read here.
        return evaluate_detector(weights_path, data_yaml, split="val", imgsz=imgsz, device=device)

    all_results = []
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
            accountant_states: dict = {}  # per-client, persisted across rounds
            client_round_fn = make_dp_client_round_fn(hyp, dp_hyp, device, freeze_stages, accountant_states)

            out_dir = f"runs/{tag}/k{k}_sigma{sigma}"
            result = run_federated_training(
                client_round_fn, client_data_yamls, client_sample_counts,
                init_weights_path="models/base_groupnorm.pt", rounds=rounds, out_dir=out_dir,
                eval_fn=eval_fn, eval_every=eval_every,
            )

            # per-client CUMULATIVE epsilon from the final round's client logs
            final_round = result["history"][-1]
            eps_per_client = {cid: info.get("epsilon") for cid, info in final_round["clients"].items()}
            epsilon_max = max(eps_per_client.values()) if eps_per_client else None
            nan_inf_any = any(info.get("nan_inf") for r in result["history"] for info in r["clients"].values())

            record = {
                "variant": variant, "k": k, "sigma": sigma, "rounds": rounds,
                "imgsz": imgsz, "batch_size": hyp["batch_size"],
                "freeze_stages": freeze_stages,
                "best_round": result["best_round"],
                "best_val_map50": result["best_val_map50"],
                "epsilon_per_client_final": eps_per_client,
                "epsilon_max_over_clients": epsilon_max,
                "delta": dp_hyp["delta"], "max_grad_norm": dp_hyp["max_grad_norm"],
                "nan_inf_any": nan_inf_any,
                "best_weights": result["best_weights"],
                "final_weights": result["final_weights"],
                # test intentionally NOT evaluated in the sweep
                "test_evaluated": False,
                "map50": None, "map50_95": None, "precision": None, "recall": None, "per_class": None,
                "history": str(Path(out_dir) / "history.json"),
                "leakage_audit_passed": True,
            }
            out_path = Path("results") / f"{tag}_k{k}_sigma{sigma}.json"
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            all_results.append(record)
            print(f"{tag} K={k} sigma={sigma}: best-val mAP@0.5={result['best_val_map50']} "
                  f"(round {result['best_round']})  epsilon_max={epsilon_max}  "
                  f"nan_inf={nan_inf_any}  -> {out_path}")
    return all_results
