#!/usr/bin/env python3
"""Diagnostic B -- clipping-only control (sigma=0, C=1.0), NOT 40 rounds.

Runs after Diagnostic A confirmed the healthy reference point: Partial No-DP
(freeze_stages=[0..10], lr0=0.01) climbed mAP50 0.368 -> 0.840 over 5 rounds
with a clean weight audit (backbone_changed=False, neck_head_changed=True,
dfl_changed=False). Freeze depth and lr0=0.01 are therefore RULED OUT as the
E2 sigma=0.5 collapse's primary cause. This isolates the next candidate:
per-sample gradient CLIPPING alone (no Gaussian noise), using the EXACT
production DP path (train_client_round_dp -- the same function E1/E2 call),
with sigma=0 so there is no noise term and NO finite (epsilon, delta)
guarantee: epsilon is always None, privacy_guarantee is always
"not_applicable_no_noise" (dp_sgd.py's existing sigma<=0 guard -- unchanged
here). This is Opacus's real per-sample-gradient path (PrivacyEngine,
GradSampleModule, DPOptimizer.step() with clip_and_accumulate()), NOT the
non-DP path Diagnostic A used -- genuinely isolates clipping from noise,
per instruction, rather than substituting sigma=0 for "no Opacus at all".

Everything else matches the healthy A0 configuration exactly:
  - init:        models/base_groupnorm.pt
  - partition:   K=4, partition seed 42 (primary federated_partitions/)
  - freeze:      configs/dp_config.yaml variants.partial.freeze_stages [0..10]
  - local hyp:   SGD lr0=0.01, momentum 0.9, wd 5e-4, 2 local epochs,
                 batch 8, imgsz 960, warmup 0 (same as A0)
  - max_grad_norm: configs/dp_config.yaml dp_sgd.max_grad_norm (C=1.0)
  - sigma:       0.0 (forced; NOT read from dp_config's sweep grid)
  - workers:     0 by default (matches 19_dp_smoke_test.py and the validated
                 E1/E2 DP scripts -- Opacus' locally-defined collate fn
                 can't be pickled into spawned workers on Windows, so
                 workers>0 crashes before training starts)
  - accountant:  threaded per client round-to-round (same mechanism as the
                 real E1/E2 sweep), but epsilon is never computed at sigma=0
  - eval:        VALIDATION split only, every round; no test-eval path exists
                 in this script

Clipping-severity instrumentation (train_client_round_dp's new
collect_grad_norms=True, additive/opt-in, unused by any real E1/E2 call
site): per round per client, pre-clipping per-sample gradient L2 norm
median/p75/p90/p95/max and the fraction of samples with norm > C (== the
clipping fraction, identical by construction under L2 clipping -- both
reported since both were asked for). Verified bit-exact against Opacus's own
internal clip_and_accumulate() computation via a monkey-patch spy test (not
in production code): this reads Opacus's public `grad_samples` property
after backward(), before step(), replicating its formula verbatim.

Outputs (all NEW namespace -- nothing under any B1/B2/E1/E2/diag_a path):
  runs/diag_b_clipping_only/k4_seed{seed}{tag}/       checkpoints + history.json
  results/diag_b_clipping_only_seed{seed}{tag}.json   metrics + audits + verdict

  python scripts/23_diag_b_clipping_only.py --device 0

--sigma (default 0.0) turns this same script into "Diagnostic C" when set to
the real E2 sigma (e.g. --sigma 0.5 --tag sigma0.5): the EXACT same
clipping-only config/instrumentation, but with noise turned on, isolating
what adding Gaussian noise changes relative to Diagnostic B's clean
clipping-only baseline (rather than clipping+noise vs no-DP-at-all, which
conflates both effects). The output PATH prefix stays "diag_b_clipping_only"
(same script, same mechanism) -- the run's actual identity (B vs C) is
recorded in the results JSON's "diagnostic" field and epsilon_max_over_clients
(None for B, a real number for C), and --tag keeps the files apart on disk.

This is a DIAGNOSTIC: its numbers are not thesis results and must not enter
any results table or privacy-utility curve. At sigma=0 (Diagnostic B),
epsilon is always null (no DP guarantee exists); at sigma>0 (Diagnostic C),
epsilon is real but still not a locked-configuration result.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.server import run_federated_training  # noqa: E402
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import audit_freeze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--rounds", type=int, default=5, help="diagnostic max is 5; override discouraged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=None, help="default: fl_config model.imgsz (960)")
    parser.add_argument("--batch", type=int, default=None, help="default: fl_config local_training.batch_size (8)")
    parser.add_argument("--lr0", type=float, default=None,
                        help="default: fl_config local_training.lr0 (0.01, the value A0 confirmed healthy)")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="override configs/dp_config.yaml dp_sgd.max_grad_norm (default: 1.0, per spec -- C tuning is a LATER step)")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="noise multiplier (default 0.0 = clipping-only diagnostic B, no privacy "
                             "guarantee). Passing e.g. --sigma 0.5 turns this into 'diagnostic C': the "
                             "exact same clipping-only config/instrumentation but with the real E2 "
                             "sigma=0.5 noise added, to isolate noise's effect from clipping's -- use "
                             "--tag to route it to its own output path")
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=0,
                        help="dataloader workers (default 0: matches the validated E1/E2 DP scripts and "
                             "scripts/19_dp_smoke_test.py -- workers>0 crashes on Windows because Opacus' "
                             "locally-defined collate fn can't be pickled into spawned workers)")
    parser.add_argument("--freeze-stages", type=int, nargs="*", default=None,
                        help="override configs/dp_config.yaml variants.partial.freeze_stages (default: "
                             "read from there, i.e. P0/current-E2 [0..10]). Pass explicit stage indices "
                             "for a different trainable subset, e.g. P2 (frozen: all except 16,19,22,23)")
    parser.add_argument("--subset-label", default="P0",
                        help="label for this trainable-subset candidate (P0/P1/P2/...) -- purely for "
                             "output filename/record clarity, does not affect the probe itself")
    args = parser.parse_args()

    if args.rounds > 5:
        print(f"[!] --rounds {args.rounds} > 5: this is meant to stay a short diagnostic, per the agreed protocol")

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    freeze_stages = (args.freeze_stages if args.freeze_stages is not None
                    else dp_cfg["variants"]["partial"]["freeze_stages"])  # default: same source as A0/E2
    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, warmup_epochs=0.0, workers=args.workers)
    if args.batch:
        hyp["batch_size"] = args.batch
    if args.lr0 is not None:
        hyp["lr0"] = args.lr0

    max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else dp_cfg["dp_sgd"]["max_grad_norm"]
    dp_hyp = {
        "sigma": args.sigma,  # default 0.0 = clipping-only; --sigma > 0 isolates noise's added effect
        "max_grad_norm": max_grad_norm,
        "delta": dp_cfg["dp_sgd"]["delta"],
        "accountant": dp_cfg["dp_sgd"]["accountant"],
    }

    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    clients_dir = splits_dir / "federated_partitions" / "k4_clients"
    client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
    client_sample_counts = dict(manifest["sizes"])

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = f"runs/diag_b_clipping_only/k4_seed{args.seed}{suffix}"

    accountant_states: dict = {}  # per-client, persisted across rounds (same mechanism as the real sweep)

    def client_round_fn(client_id, data_yaml_c, global_weights_path, round_idx, out_dir_c):
        state_dict, info = train_client_round_dp(
            global_weights_path, data_yaml_c, hyp, dp_hyp, round_idx, client_id, out_dir_c,
            device=args.device, freeze_stages=freeze_stages,
            accountant_state=accountant_states.get(client_id),
            collect_grad_norms=True,
        )
        accountant_states[client_id] = info.pop("accountant_state")
        return state_dict, info

    def eval_fn(weights_path):
        # VALIDATION only; this script has no test-eval path anywhere
        return evaluate_detector(weights_path, data_yaml, split="val", imgsz=imgsz, device=args.device)

    diag_label = "B (clipping-only)" if args.sigma == 0.0 else f"C (clipping + noise, sigma={args.sigma})"
    print(f"Diagnostic {diag_label} [tag='{args.tag or '(none)'}'] -- "
          f"{args.rounds} rounds, K=4, freeze={freeze_stages}, imgsz={imgsz}, batch={hyp['batch_size']}, "
          f"lr0={hyp['lr0']}, C={max_grad_norm}, sigma={args.sigma}"
          + (" (NO noise, NO privacy guarantee)" if args.sigma == 0.0 else " (finite DP guarantee)")
          + f", workers={hyp['workers']} -> {out_dir}")
    result = run_federated_training(
        client_round_fn, client_data_yamls, client_sample_counts,
        init_weights_path="models/base_groupnorm.pt", rounds=args.rounds, out_dir=out_dir,
        eval_fn=eval_fn, eval_every=1,
    )

    # weight/freeze audit -- DP path returns live fp32 state_dicts (no EMA/fp16
    # cast, unlike Diagnostic A's non-DP path), so exact equality is the right
    # test here, same as the E1/E2 audits.
    frozen_changed, trainable_changed, n_bn, n_gn, dfl_changed = audit_freeze(
        "models/base_groupnorm.pt", result["final_weights"], freeze_stages=freeze_stages)

    val_rows = [(h["round"], (h.get("val") or {}).get("map50"), (h.get("val") or {}).get("map50_95"),
                (h.get("val") or {}).get("precision"), (h.get("val") or {}).get("recall"))
                for h in result["history"]]
    print("\n--- validation progression ---")
    print(f"{'round':>6}{'mAP50':>10}{'mAP50-95':>11}{'precision':>11}{'recall':>9}")
    for rd, m50, m95, p, r in val_rows:
        def _f(x):
            return x if x is not None else float("nan")
        print(f"{rd:>6}{_f(m50):>10.4f}{_f(m95):>11.4f}{_f(p):>11.4f}{_f(r):>9.4f}")

    print(f"\nweight/norm audit: frozen_changed={frozen_changed}  trainable_changed={trainable_changed}  "
          f"dfl_changed={dfl_changed}  BatchNorm={n_bn}  GroupNorm={n_gn}")

    # clipping-severity table, per round per client
    print("\n--- clipping severity (pre-clipping per-sample grad norm; C={:.2f}) ---".format(max_grad_norm))
    print(f"{'round':>6}{'client':>8}{'n_obs':>7}{'median':>9}{'p75':>9}{'p90':>9}{'p95':>9}{'max':>9}{'clip_frac':>11}")
    clip_rows = []
    nan_inf_any = False
    for h in result["history"]:
        for cid, info in h["clients"].items():
            nan_inf_any = nan_inf_any or info.get("nan_inf", False)
            gs = info.get("grad_norm_stats")
            row = {"round": h["round"], "client_id": cid, "grad_norm_stats": gs}
            clip_rows.append(row)
            if gs:
                print(f"{h['round']:>6}{cid:>8}{gs['n_samples_observed']:>7}{gs['median']:>9.3f}"
                      f"{gs['p75']:>9.3f}{gs['p90']:>9.3f}{gs['p95']:>9.3f}{gs['max']:>9.3f}"
                      f"{gs['clip_fraction']:>11.3f}")
            else:
                print(f"{h['round']:>6}{cid:>8}  (no grad_norm_stats -- instrumentation did not run)")

    # per-client cumulative epsilon from the final round (None for every client
    # when sigma=0, per dp_sgd.py's existing guard -- unchanged, just read here
    # instead of hardcoded, so --sigma > 0 diagnostic-C runs report it correctly)
    final_round_clients = result["history"][-1]["clients"]
    eps_per_client = {cid: info.get("epsilon") for cid, info in final_round_clients.items()}
    eps_values = [v for v in eps_per_client.values() if v is not None]
    epsilon_max = max(eps_values) if eps_values else None

    record = {
        "diagnostic": "B_clipping_only_control" if args.sigma == 0.0 else "C_clipping_plus_noise",
        "note": ("DIAGNOSTIC ONLY -- not a thesis result; sigma=0 has NO finite privacy guarantee"
                if args.sigma == 0.0 else
                "DIAGNOSTIC ONLY -- not a thesis result (short-round noise-isolation run)"),
        "rounds": args.rounds, "seed": args.seed, "k": 4,
        "freeze_stages": freeze_stages, "imgsz": imgsz, "batch_size": hyp["batch_size"],
        "lr0": hyp["lr0"], "momentum": hyp.get("momentum"), "weight_decay": hyp.get("weight_decay"),
        "epochs_per_round": hyp["epochs_per_round"], "warmup_epochs": 0.0,
        "sigma": args.sigma, "max_grad_norm": max_grad_norm, "delta": dp_hyp["delta"],
        "dp": True, "opacus": True, "clipping": True, "noise": args.sigma > 0.0,
        "epsilon_per_client_final": eps_per_client, "epsilon_max_over_clients": epsilon_max,
        "privacy_guarantee": "dp_sgd" if args.sigma > 0.0 else "not_applicable_no_noise",
        "val_per_round": [{"round": rd, "map50": m50, "map50_95": m95, "precision": p, "recall": r}
                          for rd, m50, m95, p, r in val_rows],
        "best_round": result["best_round"], "best_val_map50": result["best_val_map50"],
        "subset_label": args.subset_label,
        "frozen_region_changed": frozen_changed, "trainable_region_changed": trainable_changed,
        "dfl_changed": dfl_changed,
        "batchnorm_count": n_bn, "groupnorm_count": n_gn,
        "clipping_severity_per_client_round": clip_rows,
        "nan_inf_any": nan_inf_any,
        "test_evaluated": False,
        "history": str(Path(out_dir) / "history.json"),
        "compare_against": "results/diag_a_partial_nodp_seed42.json (A0: same config, sigma/clipping absent)",
    }
    out_json = f"results/diag_b_clipping_only_seed{args.seed}{suffix}.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nbest val mAP50={result['best_val_map50']} @ round {result['best_round']}")
    print(f"nan_inf_any={nan_inf_any}")
    print(f"Saved {out_json}")
    print("\nCompare this run's val_per_round against results/diag_a_partial_nodp_seed42.json "
          "(A0) round-by-round, e.g. with scripts/22_compare_val_trajectories.py, to determine "
          "whether clipping ALONE reproduces the E2-style collapse.")
    print("Do NOT tune C based on this run -- that is a later, separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
