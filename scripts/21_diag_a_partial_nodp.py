#!/usr/bin/env python3
"""Diagnostic A -- Partial NO-DP frozen-backbone control (short, NOT 40 rounds).

Question this answers: can the E2 architecture (backbone stages 0-10 frozen,
neck+head trainable) learn AT ALL under plain federated SGD -- no Opacus, no
per-sample clipping, no Gaussian noise, no privacy accountant? This isolates
"frozen-backbone training" from "DP-SGD effects" after the E2 sigma=0.5
40-round run collapsed (best val mAP50 0.054) despite a fully verified
freeze/optimizer implementation.

Genuinely non-DP: this reuses the LOCKED B2 local-training path
(fedxpalm.federated.client.train_client_round -> Ultralytics trainer.train())
-- the same code that produced B2's 0.885 -- with ONLY a freeze list added.
Nothing in this path imports Opacus; there is no sigma=0 trick, no
per-sample-gradient machinery, no PrivacyEngine.

Everything else mirrors the locked production setup:
  - init:        models/base_groupnorm.pt (same shared base)
  - partition:   K=4, partition seed 42 (primary federated_partitions/)
  - freeze:      configs/dp_config.yaml variants.partial.freeze_stages [0..10]
                 (read from the same config E2 reads -- single source of truth)
  - local hyp:   fl_config local_training (SGD lr0=0.01, momentum 0.9,
                 wd 5e-4, 2 local epochs, batch 8) + warmup 0 + imgsz 960,
                 i.e. the locked B2/E2 operating hyperparameters
  - FedAvg:      the same run_federated_training server loop
  - eval:        VALIDATION split only, every round; test is never read
                 (this script contains no test-eval path at all)

Outputs (all NEW namespaces -- nothing under any B1/B2/E1/E2 path):
  runs/diag_a_partial_nodp/k4_seed{seed}/        checkpoints + history.json
  results/diag_a_partial_nodp_seed{seed}.json    metrics + audits + verdict

Post-run freeze audit accounts for a checkpoint-format subtlety of this
non-DP path: Ultralytics saves last.pt from the fp16-cast EMA model, so
frozen backbone weights undergo a ONE-TIME fp16 quantization at round 0
(base -> round0 differs at ~5e-4 absolute, pure quantization). From round 0
onward every round-trip is lossless and EMA rounding drift (~5e-7) is fully
absorbed by the fp16 cast (verified empirically), so the frozen check is
round0 -> final with a 1e-3 tolerance (expected: exactly 0.0), NOT base ->
final exact equality -- that would fail spuriously on quantization alone.
The DP path (E1/E2) doesn't have this subtlety (it returns live fp32
state_dicts), which is why 19/20's exact-equality audits remain valid there.

  python scripts/21_diag_a_partial_nodp.py --device 0

LR/freeze overrides (--lr0, --freeze-stages) and --tag are ADDITIVE: omitted,
behavior and output paths are byte-identical to the original A0 reference run
(runs/diag_a_partial_nodp/k4_seed42/, results/diag_a_partial_nodp_seed42.json)
-- so A0 is never re-run or overwritten by a later A1/A2/freeze-relaxation
diagnostic. Each --tag writes to its own runs/.../k4_seed{S}_{tag}/ and
results/..._seed{S}_{tag}.json.

The weight/freeze audit and n_trainable/n_frozen counts are derived from
THIS run's own --freeze-stages (not a hardcoded backbone/neck_head split),
required for candidates like P1 (freeze=[0..22], only stage 23 trainable):
a fixed split would lump the (now frozen) neck together with the (trainable)
head into one "changed" bucket, unable to verify the frozen part actually
stayed frozen. n_trainable/n_frozen are computed by applying Ultralytics'
own freeze-name-matching logic to a fresh parameter list -- NOT read from
the returned global checkpoint's requires_grad, which is not reliable here:
only state_dict VALUES flow through FedAvg aggregation, never Parameter
objects, so the global model's requires_grad reflects whatever
base_groupnorm.pt had at save time, not what was frozen locally per client.

This is a DIAGNOSTIC: its numbers are not thesis results and must not enter
any results table or privacy-utility curve.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.client import read_local_train_log, train_client_round  # noqa: E402
from fedxpalm.federated.server import run_federated_training  # noqa: E402
from fedxpalm.privacy.freeze_audit import load_state, stage_of  # noqa: E402

# Frozen tolerance covers a worst-case single fp16 boundary flip on a
# large-magnitude weight (ulp is ~2e-3 for |w| in [2,4)) -- EMA rounding drift
# itself is ~5e-7 and normally fully absorbed by the fp16 cast, so the
# expected observed value is exactly 0.0 (changed=0 params). Real training at
# lr0=0.01 moves neck/head weights by >>1e-1 over several rounds, so 3e-3
# still separates the two regimes by ~2 orders of magnitude.
FROZEN_TOL = 3e-3
TRAINED_MIN = 1e-2


def region_of(key: str, freeze_stages: list[int]) -> str | None:
    """"frozen"/"trainable" derived from THIS run's actual freeze_stages, not
    a hardcoded backbone/neck_head boundary -- required for P1 (freeze=[0..22],
    only stage 23 trainable) and P2-style subsets, where the frozen region
    spans backbone AND most of the neck, not just stages 0-10. Using a fixed
    boundary here would silently lump a frozen region (e.g. neck under P1)
    together with the trainable region into one bucket, making a "changed"
    verdict on that bucket uninformative about whether the frozen part
    actually stayed frozen -- exactly what P1's audit needs to catch.
    """
    if "dfl.conv" in key:
        return "dfl"
    stage = stage_of(key)
    if stage is None:
        return None
    return "frozen" if stage in freeze_stages else "trainable"


def region_diff_stats(sd_a, sd_b, freeze_stages: list[int]) -> dict:
    """Per-region (frozen/trainable/dfl) max|diff| and changed-param counts
    between two state_dicts, for THIS run's freeze_stages."""
    import torch
    stats = {r: {"max_abs_diff": 0.0, "params_changed": 0, "params_total": 0}
             for r in ("frozen", "trainable", "dfl")}
    for k in sd_b:
        if k not in sd_a or sd_a[k].shape != sd_b[k].shape:
            continue
        region = region_of(k, freeze_stages)
        if region is None:
            continue
        a, b = sd_a[k].float(), sd_b[k].float()
        stats[region]["params_total"] += 1
        if not torch.equal(a, b):
            stats[region]["params_changed"] += 1
            stats[region]["max_abs_diff"] = max(stats[region]["max_abs_diff"],
                                                float((a - b).abs().max()))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--rounds", type=int, default=5, help="short diagnostic; NOT 40")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=None, help="default: fl_config model.imgsz (960)")
    parser.add_argument("--batch", type=int, default=None, help="default: fl_config local_training.batch_size (8)")
    parser.add_argument("--lr0", type=float, default=None,
                        help="override fl_config local_training.lr0 (0.01) -- Step 1 LR sweep, "
                             "freeze_stages/K/partition/batch/imgsz/momentum/wd all held fixed")
    parser.add_argument("--freeze-stages", type=int, nargs="*", default=None,
                        help="override dp_config variants.partial.freeze_stages (default: read from "
                             "configs/dp_config.yaml, same as E2). Pass stage indices e.g. --freeze-stages "
                             "0 1 2 3 4 for a shallower freeze, or --freeze-stages with NO values for no "
                             "freeze at all (Step 2 progressive-unfreeze diagnostics)")
    parser.add_argument("--tag", default="",
                         help="output-path suffix so LR/freeze variants never collide with the A0 "
                              "reference run or each other, e.g. --tag lr0.001")
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    freeze_stages = (args.freeze_stages if args.freeze_stages is not None
                     else dp_cfg["variants"]["partial"]["freeze_stages"])  # same source E2 reads, unless overridden
    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, seed=args.seed,
               warmup_epochs=0.0, freeze_stages=freeze_stages)
    if args.batch:
        hyp["batch_size"] = args.batch
    if args.lr0 is not None:
        hyp["lr0"] = args.lr0

    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    clients_dir = splits_dir / "federated_partitions" / "k4_clients"
    client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
    client_sample_counts = dict(manifest["sizes"])

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = f"runs/diag_a_partial_nodp/k4_seed{args.seed}{suffix}"

    def client_round_fn(client_id, data_yaml_c, global_weights_path, round_idx, out_dir_c):
        import torch
        weights_path = train_client_round(global_weights_path, data_yaml_c, hyp,
                                          round_idx, client_id, out_dir_c, device=args.device)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)["model"].state_dict()
        info = read_local_train_log(Path(weights_path).parent.parent)
        return state_dict, info

    def eval_fn(weights_path):
        # VALIDATION only; this script has no test-eval path anywhere
        return evaluate_detector(weights_path, data_yaml, split="val", imgsz=imgsz, device=args.device)

    print(f"Diagnostic A [tag='{args.tag or '(none, A0 reference paths)'}']: partial NO-DP "
          f"frozen-backbone control -- {args.rounds} rounds, K=4, freeze={freeze_stages}, "
          f"imgsz={imgsz}, batch={hyp['batch_size']}, lr0={hyp['lr0']}, warmup=0, "
          f"NO Opacus/clipping/noise -> {out_dir}")
    result = run_federated_training(
        client_round_fn, client_data_yamls, client_sample_counts,
        init_weights_path="models/base_groupnorm.pt", rounds=args.rounds, out_dir=out_dir,
        eval_fn=eval_fn, eval_every=1,
    )

    # ---- post-run audits ----
    import torch.nn as nn
    base_sd, _ = load_state("models/base_groupnorm.pt")
    round0_sd, _ = load_state(str(Path(out_dir) / "global_round_0.pt"))
    final_sd, final_model = load_state(result["final_weights"])
    n_bn = sum(1 for m in final_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in final_model.modules() if isinstance(m, nn.GroupNorm))

    quant = region_diff_stats(base_sd, round0_sd, freeze_stages)     # expected: one-time fp16 quantization only
    train = region_diff_stats(round0_sd, final_sd, freeze_stages)    # the real frozen/trained evidence

    fr, tr, dfl = train["frozen"], train["trainable"], train["dfl"]
    frozen_changed = fr["max_abs_diff"] > FROZEN_TOL
    trainable_changed = tr["max_abs_diff"] >= TRAINED_MIN and tr["params_changed"] > 0
    dfl_changed = dfl["max_abs_diff"] > FROZEN_TOL

    print(f"\nfreeze_stages for this run: {freeze_stages}  "
          f"(frozen params_total={fr['params_total']}, trainable params_total={tr['params_total']})")
    print("--- weight audit: base -> round0 (expected: fp16 quantization only) ---")
    for r, s in quant.items():
        print(f"  {r:<10} max|d|={s['max_abs_diff']:.2e}  changed={s['params_changed']}/{s['params_total']}")
    print("--- weight audit: round0 -> final (the frozen/trained evidence) ---")
    for r, s in train.items():
        print(f"  {r:<10} max|d|={s['max_abs_diff']:.2e}  changed={s['params_changed']}/{s['params_total']}")

    val_rows = [(h["round"], (h.get("val") or {}).get("map50"), (h.get("val") or {}).get("map50_95"),
                (h.get("val") or {}).get("precision"), (h.get("val") or {}).get("recall"))
                for h in result["history"]]
    print("\n--- validation progression ---")
    print(f"{'round':>6}{'mAP50':>10}{'mAP50-95':>11}{'precision':>11}{'recall':>9}")
    for rd, m50, m95, p, r in val_rows:
        def _f(x):
            return x if x is not None else float("nan")
        print(f"{rd:>6}{_f(m50):>10.4f}{_f(m95):>11.4f}{_f(p):>11.4f}{_f(r):>9.4f}")

    # objective (not interpreted) trend descriptor: first vs best vs last round
    map50_vals = [m for _, m, *_ in val_rows if m is not None]
    if len(map50_vals) >= 2:
        first, last, best = map50_vals[0], map50_vals[-1], max(map50_vals)
        if last >= best - 1e-6:
            trend = "IMPROVING (or flat) through the last round -- more rounds might help"
        elif last <= first + 1e-3:
            trend = "STAGNANT/DEGRADING -- best was reached early then val did not sustain it"
        else:
            trend = "PEAKED then partially receded"
        print(f"trend: first={first:.4f} best={best:.4f} last={last:.4f} -> {trend}")

    # per-client per-round training losses -- already computed by
    # read_local_train_log() and already present in history.json (nothing was
    # lost), just not previously surfaced by this script's own console/JSON
    # output. Keys are exactly as Ultralytics writes them in results.csv.
    print("\n--- per-client training losses (from results.csv via read_local_train_log) ---")
    print(f"{'round':>6}{'client':>8}{'box_loss':>11}{'cls_loss':>11}{'dfl_loss':>11}{'lr/pg0':>10}")
    train_loss_rows = []
    for h in result["history"]:
        for cid, info in h["clients"].items():
            row = {
                "round": h["round"], "client_id": cid,
                "box_loss": info.get("train/box_loss"),
                "cls_loss": info.get("train/cls_loss"),
                "dfl_loss": info.get("train/dfl_loss"),
                "lr_pg0": info.get("lr/pg0"),
            }
            train_loss_rows.append(row)
            def _f2(x):
                return x if x is not None else float("nan")
            print(f"{row['round']:>6}{row['client_id']:>8}{_f2(row['box_loss']):>11.4f}"
                  f"{_f2(row['cls_loss']):>11.4f}{_f2(row['dfl_loss']):>11.4f}{_f2(row['lr_pg0']):>10.5f}")

    failures = []
    if frozen_changed:
        failures.append(f"frozen region (stages {freeze_stages}) changed round0->final "
                        f"(max|d|={fr['max_abs_diff']:.2e} > {FROZEN_TOL})")
    if not trainable_changed:
        failures.append(f"trainable region did NOT train (max|d|={tr['max_abs_diff']:.2e})")
    if dfl_changed:
        failures.append(f"DFL changed round0->final (max|d|={dfl['max_abs_diff']:.2e})")
    if n_bn != 0:
        failures.append(f"BatchNorm={n_bn} (must be 0)")
    if n_gn == 0:
        failures.append("GroupNorm=0 (must be >0)")

    # NOTE: requires_grad is NOT reliable on final_model here -- it is never
    # touched by server.py/client.py (only state_dict VALUES flow through
    # FedAvg aggregation, not Parameter objects), so the global checkpoint's
    # requires_grad is whatever base_groupnorm.pt had at save time (all
    # trainable except DFL), regardless of what freeze_stages was applied
    # LOCALLY inside each client's own rebuilt trainer. Compute the true
    # trainable/frozen counts by applying the SAME freeze-name-matching logic
    # Ultralytics' own BaseTrainer._setup_train() uses, on a fresh copy.
    n_trainable, n_frozen = 0, 0
    freeze_names = [f"model.{s}." for s in freeze_stages] + [".dfl"]
    for name, p in final_model.named_parameters():
        if any(x in name for x in freeze_names):
            n_frozen += p.numel()
        else:
            n_trainable += p.numel()

    record = {
        "diagnostic": "A_partial_nodp_frozen_backbone_control",
        "note": "DIAGNOSTIC ONLY -- not a thesis result; no DP mechanism involved",
        "rounds": args.rounds, "seed": args.seed, "k": 4,
        "freeze_stages": freeze_stages, "imgsz": imgsz, "batch_size": hyp["batch_size"],
        "lr0": hyp["lr0"], "momentum": hyp.get("momentum"), "weight_decay": hyp.get("weight_decay"),
        "epochs_per_round": hyp["epochs_per_round"], "warmup_epochs": 0.0,
        "dp": False, "opacus": False, "clipping": False, "noise": False, "accountant": None,
        "n_trainable_params": int(n_trainable), "n_frozen_params": int(n_frozen),
        "val_per_round": [{"round": rd, "map50": m50, "map50_95": m95, "precision": p, "recall": r}
                          for rd, m50, m95, p, r in val_rows],
        "train_losses_per_client_round": train_loss_rows,
        "best_round": result["best_round"], "best_val_map50": result["best_val_map50"],
        "audit_base_to_round0_quantization": quant,
        "audit_round0_to_final": train,
        "frozen_region_changed": frozen_changed, "trainable_region_changed": trainable_changed,
        "dfl_changed": dfl_changed, "batchnorm_count": n_bn, "groupnorm_count": n_gn,
        "mechanism_failures": failures,
        "test_evaluated": False,
        "history": str(Path(out_dir) / "history.json"),
    }
    out_json = f"results/diag_a_partial_nodp_seed{args.seed}{suffix}.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nfrozen_region_changed={frozen_changed}  trainable_region_changed={trainable_changed}  "
          f"dfl_changed={dfl_changed}  BatchNorm={n_bn}  GroupNorm={n_gn}")
    print(f"best val mAP50={result['best_val_map50']} @ round {result['best_round']}")
    print(f"Saved {out_json}")
    for msg in failures:
        print(f"FAIL: {msg}")
    print("\nDIAG A MECHANISM: " + ("PASSED" if not failures else "FAILED"))
    print("Decision rule: if val mAP50 climbs healthily over these rounds, freezing is NOT the "
          "cause -> next diagnostic is clipping-only (C=1, sigma=0). If it stays collapsed here "
          "too, investigate freeze depth / LR interaction BEFORE any DP experiment.")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
