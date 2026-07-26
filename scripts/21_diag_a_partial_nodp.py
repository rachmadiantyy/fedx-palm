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
from fedxpalm.models.model_variant import detect_model_variant, resolve_model_variant  # noqa: E402

# Frozen tolerance covers a worst-case single fp16 boundary flip on a
# large-magnitude weight (ulp is ~2e-3 for |w| in [2,4)) -- EMA rounding drift
# itself is ~5e-7 and normally fully absorbed by the fp16 cast, so the
# expected observed value is exactly 0.0 (changed=0 params). Real training at
# lr0=0.01 moves neck/head weights by >>1e-1 over several rounds, so 3e-3
# still separates the two regimes by ~2 orders of magnitude.
FROZEN_TOL = 3e-3
TRAINED_MIN = 1e-2


def region_of(key: str, freeze_stages: list) -> str | None:
    """"frozen"/"trainable" derived from THIS run's actual freeze list, using
    ULTRALYTICS' OWN matching semantics (substring of f"model.{x}."), not a
    hardcoded backbone/neck_head boundary -- required for P1 (freeze=[0..22],
    only stage 23 trainable) and P2-style subsets, where the frozen region
    spans backbone AND most of the neck, not just stages 0-10.

    freeze_stages entries may be ints (whole stages -- for pure-int lists
    this matching is provably identical to the previous `stage in
    freeze_stages` logic, since "model.{s}." is a substring of a key iff the
    key's stage equals s) or module-path STRINGS (e.g. "16.cv1.conv.base"
    for LoRA runs, where the frozen/trainable boundary cuts INSIDE stages:
    frozen base convs live in the same stage as their trainable .lora_
    branches, so stage membership alone cannot classify them).
    """
    if "dfl.conv" in key:
        return "dfl"
    if stage_of(key) is None:
        return None
    freeze_names = [f"model.{x}." for x in freeze_stages]
    return "frozen" if any(x in key for x in freeze_names) else "trainable"


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
    parser.add_argument("--init-weights", default="models/base_groupnorm.pt",
                        help="initial global checkpoint (default: the locked P2/B2 base). Pass "
                             "models/base_groupnorm_lora_r8.pt for the Phase D LoRA capacity control")
    parser.add_argument("--model-variant", choices=["yolo11n", "yolo11s"], default="yolo11n",
                        help="EXPLICIT scale-variant identity of --init-weights (default: yolo11n, "
                             "unchanged for every existing experiment). This -- not the measured "
                             "parameter count -- is the authoritative source recorded to the results "
                             "JSON as model_variant_requested; the measured total_params is only "
                             "cross-checked against it as a fallback sanity check (parameter count "
                             "alone cannot be trusted once the detection head is resized for nc=6). "
                             "A mismatch aborts before any training runs")
    parser.add_argument("--lora-freeze", action="store_true",
                        help="derive the freeze list from the LoRA design (fedxpalm.models.lora."
                             "lora_freeze_spec on the --init-weights model) instead of --freeze-stages: "
                             "P2 frozen stage ints + module-prefix strings for every frozen base conv "
                             "inside the trainable stages. Trainable set = .lora_ + cv3.x.2 + "
                             "trainable-stage GroupNorm affine. Requires a LoRA-surgered checkpoint")
    parser.add_argument("--predictor-only-freeze", action="store_true",
                        help="derive the freeze list from fedxpalm.models.predictor_only."
                             "predictor_only_freeze_spec on the --init-weights model (plain "
                             "base_groupnorm.pt, no surgery needed): every stage before Detect frozen "
                             "as whole ints, plus every Detect submodule EXCEPT the terminal box/class "
                             "projection convs (discovered programmatically, not hardcoded). P3 "
                             "candidate -- d=13,650 vs P2's 929,522")
    parser.add_argument("--subset-label", default=None,
                        help="label for this trainable-subset candidate (P1/P2/P3_predictor_only/...) "
                             "-- purely for the saved record's clarity, does not affect the run itself")
    parser.add_argument("--epochs-per-round", type=int, default=None,
                        help="override fl_config local_training.epochs_per_round (default: 2). For "
                             "communication-frequency-matched runs, e.g. --rounds 10 --epochs-per-round 1")
    parser.add_argument("--workers", type=int, default=0,
                        help="dataloader workers (default 0). client.py's train_client_round falls back "
                             "to workers=4 if unset here -- observed to accumulate orphaned worker "
                             "processes across repeated client-round trainer rebuilds on Windows "
                             "(each of the K*rounds calls to build_trainer_from_checkpoint()+trainer.train() "
                             "creates a fresh DataLoader; old worker processes were not reliably reaped "
                             "between calls, eventually stalling the run). workers=0 matches the "
                             "already-validated default used by every DP script (19/20/21's --lora-freeze "
                             "and --predictor-only-freeze paths, 23, 25, 28) for the same underlying "
                             "Windows multiprocessing reason")
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    if args.lora_freeze and args.predictor_only_freeze:
        print("FAIL: --lora-freeze and --predictor-only-freeze are mutually exclusive")
        return 1
    if args.lora_freeze:
        # LoRA mode: the frozen/trainable boundary cuts INSIDE stages, so the
        # freeze list is derived from the surgered model itself (ints for the
        # fully-frozen P2 stages + module-prefix strings for frozen base convs)
        import torch
        from fedxpalm.models.lora import ConvLoRA, lora_freeze_spec
        _ck = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        _lora_model = _ck["model"] if isinstance(_ck, dict) and "model" in _ck else _ck
        if not any(isinstance(m, ConvLoRA) for m in _lora_model.modules()):
            print(f"FAIL: --lora-freeze requires a LoRA-surgered checkpoint, but {args.init_weights} "
                  f"contains no ConvLoRA modules -- run scripts/31_build_lora_checkpoint.py first")
            return 1
        freeze_stages = lora_freeze_spec(_lora_model)
        del _ck, _lora_model
    elif args.predictor_only_freeze:
        # Predictor-only (P3) mode: freeze everything except the Detect head's
        # terminal box/class projection convs, discovered programmatically
        # from the actual --init-weights model (no surgery, no new modules).
        import torch
        from fedxpalm.models.predictor_only import predictor_only_freeze_spec, audit_predictor_only_params
        _ck = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        _po_model = _ck["model"] if isinstance(_ck, dict) and "model" in _ck else _ck
        freeze_stages = predictor_only_freeze_spec(_po_model)
        _po_audit = audit_predictor_only_params(_po_model, freeze_stages)
        print(f"[predictor-only pre-audit] trainable tensors: {_po_audit['n_trainable_tensors']}")
        for name, numel in _po_audit["trainable_param_names"]:
            print(f"    {name}: {numel}")
        _po_total = _po_audit["n_trainable"] + _po_audit["n_frozen"]
        print(f"[predictor-only pre-audit] n_trainable={_po_audit['n_trainable']}  "
              f"n_frozen={_po_audit['n_frozen']}  total={_po_total}  "
              f"private_fraction={_po_audit['n_trainable'] / _po_total * 100:.4f}%  "
              f"unexpected={_po_audit['unexpected_trainable']}")
        if _po_audit["unexpected_trainable"]:
            print("FAIL: unexpected trainable parameters outside the discovered terminal convs")
            return 1
        del _ck, _po_model
    else:
        freeze_stages = (args.freeze_stages if args.freeze_stages is not None
                         else dp_cfg["variants"]["partial"]["freeze_stages"])  # same source E2 reads, unless overridden
    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, seed=args.seed,
               warmup_epochs=0.0, freeze_stages=freeze_stages, workers=args.workers)
    if args.batch:
        hyp["batch_size"] = args.batch
    if args.lr0 is not None:
        hyp["lr0"] = args.lr0
    if args.epochs_per_round is not None:
        hyp["epochs_per_round"] = args.epochs_per_round

    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    clients_dir = splits_dir / "federated_partitions" / "k4_clients"
    client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
    client_sample_counts = dict(manifest["sizes"])

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = f"runs/diag_a_partial_nodp/k4_seed{args.seed}{suffix}"

    # ---- model-variant consistency check, BEFORE any training runs ----
    # --model-variant is the AUTHORITATIVE source (explicit CLI argument);
    # the measured total parameter count is only a fallback sanity-check,
    # since parameter count can shift once the detection head is resized
    # for a different nc and must never be trusted as the sole source of
    # truth for which architecture is actually being trained.
    _check_sd, _check_model = load_state(args.init_weights)
    _check_total_params = sum(p.numel() for p in _check_model.parameters())
    _variant_check = resolve_model_variant(args.model_variant, _check_total_params)
    del _check_sd, _check_model
    if _variant_check["mismatch"]:
        print(f"FAIL: --model-variant {args.model_variant!r} (requested) does not match the "
              f"measured architecture of --init-weights {args.init_weights!r} (detected: "
              f"{_variant_check['detected']!r}, total_params={_check_total_params}) -- refusing "
              f"to train. Pass the correct --model-variant, or verify --init-weights points to "
              f"the intended checkpoint.")
        return 1
    print(f"model-variant check OK: requested={args.model_variant!r} matches detected "
          f"({_variant_check['detected']!r}, total_params={_check_total_params})")

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
        init_weights_path=args.init_weights, rounds=args.rounds, out_dir=out_dir,
        eval_fn=eval_fn, eval_every=1,
    )

    # ---- post-run audits ----
    import torch.nn as nn
    base_sd, base_model = load_state(args.init_weights)
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

    # model identity metadata -- model_variant_requested comes from the
    # EXPLICIT --model-variant CLI argument (authoritative, already verified
    # to match above); model_variant_detected is the measured-parameter-count
    # fallback/sanity-check, kept here only for cross-reference, never as the
    # sole source of truth (parameter count alone is not reliable once the
    # detection head is resized for a different nc).
    total_params = sum(p.numel() for p in base_model.parameters())
    model_variant_detected = detect_model_variant(total_params)
    trainable_stage_indices = sorted(set(range(24)) - set(s for s in freeze_stages if isinstance(s, int)))

    record = {
        "model_weights": args.init_weights,
        "model_variant_requested": args.model_variant, "model_variant_detected": model_variant_detected,
        "total_params": int(total_params), "trainable_stage_indices": trainable_stage_indices,
        "dataset_manifest_path": str(manifest_path), "client_partition_manifest": dict(manifest["sizes"]),
        "diagnostic": "A_partial_nodp_frozen_backbone_control",
        "note": "DIAGNOSTIC ONLY -- not a thesis result; no DP mechanism involved",
        "subset_label": args.subset_label,
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
