#!/usr/bin/env python3
"""Pre-training structural + mechanism audit for YOLO11s under the P2
trainable-subset design (trainable stages {16,19,22,23}), BEFORE any real
training pilot is run. Nothing here assumes YOLO11s's stage graph matches
YOLO11n's -- every claim below is verified by constructing the real
architecture and inspecting it.

Loads the PUBLIC pretrained yolo11s.pt (auto-downloaded by Ultralytics if
missing, same mechanism scripts/04_prepare_base_model.py already uses),
rebuilds the head for this project's nc=6 (keeping shape-compatible
pretrained backbone/neck tensors, head layers left at init -- identical
recipe to the existing YOLO11n base), converts BatchNorm->GroupNorm, and
runs every check requested:
  1. prints the full stage table (index, module type, param count,
     trainable/frozen under the P2 design, backbone/neck/head region)
  2. confirms the Detect head's actual input stages (model.model[-1].f)
     match {16, 19, 22} -- the graph fact P2's stage choice depends on --
     rather than assuming YOLO11s's graph is identical to YOLO11n's
  3. converts BatchNorm->GroupNorm and reports the ACTUAL before/after
     counts (never hardcoded/assumed to equal YOLO11n's 81)
  4. runs Opacus's ModuleValidator explicitly and prints every error if
     the model does not pass cleanly
  5. computes total/trainable/frozen/DFL parameter counts from the
     REAL measured architecture (not estimated)
  6. runs a SHORT REAL update (2 logical DP-SGD steps, sigma=0 -- pure
     mechanism check, no privacy claim) via the exact same PRODUCTION
     path (train_client_round_dp, unmodified) used by every other
     experiment in this project, then verifies: frozen region byte-
     identical, trainable region changed, DFL byte-identical, no NaN/Inf

This is NOT a zero-weight-update probe (unlike scripts/25/28): a few
real parameter updates are expected and required here to prove the
trainable region actually trains, matching scripts/36's audit style.
Writes to a SCRATCH run directory only; does not touch
models/base_groupnorm.pt or any existing YOLO11n result.

  python scripts/37_audit_yolo11s_p2.py --device 0

Output: results/yolo11s_p2_structural_audit.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (GroupNorm-safe fuse() patch)
from fedxpalm.models.groupnorm import (  # noqa: E402
    convert_batchnorm_to_groupnorm, count_batchnorm_layers, disable_inplace_ops)
from fedxpalm.models.model_variant import detect_model_variant, resolve_model_variant  # noqa: E402
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import stage_of  # noqa: E402

P2_TRAINABLE_STAGES = [16, 19, 22, 23]
P2_FROZEN_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
BACKBONE_MAX_STAGE = 10  # same convention as freeze_audit.py -- 0-10 backbone, 11-22 neck, 23 head


def region_of(stage: int) -> str:
    if stage <= BACKBONE_MAX_STAGE:
        return "backbone"
    if stage <= 22:
        return "neck"
    return "head"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--arch", default="yolo11s.pt", help="public pretrained checkpoint to audit")
    parser.add_argument("--save-base-as", default=None,
                        help="if given, also saves the converted (GroupNorm, nc=6) model to this path "
                             "for reuse as --init-weights in scripts/21 and /23 -- e.g. "
                             "models/base_groupnorm_yolo11s.pt. Refuses to overwrite an existing file")
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    print(f"=== Loading public pretrained checkpoint {args.arch} (auto-downloads if missing) ===")
    yolo = YOLO(args.arch)
    pretrained_sd = yolo.model.state_dict()
    yolo.model = DetectionModel(yolo.model.yaml, nc=ds_cfg["nc"])
    compatible = {k: v for k, v in pretrained_sd.items()
                 if k in yolo.model.state_dict() and yolo.model.state_dict()[k].shape == v.shape}
    yolo.model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(yolo.model.state_dict())} pretrained tensors "
          f"(head layers differ due to nc={ds_cfg['nc']} vs COCO's 80 and are left at init)")

    model = yolo.model
    n_stages = len(model.model)
    detect = model.model[-1]
    detect_stage_idx = n_stages - 1
    detect_inputs = list(getattr(detect, "f", []))
    print(f"\n=== Graph check: Detect head inputs ===")
    print(f"n_stages={n_stages}  Detect stage index={detect_stage_idx}  Detect.f={detect_inputs}")
    graph_matches_p2 = (detect_stage_idx == 23 and sorted(detect_inputs) == [16, 19, 22])
    if not graph_matches_p2:
        print(f"[!] WARNING: this architecture's Detect-head graph does NOT match the assumed "
              f"P2 stage choice ({{16,19,22,23}}) -- do NOT proceed with the P2 freeze pattern "
              f"unless you re-derive the correct stages from the values printed above")
    else:
        print("CONFIRMED: Detect head consumes stages {16,19,22} exactly as P2's design assumes "
              "for YOLO11n -- P2's stage indices are valid for this architecture too.")

    # ---- BatchNorm -> GroupNorm conversion: report ACTUAL counts, never assumed ----
    n_bn_before = count_batchnorm_layers(model)
    convert_batchnorm_to_groupnorm(model, max_groups=dp_cfg["groupnorm"]["max_groups"])
    disable_inplace_ops(model)
    n_bn_after = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn_after = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    print(f"\n=== GroupNorm conversion ===")
    print(f"BatchNorm before={n_bn_before}  BatchNorm after={n_bn_after}  GroupNorm after={n_gn_after}")
    if n_bn_after != 0:
        print(f"FAIL: {n_bn_after} BatchNorm layers remain after conversion")
        return 1

    # ---- explicit Opacus ModuleValidator check ----
    from opacus.validators import ModuleValidator
    import copy
    validator_errors = ModuleValidator.validate(copy.deepcopy(model), strict=False)
    print(f"\n=== Opacus ModuleValidator ===")
    print(f"errors: {len(validator_errors)}")
    for e in validator_errors:
        print(f"  {e}")
    if validator_errors:
        print("FAIL: model does not pass Opacus ModuleValidator")
        return 1

    # ---- stage table ----
    # model_variant_requested comes from --arch's filename stem (the explicit,
    # authoritative signal of intent -- e.g. "yolo11s.pt" -> "yolo11s");
    # model_variant_detected is the measured-parameter-count fallback/sanity
    # check, never the sole source of truth (parameter count alone is not
    # reliable once the detection head is resized for a different nc).
    total_params = sum(p.numel() for p in model.parameters())
    requested_variant = Path(args.arch).stem
    variant_check = resolve_model_variant(requested_variant, total_params)
    model_variant_detected = variant_check["detected"]
    print(f"\n=== Stage table (model_variant_requested={requested_variant}, "
          f"model_variant_detected={model_variant_detected}, total_params={total_params}) ===")
    print(f"{'idx':>4}{'type':<14}{'params':>10}{'trainable?':>12}{'region':>10}")
    stage_rows = []
    dfl_params = 0
    for i, mod in enumerate(model.model):
        n_params = sum(p.numel() for p in mod.parameters())
        trainable = i in P2_TRAINABLE_STAGES
        region = region_of(i)
        print(f"{i:>4}{type(mod).__name__:<14}{n_params:>10}{str(trainable):>12}{region:>10}")
        stage_rows.append({"index": i, "type": type(mod).__name__, "params": int(n_params),
                          "trainable_under_p2": trainable, "region": region})
        if i == detect_stage_idx:
            dfl_params = sum(p.numel() for n, p in mod.named_parameters() if "dfl" in n)

    p2_trainable_total = sum(r["params"] for r in stage_rows if r["trainable_under_p2"]) - dfl_params
    frozen_total = total_params - p2_trainable_total - dfl_params
    print(f"\nP2 trainable (excl. DFL) = {p2_trainable_total}")
    print(f"frozen (excl. DFL)       = {frozen_total}")
    print(f"DFL                      = {dfl_params}")
    print(f"check: {p2_trainable_total + frozen_total + dfl_params} == total_params ({total_params}): "
          f"{p2_trainable_total + frozen_total + dfl_params == total_params}")

    if variant_check["mismatch"]:
        print(f"FAIL: --arch {args.arch!r} implies variant {requested_variant!r} but the measured "
              f"total_params={total_params} corresponds to detected variant "
              f"{model_variant_detected!r} -- refusing to proceed to the mechanism check. Verify "
              f"--arch points to the intended checkpoint.")
        return 1

    # ---- optionally persist as a reusable init checkpoint ----
    if args.save_base_as:
        out_path = Path(args.save_base_as)
        if out_path.exists():
            print(f"FAIL: {out_path} already exists -- refusing to overwrite")
            return 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        yolo.save(str(out_path))
        print(f"\nSaved reusable init checkpoint to {out_path}")

    # ---- short REAL mechanism check (2 logical DP-SGD steps, sigma=0) ----
    # save a throwaway checkpoint for train_client_round_dp to load (it takes a
    # path, not a live model)
    scratch_dir = Path("runs/_yolo11s_p2_audit_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_ckpt = scratch_dir / "init.pt"
    torch.save({"model": model}, scratch_ckpt)

    with open("configs/dataset.yaml") as f:
        ds_cfg2 = yaml.safe_load(f)
    splits_dir = Path(ds_cfg2["output_dir"])
    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        print(f"\n[!] {manifest_path} not found -- skipping the short real-update mechanism check "
              f"(structural audit above still stands). Run this on the machine with the real "
              f"dataset for the full audit before any training pilot.")
        record = {
            "note": "STRUCTURAL AUDIT ONLY -- mechanism check skipped, no dataset available here",
            "arch": args.arch,
            "model_variant_requested": requested_variant, "model_variant_detected": model_variant_detected,
            "total_params": int(total_params),
            "detect_stage_index": detect_stage_idx, "detect_inputs": detect_inputs,
            "graph_matches_p2_design": graph_matches_p2,
            "batchnorm_before": n_bn_before, "batchnorm_after": n_bn_after, "groupnorm_after": n_gn_after,
            "module_validator_errors": [str(e) for e in validator_errors],
            "stage_table": stage_rows,
            "p2_trainable_params": int(p2_trainable_total), "frozen_params": int(frozen_total),
            "dfl_params": int(dfl_params),
        }
        Path("results").mkdir(exist_ok=True)
        with open("results/yolo11s_p2_structural_audit.json", "w") as f:
            json.dump(record, f, indent=2)
        print("Saved results/yolo11s_p2_structural_audit.json")
        return 0

    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    client0 = sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    data_yaml = str(splits_dir / "federated_partitions" / "k4_clients" / f"client{client0}" / "data.yaml")

    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, warmup_epochs=0.0, workers=0, batch_size=8)
    dp_hyp = {"sigma": 0.0, "max_grad_norm": dp_cfg["dp_sgd"]["max_grad_norm"],
             "delta": dp_cfg["dp_sgd"]["delta"], "accountant": dp_cfg["dp_sgd"]["accountant"]}

    print(f"\n=== Short real mechanism check: 2 logical steps, client{client0}, sigma=0 (no privacy claim) ===")
    final_state, info = train_client_round_dp(
        global_weights_path=str(scratch_ckpt), client_data_yaml=data_yaml,
        hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id=f"audit_{client0}",
        out_dir=str(scratch_dir), device=args.device, freeze_stages=P2_FROZEN_STAGES,
        accountant_state=None, max_steps=2,
    )

    init_sd = {k.replace("_module.", "", 1): v.detach().clone() for k, v in model.state_dict().items()}
    freeze_names = [f"model.{s}." for s in P2_FROZEN_STAGES] + [".dfl"]
    frozen_changed = False
    trainable_changed = False
    dfl_changed = False
    for k in final_state:
        if k not in init_sd or init_sd[k].shape != final_state[k].shape:
            continue
        differs = not torch.equal(init_sd[k].float(), final_state[k].detach().cpu().float())
        if "dfl.conv" in k:
            dfl_changed = dfl_changed or differs
            continue
        is_frozen = any(x in k for x in freeze_names)
        if is_frozen and differs:
            frozen_changed = True
        if not is_frozen and differs:
            trainable_changed = True

    print(f"frozen_changed={frozen_changed} (must be False)")
    print(f"trainable_changed={trainable_changed} (must be True)")
    print(f"dfl_changed={dfl_changed} (must be False)")
    print(f"nan_inf={info['nan_inf']} (must be False)")

    mechanism_failures = []
    if frozen_changed:
        mechanism_failures.append("frozen region changed")
    if not trainable_changed:
        mechanism_failures.append("trainable region did NOT change")
    if dfl_changed:
        mechanism_failures.append("DFL changed")
    if info["nan_inf"]:
        mechanism_failures.append("NaN/Inf detected")

    record = {
        "arch": args.arch,
        "model_variant_requested": requested_variant, "model_variant_detected": model_variant_detected,
        "total_params": int(total_params),
        "detect_stage_index": detect_stage_idx, "detect_inputs": detect_inputs,
        "graph_matches_p2_design": graph_matches_p2,
        "batchnorm_before": n_bn_before, "batchnorm_after": n_bn_after, "groupnorm_after": n_gn_after,
        "module_validator_errors": [str(e) for e in validator_errors],
        "stage_table": stage_rows,
        "p2_trainable_stages": P2_TRAINABLE_STAGES, "p2_frozen_stages": P2_FROZEN_STAGES,
        "p2_trainable_params": int(p2_trainable_total), "frozen_params": int(frozen_total),
        "dfl_params": int(dfl_params),
        "mechanism_check": {
            "client_id": client0, "logical_steps": 2, "sigma": 0.0,
            "frozen_changed": frozen_changed, "trainable_changed": trainable_changed,
            "dfl_changed": dfl_changed, "nan_inf": info["nan_inf"],
            "failures": mechanism_failures, "passed": not mechanism_failures,
        },
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/yolo11s_p2_structural_audit.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved results/yolo11s_p2_structural_audit.json")
    for msg in mechanism_failures:
        print(f"FAIL: {msg}")
    print("YOLO11s P2 STRUCTURAL + MECHANISM AUDIT: " + ("PASSED" if not mechanism_failures else "FAILED"))
    return 0 if not mechanism_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
