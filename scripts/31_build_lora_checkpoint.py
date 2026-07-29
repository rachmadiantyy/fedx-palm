#!/usr/bin/env python3
"""PHASE D GATE 1 -- Conv-LoRA (rank 8) checkpoint surgery + full audit.

Takes the REAL models/base_groupnorm.pt (6-class, GroupNorm-converted),
wraps the P2 trainable-stage convs in ConvLoRA (see fedxpalm/models/lora.py
for the exact design), and saves a NEW isolated checkpoint:

    models/base_groupnorm_lora_r8.pt

The input checkpoint is NEVER modified; the output is refused if it already
exists (unless --force). No training happens here.

Audits (ALL must pass, each printed + saved to the audit JSON):
  1. base identity: total params == 2,591,010 (the locked P2 accounting:
     929,522 trainable + 1,661,488 frozen incl. the 16-param DFL),
     BatchNorm == 0, GroupNorm == 81.
  2. injection report: exactly which modules were wrapped (LoRA), trained
     fully (cv3.x.2), or left frozen (depthwise, DFL).
  3. initialization equivalence: model output on a fixed random input is
     BITWISE identical (torch.equal) before vs after injection -- lora_B is
     zero-initialized, so this is exact, not a tolerance test.
  4. freeze-spec audit: applying Ultralytics' EXACT freeze-name matching to
     the generated mixed freeze list yields the expected trainable set --
     .lora_ + cv3.x.2 + trainable-stage GroupNorm affine, nothing else
     (unexpected_trainable must be []).
  5. checkpoint reload equivalence: the SAVED file, loaded back fresh,
     reproduces the same bitwise-identical output and the same audit counts.
  6. fuse safety: BaseModel.fuse() under the project's GroupNorm-safe patch
     leaves the LoRA model's output bitwise unchanged (all norms are GN, so
     the patch skips every Conv wrapper without touching .conv).

Verified in advance on an architectural twin (DetectionModel(yolo11n.yaml,
nc=6) + convert_batchnorm_to_groupnorm -- reproduces the locked per-stage
param sums 32,096/86,720/378,880/431,826 EXACTLY): expected trainable
d = 123,826 (lora 116,736 + cv3.x.2 1,170 + GN affine 5,920),
sqrt(d/929,522) = 0.365 (~2.74x expected dimensional noise-norm reduction).
This script re-verifies every number on the REAL checkpoint.

  python scripts/31_build_lora_checkpoint.py

Outputs:
  models/base_groupnorm_lora_r8.pt        (new init checkpoint)
  results/lora_r8_surgery_audit.json      (full audit record)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe fuse() patch)
from fedxpalm.models.lora import (  # noqa: E402
    audit_lora_params, inject_conv_lora, lora_freeze_spec)

EXPECTED_BASE_TOTAL = 2_591_010     # 929,522 trainable + 1,661,488 frozen (incl DFL 16)
EXPECTED_TRAINABLE_D = 123_826      # lora 116,736 + cv3.x.2 1,170 + GN affine 5,920 (twin-verified)


def _flat_out(y):
    if torch.is_tensor(y):
        return [y]
    if isinstance(y, (list, tuple)):
        out = []
        for i in y:
            out += _flat_out(i)
        return out
    if isinstance(y, dict):
        out = []
        for i in y.values():
            out += _flat_out(i)
        return out
    return []


def _outputs_equal(a, b) -> bool:
    ta, tb = _flat_out(a), _flat_out(b)
    return len(ta) == len(tb) and all(torch.equal(x, y) for x, y in zip(ta, tb))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/base_groupnorm.pt")
    parser.add_argument("--out", default="models/base_groupnorm_lora_r8.pt")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out file")
    parser.add_argument("--imgsz", type=int, default=960, help="forward-equivalence test input size")
    args = parser.parse_args()

    if not Path(args.base).exists():
        print(f"FAIL: {args.base} not found"); return 1
    if Path(args.out).exists() and not args.force:
        print(f"FAIL: {args.out} already exists -- refusing to overwrite (use --force only if intended)")
        return 1

    failures = []

    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    orig_dtype = next(model.parameters()).dtype
    model = model.float().eval()

    # ---- audit 1: base identity ----
    base_total = sum(p.numel() for p in model.parameters())
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    print(f"[1] base: total_params={base_total} (expected {EXPECTED_BASE_TOTAL})  BN={n_bn}  GN={n_gn}")
    if base_total != EXPECTED_BASE_TOTAL:
        failures.append(f"base total {base_total} != {EXPECTED_BASE_TOTAL}")
    if n_bn != 0 or n_gn != 81:
        failures.append(f"norm layout BN={n_bn}/GN={n_gn} != 0/81")

    torch.manual_seed(0)
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        ref = model(x)

    # ---- audit 2: injection ----
    report = inject_conv_lora(model, rank=args.rank)
    print(f"[2] injected: wrapped={report['n_wrapped']}  lora_params={report['n_lora_params']}  "
          f"full_train={report['full_train']}  frozen_dw={len(report['frozen_dw'])}  "
          f"frozen_dfl={report['frozen_dfl']}")

    # ---- audit 3: initialization equivalence (bitwise) ----
    with torch.no_grad():
        post = model(x)
    init_equiv = _outputs_equal(ref, post)
    print(f"[3] forward bitwise-identical after injection: {init_equiv}")
    if not init_equiv:
        failures.append("post-injection forward differs from base (must be bitwise identical)")

    # ---- audit 4: freeze-spec / trainable-set audit ----
    spec = lora_freeze_spec(model)
    audit = audit_lora_params(model, spec)
    print(f"[4] freeze spec: {sum(1 for s in spec if isinstance(s, int))} stage ints + "
          f"{sum(1 for s in spec if not isinstance(s, int))} module strings")
    print(f"    n_trainable={audit['n_trainable']} (expected {EXPECTED_TRAINABLE_D})  "
          f"n_frozen={audit['n_frozen']}")
    print(f"    breakdown={audit['trainable_breakdown']}  unexpected={audit['unexpected_trainable']}")
    if audit["n_trainable"] != EXPECTED_TRAINABLE_D:
        failures.append(f"trainable {audit['n_trainable']} != expected {EXPECTED_TRAINABLE_D}")
    if audit["unexpected_trainable"]:
        failures.append(f"unexpected trainable params: {audit['unexpected_trainable'][:5]}")
    if audit["n_trainable"] + audit["n_frozen"] != base_total + report["n_lora_params"]:
        failures.append("param conservation failed (trainable+frozen != base+lora)")

    # ---- save (never touching the input path) ----
    model_to_save = model.to(orig_dtype)
    ckpt_out = dict(ckpt) if isinstance(ckpt, dict) else {"model": None}
    ckpt_out["model"] = model_to_save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_out, args.out)
    print(f"saved {args.out} (input {args.base} untouched)")

    # ---- audit 5: reload equivalence ----
    re_ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    re_model = re_ckpt["model"].float().eval()
    with torch.no_grad():
        re_out = re_model(x)
    reload_equiv = _outputs_equal(ref, re_out)
    re_audit = audit_lora_params(re_model, lora_freeze_spec(re_model))
    print(f"[5] reload: forward bitwise-identical={reload_equiv}  "
          f"n_trainable={re_audit['n_trainable']}")
    if not reload_equiv:
        failures.append("reloaded checkpoint forward differs from base")
    if re_audit["n_trainable"] != audit["n_trainable"]:
        failures.append("reloaded trainable count differs")

    # ---- audit 6: fuse safety under the GroupNorm-safe patch ----
    try:
        re_model.fuse(verbose=False)
        with torch.no_grad():
            fused_out = re_model(x)
        fuse_equiv = _outputs_equal(ref, fused_out)
        print(f"[6] fuse(): OK, forward bitwise-identical={fuse_equiv}")
        if not fuse_equiv:
            failures.append("fuse() changed the model output")
    except Exception as e:  # noqa: BLE001
        fuse_equiv = False
        failures.append(f"fuse() crashed: {type(e).__name__}: {e}")
        print(f"[6] fuse() FAILED: {type(e).__name__}: {e}")

    record = {
        "gate": "phase_d_gate1_lora_surgery",
        "base_checkpoint": args.base, "output_checkpoint": args.out, "rank": args.rank,
        "base_total_params": int(base_total), "batchnorm_count": n_bn, "groupnorm_count": n_gn,
        "injection_report": report,
        "freeze_spec": spec,
        "trainable_audit": audit,
        "expected_trainable_d": EXPECTED_TRAINABLE_D,
        "init_forward_bitwise_identical": init_equiv,
        "reload_forward_bitwise_identical": reload_equiv,
        "fuse_forward_bitwise_identical": fuse_equiv,
        "failures": failures,
        "passed": not failures,
    }
    Path("results").mkdir(exist_ok=True)
    out_json = "results/lora_r8_surgery_audit.json"
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {out_json}")
    for msg in failures:
        print(f"FAIL: {msg}")
    print("SURGERY AUDIT: " + ("PASSED" if not failures else "FAILED"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
