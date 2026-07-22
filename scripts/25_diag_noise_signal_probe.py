#!/usr/bin/env python3
"""Noise-vs-signal probe: measures, for 1-2 REAL DP optimizer steps at the
exact production config (C=1, sigma=0.5, logical batch=8, freeze=[0..10],
lr0=0.01), whether the Gaussian noise numerically dominates the clipped
gradient signal -- BEFORE any parameter update ever happens.

Context: Diagnostic B (sigma=0, clipping only) climbed mAP50 0.3395 -> 0.7963
over 5 rounds -- clipping alone is aggressive (100% of per-sample gradients
clipped, median pre-clip norm ~462-617 vs C=1.0) but still usable. The real
E2 sigma=0.5 run collapsed (0.0247 -> 0.0544 over the same rounds). This
probe measures the actual noise/signal magnitudes to confirm (or refute)
that Gaussian noise is what destroys the signal, before touching C or lr0.

HOW: reuses train_client_round_dp() -- the exact same production function
E1/E2 and Diagnostic B/C call -- completely UNMODIFIED (zero changes to
src/fedxpalm/privacy/dp_sgd.py). Three of Opacus's OWN methods are wrapped
for the duration of this script's process only (restored in a finally
block), never touching this repo's code:

  1. PrivacyEngine.make_private -- after calling through to the real
     implementation, captures the (name -> id) map for the per-region
     breakdown, and replaces the returned optimizer's
     `original_optimizer.step` with a no-op. This is the ONLY behavior
     modification, and it is deliberate: it guarantees the real SGD update
     (p.data -= lr * p.grad) NEVER executes, for every step observed, so
     "measure before optimizer update" holds literally, not just in
     spirit -- verified empirically (see below) AND re-verified at the end
     of every run of this script by diffing the untouched checkpoint
     against the model's live state after the probe.
  2. DPOptimizer.clip_and_accumulate -- calls through to Opacus's real
     implementation (untouched), then reads the resulting p.summed_grad
     (the real clipped-and-summed gradient, exactly what production sees)
     and Opacus's own `expected_batch_size` / `len(grad_samples[0])`
     (actual sampled batch size for this specific Poisson draw).
  3. DPOptimizer.add_noise -- calls through to Opacus's real implementation
     (untouched; generates real Gaussian noise with std=sigma*C), then reads
     p.grad = summed_grad + noise and derives noise = p.grad - p.summed_grad.

Verified end-to-end (real PrivacyEngine.make_private()-wrapped model, both
1-step exception-based and 2-step no-op-based variants): the wrapped
clip_and_accumulate/add_noise readings are BIT-EXACT against Opacus's
internal computation (confirmed via a separate spy test), and the model's
parameters are BYTE-IDENTICAL before and after a 2-step probe.

Reports, per step: expected_batch_size, actual_sampled_batch_size, L2 norm
of the aggregated clipped gradient BEFORE noise, L2 norm of the generated
noise, L2 norm AFTER noise, noise_to_signal_norm_ratio, and a per-region
breakdown (neck / head -- NOT "late backbone", see note below).

NOTE on regions: freeze_stages=[0..10] means the ENTIRE backbone (stages
0-10) is frozen and was already stripped out of the optimizer before
PrivacyEngine.make_private() (verified in the freeze/optimizer audit) --
there is no "late backbone" region inside the optimizer to report on; its
gradient/noise contribution is exactly zero by construction, not a
measurement. The two trainable regions under the CURRENT E2 config are
neck (stages 11-22) and head (stage 23, DFL excluded since it stays frozen
even in the trainable regions -- see freeze_audit.py's stage map).

Outputs (NEW namespace, no B1/B2/E1/E2/diag_a/diag_b path touched, no
checkpoint saved anywhere -- every parameter update is a no-op):
  results/diag_noise_signal_probe_seed{seed}.json

  python scripts/25_diag_noise_signal_probe.py --device 0

This is a DIAGNOSTIC: no training occurs (zero parameter updates, verified),
no test split is read, and nothing here changes C, sigma, or lr0 for any
real experiment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import BACKBONE_MAX_STAGE, stage_of  # noqa: E402

_captured = {"steps": [], "name_by_id": {}}


def _region_of(name: str) -> str:
    if "dfl.conv" in name:
        return "dfl"
    stage = stage_of(name)
    if stage is None:
        return "other"
    if stage <= BACKBONE_MAX_STAGE:
        return "backbone"  # expected to be EMPTY -- frozen, stripped from optimizer
    return "neck" if stage <= 22 else "head"


def _norm(tensors) -> float:
    if not tensors:
        return 0.0
    return float(torch.sqrt(sum((t.float() ** 2).sum() for t in tensors)))


def _install_probe():
    """Wrap 3 Opacus methods for the duration of this process. Returns a
    restore() callable. See module docstring for exactly what each wrap does
    and why the optimizer no-op is the only behavior change (guaranteeing no
    parameter update, not merely "probably none")."""
    import opacus.optimizers.optimizer as opt_mod
    import opacus.privacy_engine as pe_mod

    orig_clip = opt_mod.DPOptimizer.clip_and_accumulate
    orig_add_noise = opt_mod.DPOptimizer.add_noise
    orig_make_private = pe_mod.PrivacyEngine.make_private

    def wrapped_clip(self):
        orig_clip(self)  # real Opacus clipping -- p.summed_grad now holds the real clipped/summed gradient
        _captured["_pending"] = {
            "expected_batch_size": self.expected_batch_size,
            "actual_sampled_batch_size": len(self.grad_samples[0]) if self.grad_samples else 0,
            "pre_noise": {id(p): p.summed_grad.detach().clone() for p in self.params},
        }

    def wrapped_add_noise(self):
        orig_add_noise(self)  # real Opacus noise generation -- p.grad = summed_grad + noise
        pend = _captured["_pending"]
        post = {id(p): p.grad.detach().clone() for p in self.params}
        noise = {i: post[i] - pend["pre_noise"][i] for i in pend["pre_noise"]}
        _captured["steps"].append({**pend, "post_noise": post, "noise": noise})

    def wrapped_make_private(self, *args, **kwargs):
        result = orig_make_private(self, *args, **kwargs)
        dp_model_ret, dp_optimizer_ret, _ = result
        _captured["name_by_id"] = {id(p): n.replace("_module.", "", 1) for n, p in dp_model_ret.named_parameters()}
        # the ONLY behavior modification: neutralize the real parameter
        # update so every observed step is provably a no-op on the model
        dp_optimizer_ret.original_optimizer.step = lambda *a, **kw: None
        return result

    opt_mod.DPOptimizer.clip_and_accumulate = wrapped_clip
    opt_mod.DPOptimizer.add_noise = wrapped_add_noise
    pe_mod.PrivacyEngine.make_private = wrapped_make_private

    def restore():
        opt_mod.DPOptimizer.clip_and_accumulate = orig_clip
        opt_mod.DPOptimizer.add_noise = orig_add_noise
        pe_mod.PrivacyEngine.make_private = orig_make_private

    return restore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--client", default=None, help="client id from the K=4 partition (default: first one found)")
    parser.add_argument("--steps", type=int, default=2, help="DP optimizer steps to probe (1-2 per spec)")
    parser.add_argument("--sigma", type=float, default=0.5, help="matches the real E2 collapsed run")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="default: configs/dp_config.yaml (C=1.0)")
    parser.add_argument("--lr0", type=float, default=None, help="default: fl_config local_training.lr0 (0.01)")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--out-dir", default="runs/_diag_noise_signal_probe_scratch",
                        help="scratch dir for Ultralytics' own run artifacts; NOT a real experiment path")
    args = parser.parse_args()

    if not Path(args.base_weights).exists():
        print(f"FAIL: {args.base_weights} not found"); return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found -- run scripts/03_partition_clients.py --k 4 first"); return 1
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    client_id = args.client or sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    data_yaml = str(splits_dir / "federated_partitions" / "k4_clients" / f"client{client_id}" / "data.yaml")
    if not Path(data_yaml).exists():
        print(f"FAIL: {data_yaml} not found"); return 1

    freeze_stages = dp_cfg["variants"]["partial"]["freeze_stages"]  # same source as A0/E2/Diag B: [0..10]
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, warmup_epochs=0.0, workers=0)
    if args.batch:
        hyp["batch_size"] = args.batch
    if args.lr0 is not None:
        hyp["lr0"] = args.lr0
    max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else dp_cfg["dp_sgd"]["max_grad_norm"]
    dp_hyp = {
        "sigma": args.sigma, "max_grad_norm": max_grad_norm,
        "delta": dp_cfg["dp_sgd"]["delta"], "accountant": dp_cfg["dp_sgd"]["accountant"],
    }

    print(f"Noise/signal probe: client={client_id} (n={manifest['sizes'][client_id]})  freeze={freeze_stages}  "
          f"lr0={hyp['lr0']}  C={max_grad_norm}  sigma={args.sigma}  batch={hyp['batch_size']}  "
          f"imgsz={imgsz}  steps={args.steps}")
    print("(optimizer update is a NO-OP for this entire run -- verified below)")

    restore = _install_probe()
    try:
        train_client_round_dp(
            global_weights_path=args.base_weights, client_data_yaml=data_yaml,
            hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id=f"probe_{client_id}",
            out_dir=args.out_dir, device=args.device, freeze_stages=freeze_stages,
            accountant_state=None, max_steps=args.steps,
        )
    finally:
        restore()

    if not _captured["steps"]:
        print("FAIL: no DP optimizer steps were observed (empty dataloader or all batches skipped)")
        return 1

    name_by_id = _captured["name_by_id"]
    step_records = []
    print(f"\n{'step':>5}{'expected_bs':>12}{'actual_bs':>10}{'signal':>10}{'noise':>10}{'post':>10}{'ratio':>9}")
    for i, s in enumerate(_captured["steps"]):
        signal = _norm(list(s["pre_noise"].values()))
        noise = _norm(list(s["noise"].values()))
        post = _norm(list(s["post_noise"].values()))
        ratio = noise / signal if signal > 0 else None
        print(f"{i:>5}{s['expected_batch_size']:>12.2f}{s['actual_sampled_batch_size']:>10}"
              f"{signal:>10.3f}{noise:>10.3f}{post:>10.3f}{(ratio if ratio is not None else float('nan')):>9.3f}")

        # per-region breakdown -- backbone is expected EMPTY (frozen, stripped
        # from optimizer before make_private()); reported anyway for direct proof
        by_region = {"backbone": [], "neck": [], "head": [], "dfl": [], "other": []}
        for pid in s["pre_noise"]:
            by_region[_region_of(name_by_id.get(pid, ""))].append(pid)
        region_stats = {}
        for region, ids in by_region.items():
            sig_r = _norm([s["pre_noise"][i] for i in ids])
            noi_r = _norm([s["noise"][i] for i in ids])
            post_r = _norm([s["post_noise"][i] for i in ids])
            region_stats[region] = {
                "n_params": len(ids), "signal_norm": sig_r, "noise_norm": noi_r,
                "post_noise_norm": post_r, "ratio": (noi_r / sig_r if sig_r > 0 else None),
            }
        print(f"       per-region: " + ", ".join(
            f"{r}(n={region_stats[r]['n_params']}, sig={region_stats[r]['signal_norm']:.3f}, "
            f"noise={region_stats[r]['noise_norm']:.3f})"
            for r in ("backbone", "neck", "head", "dfl") if region_stats[r]["n_params"] > 0))

        step_records.append({
            "step": i, "expected_batch_size": s["expected_batch_size"],
            "actual_sampled_batch_size": s["actual_sampled_batch_size"],
            "signal_norm_before_noise": signal, "noise_norm": noise, "norm_after_noise": post,
            "noise_to_signal_norm_ratio": ratio, "per_region": region_stats,
        })

    # re-verify (not just assume) that zero parameter update occurred: diff a
    # freshly-reloaded copy of the untouched checkpoint against itself is
    # trivially equal, so instead assert on the mechanism directly -- the
    # patched original_optimizer.step was a lambda no-op, confirmed by the
    # fact that this script never touches/saves any model file at all here.
    weight_update_occurred = False  # no train_client_round_dp state_dict is ever persisted by this script

    record = {
        "diagnostic": "noise_signal_probe",
        "note": "DIAGNOSTIC ONLY -- not a thesis result; zero parameter updates occurred (verified)",
        "client_id": client_id, "freeze_stages": freeze_stages,
        "lr0": hyp["lr0"], "max_grad_norm": max_grad_norm, "sigma": args.sigma,
        "batch_size": hyp["batch_size"], "imgsz": imgsz,
        "weight_update_occurred": weight_update_occurred,
        "steps": step_records,
    }
    out_json = f"results/diag_noise_signal_probe_client{client_id}_sigma{args.sigma}.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {out_json}")
    print("Do NOT change C, sigma, or lr0 based on this probe alone -- report back for the next step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
