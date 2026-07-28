#!/usr/bin/env python3
"""VERIFICATION-ONLY PROBE (no training, no checkpoint writes, no accountant
persisted, dp_sgd.py's PRODUCTION code path not modified by running this) --
matched one-logical-step comparison of THREE loss-handling generations, on
the REAL P2 model, for TWO independent clients:

  A. legacy               : loss_reduction="mean", backward on loss.sum()
                             (Ultralytics' raw mean*batch_size loss, untouched)
  B. rejected_direct_sum  : loss_reduction="sum",  backward on loss.sum()
                             (the first, REJECTED fix attempt)
  C. canonical            : loss_reduction="mean", backward on
                             loss.sum() / actual_microbatch_size
                             (the fix now in dp_sgd.py -- see its make_private()
                             comment for the full derivation)

Background: scripts/40_verify_loss_reduction_scaling.py proved (on client0
and client2) that legacy's raw grad_sample norms were inflated by exactly
the physical micro-batch size, and that "direct_sum" fixed that measurement
but removed DPOptimizer.scale_grad()'s division by expected_batch_size,
making the final per-step parameter update ~expected_batch_size too large.
A 5-round pilot under direct_sum showed unstable, oscillating precision/
recall (best mAP50 0.027 vs legacy's 0.1766) and was REJECTED (tag:
rejected_direct_sum_loss_reduction). This script checks whether "canonical"
gets BOTH right: true per-sample grad_sample (matching direct_sum) AND a
final parameter delta matching legacy's (~1/expected_batch_size of
direct_sum's) -- in every clip_fraction regime, not just by relying on
clipping's coincidental cancellation the way legacy does.

Method: for each of client0 and client2, build three independent
model/optimizer copies from the SAME checkpoint (byte-identical initial
weights), run ONE matched real logical step under Opacus's actual
BatchMemoryManager (logical batch=64, physical batch=8), using the SAME
Poisson sampling seed and the SAME noise generator seed across all three
copies so batches drawn and noise realizations are directly comparable,
not just statistically similar. No checkpoint is written, no accountant
state is persisted, and dp_sgd.py itself is not imported for its mechanism
here (only two already-audited helper functions, same as scripts/40) -- the
three modes are driven directly via Opacus's public API, matching exactly
how dp_sgd.py's own train_client_round_dp drives it.

    python scripts/41_verify_canonical_loss_normalization.py --device 0

Output: prints a comparison table per client and saves
  results/audit_dp_seedfix/canonical_loss_normalization_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from opacus import PrivacyEngine  # noqa: E402
from opacus.utils.batch_memory_manager import BatchMemoryManager  # noqa: E402

import fedxpalm  # noqa: E402,F401 (GroupNorm-safe fuse() patch)
from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint  # noqa: E402
from fedxpalm.models.groupnorm import disable_inplace_ops  # noqa: E402
from fedxpalm.privacy.dp_sgd import _strip_frozen_from_optimizer  # noqa: E402

P2_FREEZE_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]

MODES = ("legacy", "rejected_direct_sum", "canonical")


def _named_params_clean(dp_model) -> dict:
    return {n.replace("_module.", "", 1): p for n, p in dp_model.named_parameters()}


def _flatten_params(named_params: dict):
    names = sorted(named_params)
    return torch.cat([named_params[n].detach().float().reshape(-1) for n in names]), names


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _torch_device_str(device_arg: str) -> str:
    if device_arg.lower() == "cpu":
        return "cpu"
    if device_arg.startswith("cuda"):
        return device_arg
    return f"cuda:{device_arg.split(',')[0]}"


def run_one_client(base_weights, overrides, mode: str, seed: int, device: str,
                    physical_batch: int) -> dict:
    """One matched real logical step under BatchMemoryManager for a single
    (client, mode) pair. Returns per-step diagnostics; raises on NaN/Inf."""
    trainer = build_trainer_from_checkpoint(base_weights, overrides)
    disable_inplace_ops(trainer.model)
    trainer._setup_train()
    trainer.model.train()
    _strip_frozen_from_optimizer(trainer.optimizer, trainer.model)

    gen = torch.Generator(device=_torch_device_str(device))
    gen.manual_seed(seed)

    loss_reduction = "sum" if mode == "rejected_direct_sum" else "mean"
    privacy_engine = PrivacyEngine(accountant="rdp")
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=trainer.model, optimizer=trainer.optimizer, data_loader=trainer.train_loader,
        noise_multiplier=0.75, max_grad_norm=1.0, poisson_sampling=True,
        loss_reduction=loss_reduction, noise_generator=gen,
    )

    pre_flat, names = _flatten_params(_named_params_clean(dp_model))
    chunk_sizes = []
    nan_inf = False
    raw_norms_first_chunk = None
    clip_fraction_first_chunk = None
    n_iter = 0
    with BatchMemoryManager(data_loader=dp_loader, max_physical_batch_size=physical_batch,
                            optimizer=dp_optimizer) as active_loader:
        for chunk in active_loader:
            if len(chunk["img"]) == 0 or len(chunk["cls"]) == 0:
                continue
            chunk = trainer.preprocess_batch(chunk)
            n = int(chunk["img"].shape[0])
            chunk_sizes.append(n)
            dp_optimizer.zero_grad()
            loss, _ = dp_model(chunk)
            if mode == "canonical":
                total_loss = loss.sum() / n
            else:
                total_loss = loss.sum()
            if not torch.isfinite(total_loss):
                nan_inf = True
            total_loss.backward()
            # raw per-sample grad norm + clip fraction, first chunk only (cheap,
            # representative instrumentation -- matches scripts/40's pattern).
            # grad_samples is only populated AFTER backward(); read here, before
            # clip_and_accumulate() consumes it inside step() below
            if raw_norms_first_chunk is None:
                flat_gs = torch.cat([g.reshape(n, -1) for g in dp_optimizer.grad_samples], dim=1)
                norms = flat_gs.norm(dim=1)
                raw_norms_first_chunk = norms.tolist()
                clip_fraction_first_chunk = float((norms > 1.0).float().mean())
            dp_optimizer.step()
            n_iter += 1
            if not dp_optimizer._is_last_step_skipped:
                break
            if n_iter >= 64:
                raise RuntimeError(f"mode={mode}: no real logical step after {n_iter} chunks")
    if any(not torch.isfinite(p).all() for p in dp_model.parameters()):
        nan_inf = True
    post_flat, _ = _flatten_params(_named_params_clean(dp_model))
    delta = post_flat - pre_flat
    return {
        "mode": mode, "chunk_sizes": chunk_sizes, "expected_batch_size": dp_optimizer.expected_batch_size,
        "raw_grad_norms_first_chunk": raw_norms_first_chunk,
        "clip_fraction_first_chunk": clip_fraction_first_chunk,
        "delta": delta, "nan_inf": nan_inf,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--physical-batch", type=int, default=8)
    parser.add_argument("--sampling-noise-seed", type=int, default=1234,
                        help="shared seed for BOTH Poisson sampling generator and noise "
                             "generator, identical across all three modes, so batches drawn "
                             "and noise realizations are directly comparable")
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--client-ids", nargs="+", default=None,
                        help="default: client0 and client2 (the two already probed in "
                             "scripts/40). Pass e.g. --client-ids 0 2 1 3 to check all four")
    args = parser.parse_args()

    if not Path(args.base_weights).exists():
        print(f"FAIL: {args.base_weights} not found")
        return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found")
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    sorted_clients = sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)
    client_ids = args.client_ids or [sorted_clients[0], sorted_clients[2]]  # client0, client2
    for cid in client_ids:
        if cid not in manifest["sizes"]:
            print(f"FAIL: client {cid!r} not found (available: {sorted_clients})")
            return 1

    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    clients_dir = splits_dir / "federated_partitions" / "k4_clients"

    def make_overrides(data_yaml):
        return dict(
            data=data_yaml, model=args.base_weights, epochs=1, batch=64, imgsz=imgsz,
            optimizer=fl_cfg["local_training"].get("optimizer", "SGD"),
            lr0=fl_cfg["local_training"].get("lr0", 0.01),
            momentum=fl_cfg["local_training"].get("momentum", 0.9),
            weight_decay=fl_cfg["local_training"].get("weight_decay", 0.0005),
            warmup_epochs=0.0, device=args.device, workers=0, amp=False, plots=False,
            val=False, verbose=False, freeze=P2_FREEZE_STAGES, exist_ok=True,
            project="runs/_canonical_probe_scratch", name="probe",
        )

    all_results = {}
    for cid in client_ids:
        print(f"\n{'=' * 60}\nclient{cid} (n_samples={manifest['sizes'][cid]})\n{'=' * 60}")
        data_yaml = str(clients_dir / f"client{cid}" / "data.yaml")
        overrides = make_overrides(data_yaml)
        per_mode = {}
        for mode in MODES:
            r = run_one_client(args.base_weights, overrides, mode, args.sampling_noise_seed,
                               args.device, args.physical_batch)
            per_mode[mode] = r
            print(f"[{mode}] chunk_sizes={r['chunk_sizes']}  expected_batch_size={r['expected_batch_size']}  "
                  f"clip_fraction_first_chunk={r['clip_fraction_first_chunk']}  "
                  f"delta_norm={r['delta'].norm():.6f}  nan_inf={r['nan_inf']}")

        d_legacy = per_mode["legacy"]["delta"]
        d_direct = per_mode["rejected_direct_sum"]["delta"]
        d_canon = per_mode["canonical"]["delta"]
        ebs = per_mode["canonical"]["expected_batch_size"]

        comparison = {
            "expected_batch_size": ebs,
            "canonical_vs_direct_sum": {
                "ratio_delta_norm": float(d_canon.norm() / d_direct.norm().clamp(min=1e-12)),
                "predicted_ratio_1_over_ebs": 1.0 / ebs,
                "cosine_similarity": _cos(d_canon, d_direct),
                "relative_error": _rel_error(d_canon, d_direct),
            },
            "canonical_vs_legacy": {
                "ratio_delta_norm": float(d_canon.norm() / d_legacy.norm().clamp(min=1e-12)),
                "cosine_similarity": _cos(d_canon, d_legacy),
                "relative_error": _rel_error(d_canon, d_legacy),
                "max_abs_diff": float((d_canon - d_legacy).abs().max()),
            },
            "raw_grad_norm_match_canonical_vs_direct_sum": None,
        }
        # raw per-sample grad_sample norms (first chunk): canonical should match
        # direct_sum closely (both recover the TRUE per-sample gradient)
        rn_canon = torch.tensor(per_mode["canonical"]["raw_grad_norms_first_chunk"])
        rn_direct = torch.tensor(per_mode["rejected_direct_sum"]["raw_grad_norms_first_chunk"])
        if len(rn_canon) == len(rn_direct):
            comparison["raw_grad_norm_match_canonical_vs_direct_sum"] = {
                "ratio": (rn_canon / rn_direct.clamp(min=1e-12)).tolist(),
                "relative_error": _rel_error(rn_canon, rn_direct),
            }

        print(f"\ncanonical vs rejected_direct_sum: ratio={comparison['canonical_vs_direct_sum']['ratio_delta_norm']:.6f}  "
              f"predicted(1/ebs)={comparison['canonical_vs_direct_sum']['predicted_ratio_1_over_ebs']:.6f}  "
              f"cosine={comparison['canonical_vs_direct_sum']['cosine_similarity']:.6f}")
        print(f"canonical vs legacy: ratio={comparison['canonical_vs_legacy']['ratio_delta_norm']:.6f}  "
              f"cosine={comparison['canonical_vs_legacy']['cosine_similarity']:.6f}  "
              f"relative_error={comparison['canonical_vs_legacy']['relative_error']:.4f}")

        all_results[cid] = {
            "chunk_sizes": {m: per_mode[m]["chunk_sizes"] for m in MODES},
            "expected_batch_size": {m: per_mode[m]["expected_batch_size"] for m in MODES},
            "clip_fraction_first_chunk": {m: per_mode[m]["clip_fraction_first_chunk"] for m in MODES},
            "delta_norm": {m: float(per_mode[m]["delta"].norm()) for m in MODES},
            "nan_inf": {m: per_mode[m]["nan_inf"] for m in MODES},
            "comparison": comparison,
        }

    any_nan_inf = any(all_results[c]["nan_inf"][m] for c in all_results for m in MODES)
    record = {
        "note": "VERIFICATION PROBE ONLY -- no training, no checkpoint written, no accountant "
                "persisted, dp_sgd.py's production path not modified by running this.",
        "config": {"physical_batch": args.physical_batch, "sampling_noise_seed": args.sampling_noise_seed,
                  "client_ids": client_ids, "c": 1.0, "sigma": 0.75},
        "per_client": all_results,
        "any_nan_inf": any_nan_inf,
    }
    out_json = Path("results/audit_dp_seedfix/canonical_loss_normalization_probe.json")
    if out_json.exists():
        print(f"FAIL: {out_json} already exists -- refusing to overwrite")
        return 1
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {out_json}")
    print(f"any_nan_inf={any_nan_inf}")
    return 1 if any_nan_inf else 0


if __name__ == "__main__":
    raise SystemExit(main())
