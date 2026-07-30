#!/usr/bin/env python3
"""MATCHED quantitative XAI comparison: B2 (FL Non-DP, seed42, round9) vs
E1 (Full DP canonical, seed42, round19) vs E2 (Partial DP P2 canonical,
seed42, round19), on the IDENTICAL sample set built by
scripts/49_build_xai_matched_manifest.py.

Post-hoc analysis ONLY: does not select checkpoints, does not tune
hyperparameters, does not train, does not touch the model/optimizer state.
Every checkpoint's identity is re-verified against its own already-locked
manifest (results/final_b2/b2_checkpoint_manifest.json,
results/final_dp_canonical/final_artifact_manifest.json) before use --
hard-stops on any hash mismatch.

Preserves the existing, validated Grad-CAM++/Average-Drop/Focus-Retention-
Rate formulas from src/fedxpalm/xai/{gradcam,metrics}.py UNCHANGED (only
gradcam.py's generate() gained an additive 4th return value -- the model's
own decoded box for the explained anchor -- nothing about the CAM math
itself changed). target_layer_idx=22, imgsz=960, top_fraction=0.2 for all
three models, no per-model deviation.

"Correct detection" is defined here as IoU(decoded_box, GT box) >= 0.5 at
the SAME anchor Grad-CAM++ explained (the model's own argmax-by-raw-score
anchor for the queried class) -- this is a spatial-alignment check, not a
claim that the model's full NMS/confidence-thresholded prediction pipeline
detected this object. A sample where this is False (e.g. every Empty Bunch
sample under E1/E2, matching their measured recall=0.0000) still gets a
full AD/FRR record -- it is never dropped, and the record explicitly notes
that no correct detection occurred, so the CAM must not be read as evidence
of detection success.

Per-sample-per-model failures (forward/backward exception, non-finite CAM,
wrong CAM shape) are recorded with a reason and EXCLUDED from that model's
aggregates -- they do not abort the run for the remaining samples.

SMOKE TEST (one sample, before the main loop, per checkpoint) hard-stops if:
  - target layer 22 does not exist on this model;
  - preds["scores"] is not rank-3 [1, nc, num_anchors] with nc==6;
  - activations or gradients are None/empty after one forward+backward;
  - the resulting CAM is non-finite or the wrong (imgsz, imgsz) shape;
  - the checkpoint's live sha256 does not match its locked manifest.

Usage:
    python scripts/50_evaluate_xai_matched.py --device 0

Outputs (both refuse to overwrite):
    results/xai_matched/quantitative_records_raw.json   (every per-sample-per-model record)
    results/xai_matched/quantitative_b2_e1_e2.json       (aggregates + paired deltas)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.xai.gradcam import YOLOGradCAMPlusPlus  # noqa: E402
from fedxpalm.xai.metrics import average_drop, class_confidence, focus_retention_rate, occlude_top_region  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_LAYER_IDX = 22
IMGSZ = 960
TOP_FRACTION = 0.2
IOU_CORRECT_THRESHOLD = 0.5

MANIFEST_PATH_DEFAULT = REPO_ROOT / "results/xai_matched/sample_manifest.json"
OUT_DIR_DEFAULT = REPO_ROOT / "results/xai_matched"

# fixed, already-locked identities -- this script compares exactly these
# three models, not an arbitrary/parametrized set
MODEL_SPECS = {
    "B2": {
        "lock_manifest": REPO_ROOT / "results/final_b2/b2_checkpoint_manifest.json",
        "checkpoint_key": "checkpoint_path", "sha256_key": "checkpoint_sha256",
        "nested": False,
    },
    "E1_full": {
        "lock_manifest": REPO_ROOT / "results/final_dp_canonical/final_artifact_manifest.json",
        "checkpoint_key": "selected_checkpoint_path", "sha256_key": "selected_checkpoint_sha256",
        "nested": "E1_full",
    },
    "E2_partial_P2": {
        "lock_manifest": REPO_ROOT / "results/final_dp_canonical/final_artifact_manifest.json",
        "checkpoint_key": "selected_checkpoint_path", "sha256_key": "selected_checkpoint_sha256",
        "nested": "E2_partial_P2",
    },
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception as e:
        return f"unknown ({e})"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def resolve_checkpoint(label: str, spec: dict) -> dict:
    if not spec["lock_manifest"].exists():
        raise FileNotFoundError(f"{label}: lock manifest not found: {spec['lock_manifest']}")
    with open(spec["lock_manifest"]) as f:
        m = json.load(f)
    if spec["nested"]:
        m = m["experiments"][spec["nested"]]
    ckpt_path = REPO_ROOT / m[spec["checkpoint_key"]]
    locked_sha256 = m[spec["sha256_key"]]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{label}: checkpoint not found: {ckpt_path}")
    live_sha256 = sha256_of(ckpt_path)
    if live_sha256 != locked_sha256:
        raise ValueError(f"{label}: checkpoint sha256 mismatch -- live={live_sha256} "
                         f"locked={locked_sha256}")
    return {"path": ckpt_path, "sha256": live_sha256}


def load_image_tensor(image_path: Path, imgsz: int, device: str) -> tuple[torch.Tensor, np.ndarray]:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"cv2 could not read {image_path}")
    resized_bgr = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device), resized_bgr


def xywh_to_xyxy(box_xywh: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, w, h = (float(v) for v in box_xywh)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def run_one_sample(cam_engine, detection_model, sample: dict, device: str) -> dict:
    image_path = REPO_ROOT / sample["image_path"]
    tensor, _bgr = load_image_tensor(image_path, IMGSZ, device)
    class_id = sample["class_id"]
    gt_xyxy = tuple(sample["gt_box_xyxy_imgsz"])

    cam, raw_score, anchor_idx, decoded_box_xywh = cam_engine.generate(
        tensor.clone(), class_id, output_size=(IMGSZ, IMGSZ))

    if cam.shape != (IMGSZ, IMGSZ):
        raise ValueError(f"CAM shape {cam.shape} != expected ({IMGSZ}, {IMGSZ})")
    if not np.isfinite(cam).all():
        raise ValueError("CAM contains non-finite values")

    y_c = float(torch.sigmoid(torch.tensor(raw_score)).item())
    occluded = occlude_top_region(tensor, cam, top_fraction=TOP_FRACTION)
    o_c = class_confidence(detection_model, occluded, class_id, anchor_idx)
    ad = average_drop(y_c, o_c)
    frr = focus_retention_rate(cam, gt_xyxy)

    decoded_xyxy = xywh_to_xyxy(decoded_box_xywh)
    iou = iou_xyxy(decoded_xyxy, gt_xyxy)
    correct_detection = iou >= IOU_CORRECT_THRESHOLD

    return {
        "sample_id": sample["sample_id"], "image_path": sample["image_path"],
        "box_index": sample["box_index"], "class_id": class_id, "class_name": sample["class_name"],
        "gt_box_xyxy_imgsz": sample["gt_box_xyxy_imgsz"],
        "anchor_idx": int(anchor_idx),
        "raw_score": float(raw_score),
        "confidence_before_occlusion": y_c,
        "confidence_after_occlusion": float(o_c),
        "average_drop": float(ad), "focus_retention_rate": float(frr),
        "decoded_box_xywh": [float(v) for v in decoded_box_xywh],
        "decoded_box_xyxy": list(decoded_xyxy),
        "iou_vs_gt": iou, "correct_detection": bool(correct_detection),
        "correct_detection_note": (None if correct_detection else
            "anchor is the model's own best class-score location for this GT class; "
            "IoU<0.5 vs GT box -- NO correct detection occurred here. The CAM/AD/FRR below "
            "describe the model's internal response to this class at this anchor, NOT a "
            "successful detection."),
        "cam_min": float(cam.min()), "cam_max": float(cam.max()), "cam_sum": float(cam.sum()),
        "cam_finite": True,
    }


def smoke_test(label: str, detection_model: torch.nn.Module, manifest_samples: list[dict], device: str) -> None:
    try:
        target_layer = detection_model.model[TARGET_LAYER_IDX]
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"{label}: target layer idx={TARGET_LAYER_IDX} not found: {e}")
    del target_layer

    cam_engine = YOLOGradCAMPlusPlus(detection_model, target_layer_idx=TARGET_LAYER_IDX)
    try:
        sample = manifest_samples[0]
        image_path = REPO_ROOT / sample["image_path"]
        tensor, _ = load_image_tensor(image_path, IMGSZ, device)
        detection_model.zero_grad(set_to_none=True)
        _y, preds = detection_model(tensor)
        scores = preds.get("scores")
        if scores is None or scores.dim() != 3 or scores.shape[0] != 1 or scores.shape[1] != 6:
            raise RuntimeError(f"{label}: preds['scores'] has unexpected shape "
                              f"{None if scores is None else tuple(scores.shape)}, expected [1, 6, N]")
        cam, _raw_score, _anchor_idx, _box = cam_engine.generate(
            tensor.clone(), sample["class_id"], output_size=(IMGSZ, IMGSZ))
        if cam_engine._activations is None or cam_engine._gradients is None:
            raise RuntimeError(f"{label}: activations/gradients empty after smoke forward+backward")
        if cam.shape != (IMGSZ, IMGSZ):
            raise RuntimeError(f"{label}: smoke CAM shape {cam.shape} != ({IMGSZ}, {IMGSZ})")
        if not np.isfinite(cam).all():
            raise RuntimeError(f"{label}: smoke CAM is non-finite")
    finally:
        cam_engine.remove_hooks()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH_DEFAULT))
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    raw_out = out_dir / "quantitative_records_raw.json"
    agg_out = out_dir / "quantitative_b2_e1_e2.json"
    for p in (raw_out, agg_out):
        if p.exists():
            print(f"FAIL: {p} already exists -- refusing to overwrite")
            return 1

    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found -- run scripts/49_build_xai_matched_manifest.py first")
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("imgsz") != IMGSZ:
        print(f"FAIL: manifest imgsz={manifest.get('imgsz')!r} != this script's fixed IMGSZ={IMGSZ} "
              f"-- GT boxes in the manifest were computed at a different resolution and are not "
              f"comparable to this script's decoded boxes/CAMs")
        return 1
    samples = manifest["samples"]
    manifest_sha256 = sha256_of(manifest_path)

    device = args.device if args.device in ("cpu",) else (
        f"cuda:{args.device}" if str(args.device).isdigit() else args.device)

    resolved = {}
    for label, spec in MODEL_SPECS.items():
        try:
            resolved[label] = resolve_checkpoint(label, spec)
        except (FileNotFoundError, ValueError) as e:
            print(f"FAIL: {e}")
            return 1
        print(f"{label}: checkpoint={resolved[label]['path']} sha256={resolved[label]['sha256'][:16]}...")

    from ultralytics import YOLO

    models = {}
    for label, r in resolved.items():
        yolo = YOLO(str(r["path"]))
        detection_model = yolo.model.to(device)
        models[label] = detection_model

    print("\n=== smoke test (1 sample per checkpoint) ===")
    for label, detection_model in models.items():
        smoke_test(label, detection_model, samples, device)
        print(f"{label}: smoke test PASSED")

    n_total = len(samples)
    print(f"\n=== evaluating {n_total} matched samples x {len(models)} models ===")
    all_records = {label: [] for label in models}
    failed = {label: [] for label in models}

    for i, sample in enumerate(samples):
        for label, detection_model in models.items():
            cam_engine = YOLOGradCAMPlusPlus(detection_model, target_layer_idx=TARGET_LAYER_IDX)
            try:
                record = run_one_sample(cam_engine, detection_model, sample, device)
                record["model"] = label
                record["checkpoint_path"] = str(resolved[label]["path"].relative_to(REPO_ROOT))
                record["checkpoint_sha256"] = resolved[label]["sha256"]
                all_records[label].append(record)
            except Exception as e:  # noqa: BLE001 -- deliberately broad: one bad sample must not abort the run
                failed[label].append({"sample_id": sample["sample_id"], "reason": f"{type(e).__name__}: {e}"})
            finally:
                cam_engine.remove_hooks()
        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            print(f"  {i + 1}/{n_total} samples done")

    for label in models:
        print(f"{label}: {len(all_records[label])} ok, {len(failed[label])} failed")

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_out, "w") as f:
        json.dump({"records": all_records, "failed": failed}, f, indent=2)
    print(f"Saved {raw_out}")

    # ---- aggregation (Part 5) ----
    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "mean": None, "sd": None, "median": None, "iqr": None}
        arr = np.array(values, dtype=float)
        return {
            "n": len(arr), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }

    per_model_agg = {}
    for label, recs in all_records.items():
        ad_vals = [r["average_drop"] for r in recs]
        frr_vals = [r["focus_retention_rate"] for r in recs]
        per_class = {}
        for r in recs:
            per_class.setdefault(r["class_name"], {"ad": [], "frr": []})
            per_class[r["class_name"]]["ad"].append(r["average_drop"])
            per_class[r["class_name"]]["frr"].append(r["focus_retention_rate"])
        per_model_agg[label] = {
            "global_average_drop": stats(ad_vals),
            "global_focus_retention_rate": stats(frr_vals),
            "per_class": {cname: {"average_drop": stats(v["ad"]), "focus_retention_rate": stats(v["frr"])}
                         for cname, v in per_class.items()},
            "n_ok": len(recs), "n_failed": len(failed[label]),
        }

    # matched paired deltas: join by sample_id, only where ALL relevant
    # models succeeded for that sample
    def paired_delta(label_a: str, label_b: str) -> dict:
        by_id_a = {r["sample_id"]: r for r in all_records[label_a]}
        by_id_b = {r["sample_id"]: r for r in all_records[label_b]}
        common_ids = sorted(set(by_id_a) & set(by_id_b))
        ad_deltas = [by_id_a[i]["average_drop"] - by_id_b[i]["average_drop"] for i in common_ids]
        frr_deltas = [by_id_a[i]["focus_retention_rate"] - by_id_b[i]["focus_retention_rate"] for i in common_ids]
        out = {"n_paired": len(common_ids)}
        for metric_name, deltas in (("average_drop_delta", ad_deltas), ("focus_retention_rate_delta", frr_deltas)):
            if deltas:
                arr = np.array(deltas, dtype=float)
                out[metric_name] = {
                    "mean": float(arr.mean()), "median": float(np.median(arr)),
                    "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                }
            else:
                out[metric_name] = {"mean": None, "median": None, "sd": None}
        out["note"] = "exploratory paired comparison only -- single seed/run per model, no " \
                      "population-level significance claim is made or implied"
        return out

    paired = {
        "E1_minus_B2": paired_delta("E1_full", "B2"),
        "E2_minus_B2": paired_delta("E2_partial_P2", "B2"),
        "E2_minus_E1": paired_delta("E2_partial_P2", "E1_full"),
    }

    aggregate = {
        "manifest_kind": "xai_matched_quantitative_aggregate",
        "note": "Post-hoc XAI analysis. No checkpoint selection, no tuning, no training/optimizer "
                "step occurred anywhere in this script.",
        "git_commit": git_commit(), "git_dirty": git_dirty(),
        "target_layer_idx": TARGET_LAYER_IDX, "imgsz": IMGSZ, "top_fraction": TOP_FRACTION,
        "iou_correct_detection_threshold": IOU_CORRECT_THRESHOLD,
        "sample_manifest_path": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_absolute()
            else str(manifest_path),
        "sample_manifest_sha256": manifest_sha256,
        "n_samples_in_manifest": n_total,
        "models": {label: {"checkpoint_path": str(resolved[label]["path"].relative_to(REPO_ROOT)),
                           "checkpoint_sha256": resolved[label]["sha256"]}
                  for label in models},
        "per_model": per_model_agg,
        "paired_deltas": paired,
        "failed_samples": failed,
        "total_forward_backward_calls": {label: len(all_records[label]) + len(failed[label])
                                         for label in models},
        "no_model_update": True, "no_optimizer_step": True, "no_checkpoint_selection": True,
        "no_tuning_based_on_xai": True,
    }
    with open(agg_out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved {agg_out}")

    print("\n=== per-model global summary ===")
    for label, agg in per_model_agg.items():
        ad, frr = agg["global_average_drop"], agg["global_focus_retention_rate"]
        print(f"{label:>15}: n={agg['n_ok']:4d} (failed={agg['n_failed']})  "
              f"AD mean={ad['mean']:.3f} sd={ad['sd']:.3f}  "
              f"FRR mean={frr['mean']:.3f} sd={frr['sd']:.3f}")
    print("\n=== paired deltas (exploratory) ===")
    for name, d in paired.items():
        print(f"{name}: n_paired={d['n_paired']}  "
              f"AD_delta mean={d['average_drop_delta']['mean']}  "
              f"FRR_delta mean={d['focus_retention_rate_delta']['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
