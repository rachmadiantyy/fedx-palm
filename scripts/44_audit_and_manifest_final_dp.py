#!/usr/bin/env python3
"""FINAL AUDIT + IMMUTABILITY MANIFEST for the locked E1/E2 canonical 20-round
results (results/final_dp_canonical/{e1_full,e2_partial_p2}_seed42_20r.json).

Read-only: opens the two result JSONs and the checkpoint/source files named
below, hashes them, and re-verifies every locked protocol field against the
values the thesis is committing to (Part B's checklist). NO training,
validation, FedAvg, optimizer step, or accountant step happens here, and
nothing about e1_full/e2_partial_p2's JSON or .pt files is modified.

Hard-stops (refuses to write the manifest) on ANY mismatch -- see the
`failures` list built below. This is deliberately exhaustive rather than
spot-checking a few fields, because this manifest is the artifact the rest
of Bab 4 (held-out test, per-class AP, XAI) will be built on.

Usage:
    python scripts/44_audit_and_manifest_final_dp.py

Output (refuses to overwrite):
    results/final_dp_canonical/final_artifact_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

E1_JSON = REPO_ROOT / "results/final_dp_canonical/e1_full_seed42_20r.json"
E2_JSON = REPO_ROOT / "results/final_dp_canonical/e2_partial_p2_seed42_20r.json"
E1_BEST_CKPT = REPO_ROOT / "runs/final_dp_canonical/e1_full/seed42_20r/best_validation.pt"
E2_BEST_CKPT = REPO_ROOT / "runs/final_dp_canonical/e2_partial_p2/seed42_20r/best_validation.pt"
E1_FINAL_CKPT = REPO_ROOT / "runs/final_dp_canonical/e1_full/seed42_20r/final_round_20.pt"
E2_FINAL_CKPT = REPO_ROOT / "runs/final_dp_canonical/e2_partial_p2/seed42_20r/final_round_20.pt"
DP_SGD_PY = REPO_ROOT / "src/fedxpalm/privacy/dp_sgd.py"
SCRIPT_38 = REPO_ROOT / "scripts/38_run_final_dp.py"
DATASET_YAML = REPO_ROOT / "configs/dataset.yaml"
FL_CONFIG_YAML = REPO_ROOT / "configs/fl_config.yaml"
DP_CONFIG_YAML = REPO_ROOT / "configs/dp_config.yaml"

OUT_MANIFEST = REPO_ROOT / "results/final_dp_canonical/final_artifact_manifest.json"

# ---- locked protocol values every JSON must match exactly ----
EXPECTED_COMMON = {
    "dp_loss_reduction": "mean",
    "upstream_loss_convention": "ultralytics_sum",
    "explicit_loss_normalization": "actual_microbatch_mean",
    "sigma": 0.75,
    "max_grad_norm": 1.0,
    "delta": 1.0e-5,
    "accountant_requested": "prv",
    "training_seed": 42,
    "partition_seed": 42,
    "logical_batch_size": 64,
    "physical_batch_size": 8,
    "epochs_per_round": 2,
    "optimizer": "SGD",
    "lr0": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "batchnorm_count": 0,
    "groupnorm_count": 81,
}
EXPECTED_PER_VARIANT = {
    "E1_full": {"n_trainable_params": 2_590_994, "n_frozen_params": 16,
                "trainable_stage_indices": sorted(range(24))},
    "E2_partial_P2": {"n_trainable_params": 929_522, "n_frozen_params": 1_661_488,
                      "trainable_stage_indices": [16, 19, 22, 23]},
}
EXPECTED_FINAL_ROUND_STEPS = {"0": 560, "1": 520, "2": 3320, "3": 1280}


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


def audit_one(label: str, path: Path, record: dict, failures: list[str]) -> None:
    variant = record.get("variant")
    if variant not in EXPECTED_PER_VARIANT:
        failures.append(f"{label}: unrecognized variant {variant!r}")
        return

    if record.get("run_valid") is not True:
        failures.append(f"{label}: run_valid != True (got {record.get('run_valid')!r}); "
                        f"invalid_reasons={record.get('invalid_reasons')}")

    for key, expected in EXPECTED_COMMON.items():
        got = record.get(key)
        if got != expected:
            failures.append(f"{label}: {key}={got!r} != expected {expected!r}")

    per_variant = EXPECTED_PER_VARIANT[variant]
    for key, expected in per_variant.items():
        got = record.get(key)
        if got != expected:
            failures.append(f"{label}: {key}={got!r} != expected {expected!r} for variant {variant}")

    if record.get("nan_inf_any") is not False:
        failures.append(f"{label}: nan_inf_any != False (got {record.get('nan_inf_any')!r})")
    if record.get("dfl_changed") is not False:
        failures.append(f"{label}: dfl_changed != False (got {record.get('dfl_changed')!r})")
    if record.get("frozen_region_changed") is not False:
        failures.append(f"{label}: frozen_region_changed != False "
                        f"(got {record.get('frozen_region_changed')!r})")
    if record.get("trainable_region_changed") is not True:
        failures.append(f"{label}: trainable_region_changed != True "
                        f"(got {record.get('trainable_region_changed')!r})")
    if record.get("test_evaluated") is not False:
        failures.append(f"{label}: test_evaluated != False (got {record.get('test_evaluated')!r}) -- "
                        f"held-out test must never have been touched before this audit")

    # accountant: PRV throughout, no fallback, for every client at the final round
    accountant_used = record.get("accountant_used_per_client_final") or {}
    prv_failed = record.get("prv_failed_per_client_final") or {}
    for cid, used in accountant_used.items():
        if used != "prv":
            failures.append(f"{label}: client{cid} accountant_used_per_client_final={used!r} != 'prv'")
    for cid, failed in prv_failed.items():
        if failed:
            failures.append(f"{label}: client{cid} prv_failed_per_client_final=True -- "
                            f"a fallback epsilon was used")

    # cumulative private steps at the FINAL round (round 20) must match exactly
    steps_final = record.get("private_steps_per_client_at_final_round") or {}
    for cid, expected_steps in EXPECTED_FINAL_ROUND_STEPS.items():
        got_steps = steps_final.get(cid)
        if got_steps != expected_steps:
            failures.append(f"{label}: client{cid} private_steps_per_client_at_final_round="
                            f"{got_steps!r} != expected {expected_steps}")


def main() -> int:
    for p in (E1_JSON, E2_JSON, E1_BEST_CKPT, E2_BEST_CKPT, E1_FINAL_CKPT, E2_FINAL_CKPT,
             DP_SGD_PY, SCRIPT_38, DATASET_YAML, FL_CONFIG_YAML, DP_CONFIG_YAML):
        if not p.exists():
            print(f"FAIL: required file not found: {p}")
            return 1

    if OUT_MANIFEST.exists():
        print(f"FAIL: {OUT_MANIFEST} already exists -- refusing to overwrite an existing manifest. "
              f"Move/rename it first if you intend to regenerate one")
        return 1

    with open(E1_JSON) as f:
        e1 = json.load(f)
    with open(E2_JSON) as f:
        e2 = json.load(f)

    failures: list[str] = []
    audit_one("E1_full", E1_JSON, e1, failures)
    audit_one("E2_partial_P2", E2_JSON, e2, failures)

    # cross-file check: epsilon trajectory must be IDENTICAL between E1 and E2
    # (same logical/physical steps per round regardless of which params are frozen)
    e1_eps = e1.get("epsilon_max_per_round")
    e2_eps = e2.get("epsilon_max_per_round")
    if e1_eps != e2_eps:
        failures.append(f"epsilon_max_per_round differs between E1 and E2: "
                        f"E1={e1_eps} E2={e2_eps} (expected identical)")

    if failures:
        print(f"\n=== FINAL AUDIT: {len(failures)} FAILURE(S) -- refusing to write manifest ===")
        for m in failures:
            print(f"FAIL: {m}")
        return 1

    print("=== FINAL AUDIT: ALL CHECKS PASSED ===")

    files_to_hash = {
        "e1_full_result_json": E1_JSON, "e2_partial_p2_result_json": E2_JSON,
        "e1_full_best_checkpoint": E1_BEST_CKPT, "e2_partial_p2_best_checkpoint": E2_BEST_CKPT,
        "e1_full_final_round_checkpoint": E1_FINAL_CKPT,
        "e2_partial_p2_final_round_checkpoint": E2_FINAL_CKPT,
        "dp_sgd_py": DP_SGD_PY, "script_38_run_final_dp_py": SCRIPT_38,
        "configs_dataset_yaml": DATASET_YAML, "configs_fl_config_yaml": FL_CONFIG_YAML,
        "configs_dp_config_yaml": DP_CONFIG_YAML,
    }
    sha256_by_artifact = {name: sha256_of(p) for name, p in files_to_hash.items()}
    for name, digest in sha256_by_artifact.items():
        print(f"  sha256({name}) = {digest}")

    manifest = {
        "manifest_kind": "final_dp_canonical_immutability_manifest",
        "note": "Locked, audited artifacts for the E1_full / E2_partial_P2 canonical 20-round "
                "results. Model selection used ONLY the validation split for both variants "
                "(test_evaluated=False confirmed in both source JSONs at audit time); the "
                "held-out test set had not been touched by either run.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "git_dirty": git_dirty(),
        "experiments": {
            "E1_full": {
                "result_json_path": str(E1_JSON.relative_to(REPO_ROOT)),
                "result_json_sha256": sha256_by_artifact["e1_full_result_json"],
                "selected_round_number": e1["best_round_number"],
                "selected_checkpoint_path": str(E1_BEST_CKPT.relative_to(REPO_ROOT)),
                "selected_checkpoint_sha256": sha256_by_artifact["e1_full_best_checkpoint"],
                "final_round_checkpoint_path": str(E1_FINAL_CKPT.relative_to(REPO_ROOT)),
                "final_round_checkpoint_sha256": sha256_by_artifact["e1_full_final_round_checkpoint"],
                "protocol_fingerprint": e1["protocol_fingerprint"],
                "run_fingerprint": e1["run_fingerprint"],
                "best_validation_map50": e1["best_validation_map50"],
                "epsilon_max_at_best_round": e1["epsilon_max_at_best_round"],
            },
            "E2_partial_P2": {
                "result_json_path": str(E2_JSON.relative_to(REPO_ROOT)),
                "result_json_sha256": sha256_by_artifact["e2_partial_p2_result_json"],
                "selected_round_number": e2["best_round_number"],
                "selected_checkpoint_path": str(E2_BEST_CKPT.relative_to(REPO_ROOT)),
                "selected_checkpoint_sha256": sha256_by_artifact["e2_partial_p2_best_checkpoint"],
                "final_round_checkpoint_path": str(E2_FINAL_CKPT.relative_to(REPO_ROOT)),
                "final_round_checkpoint_sha256": sha256_by_artifact["e2_partial_p2_final_round_checkpoint"],
                "protocol_fingerprint": e2["protocol_fingerprint"],
                "run_fingerprint": e2["run_fingerprint"],
                "best_validation_map50": e2["best_validation_map50"],
                "epsilon_max_at_best_round": e2["epsilon_max_at_best_round"],
            },
        },
        "canonical_loss_metadata": {
            "dp_loss_reduction": EXPECTED_COMMON["dp_loss_reduction"],
            "upstream_loss_convention": EXPECTED_COMMON["upstream_loss_convention"],
            "explicit_loss_normalization": EXPECTED_COMMON["explicit_loss_normalization"],
        },
        "source_code_and_config_sha256": {
            "dp_sgd_py": sha256_by_artifact["dp_sgd_py"],
            "script_38_run_final_dp_py": sha256_by_artifact["script_38_run_final_dp_py"],
            "configs_dataset_yaml": sha256_by_artifact["configs_dataset_yaml"],
            "configs_fl_config_yaml": sha256_by_artifact["configs_fl_config_yaml"],
            "configs_dp_config_yaml": sha256_by_artifact["configs_dp_config_yaml"],
        },
    }

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
