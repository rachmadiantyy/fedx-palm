#!/usr/bin/env python3
"""LOCK B2's seed-42 checkpoint as the matched Non-DP reference for the
E1/E2 XAI comparison (see scripts/44/45 for the equivalent E1/E2 lock).

Read-only: verifies the checkpoint + its own training history/results JSONs
agree with each other on the selected round (9), hard-stops on ANY
disagreement, then hashes the checkpoint and writes a small lock manifest.
No training, no evaluation, no held-out test access happens here -- that is
scripts/48's job, and it refuses to run without this script's manifest.

Checked sources (fixed paths -- this locks ONE specific, already-completed
run, not a parametrized sweep):
    runs/b2_federated/k4_seed42_leakagefree_stageA_seedfix/best_global.pt
    runs/b2_federated/k4_seed42_leakagefree_stageA_seedfix/history.json
    results/b2_k4_seed42_leakagefree_stageA_seedfix.json

Usage:
    python scripts/47_lock_b2_checkpoint.py

Output (refuses to overwrite):
    results/final_b2/b2_checkpoint_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs/b2_federated/k4_seed42_leakagefree_stageA_seedfix"
CHECKPOINT = RUN_DIR / "best_global.pt"
HISTORY_JSON = RUN_DIR / "history.json"
RESULTS_JSON = REPO_ROOT / "results/b2_k4_seed42_leakagefree_stageA_seedfix.json"
OUT_MANIFEST = REPO_ROOT / "results/final_b2/b2_checkpoint_manifest.json"

EXPECTED_SEED = 42
EXPECTED_BEST_ROUND_NUMBER = 9  # 1-indexed, matches Table 4.2 / the manuscript's own history


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


def main() -> int:
    failures: list[str] = []

    for p in (CHECKPOINT, HISTORY_JSON, RESULTS_JSON):
        if not p.exists():
            failures.append(f"required file not found: {p}")
    if failures:
        for m in failures:
            print(f"FAIL: {m}")
        return 1

    if OUT_MANIFEST.exists():
        print(f"FAIL: {OUT_MANIFEST} already exists -- refusing to overwrite. Move/rename it "
              f"first if you intend to regenerate one")
        return 1

    with open(HISTORY_JSON) as f:
        history = json.load(f)
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    # cross-check 1: recompute best round directly from history.json's own
    # per-round val.map50 -- independent of anything results.json claims
    best_idx, best_map50 = None, -1.0
    for rec in history:
        val = rec.get("val")
        if val and val.get("map50") is not None and val["map50"] > best_map50:
            best_idx, best_map50 = rec["round"], val["map50"]
    if best_idx is None:
        failures.append(f"{HISTORY_JSON}: no round has a val.map50 -- cannot determine best round")
    else:
        best_round_number = best_idx + 1
        if best_round_number != EXPECTED_BEST_ROUND_NUMBER:
            failures.append(f"{HISTORY_JSON}: best round (recomputed from history) = "
                            f"{best_round_number}, expected {EXPECTED_BEST_ROUND_NUMBER}")

    # cross-check 2: results.json's own recorded best_round must agree
    results_best_round = results.get("best_round")
    if results_best_round is None or (results_best_round + 1) != EXPECTED_BEST_ROUND_NUMBER:
        failures.append(f"{RESULTS_JSON}: best_round={results_best_round!r} (round "
                        f"{None if results_best_round is None else results_best_round + 1}) "
                        f"!= expected round {EXPECTED_BEST_ROUND_NUMBER}")
    if results.get("seed") != EXPECTED_SEED:
        failures.append(f"{RESULTS_JSON}: seed={results.get('seed')!r} != expected {EXPECTED_SEED}")

    # cross-check 3: results.json's own "checkpoint" field must point at the
    # SAME file we are about to hash and lock (not a different round/path)
    recorded_ckpt = results.get("checkpoint")
    if recorded_ckpt is None:
        failures.append(f"{RESULTS_JSON}: no 'checkpoint' field recorded")
    else:
        recorded_ckpt_path = (REPO_ROOT / recorded_ckpt).resolve()
        if recorded_ckpt_path != CHECKPOINT.resolve():
            failures.append(f"{RESULTS_JSON}: checkpoint={recorded_ckpt!r} does not resolve to "
                            f"the locked path {CHECKPOINT} -- refusing to lock a mismatched pair")

    # hard-stop: the held-out test split must NEVER have been touched by this
    # training run's own --eval-test path -- these fields must all be null
    test_fields = {k: results.get(k) for k in ("map50", "map50_95", "precision", "recall", "per_class")}
    if any(v is not None for v in test_fields.values()):
        failures.append(f"{RESULTS_JSON} already has non-null test fields {test_fields} -- this "
                        f"means --eval-test was used at TRAINING time, contradicting the "
                        f"validation-only status this lock is meant to certify. Refusing to lock")

    if failures:
        print(f"\n=== B2 CHECKPOINT LOCK: {len(failures)} FAILURE(S) -- refusing to write manifest ===")
        for m in failures:
            print(f"FAIL: {m}")
        return 1

    print("=== B2 CHECKPOINT LOCK: ALL CHECKS PASSED ===")
    ckpt_sha256 = sha256_of(CHECKPOINT)
    print(f"checkpoint={CHECKPOINT}")
    print(f"sha256={ckpt_sha256}")
    print(f"selected_round_number={EXPECTED_BEST_ROUND_NUMBER}  val_map50={results['best_val_map50']}")

    manifest = {
        "manifest_kind": "b2_seed42_checkpoint_lock",
        "note": "selected exclusively using validation before held-out test/XAI",
        "experiment": "B2", "seed": EXPECTED_SEED,
        "partition_seed": results.get("partition_seed"),
        "num_clients": results.get("num_clients"),
        "checkpoint_path": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "checkpoint_sha256": ckpt_sha256,
        "selected_round_number": EXPECTED_BEST_ROUND_NUMBER,
        "best_validation_map50": results["best_val_map50"],
        "source_history_json": str(HISTORY_JSON.relative_to(REPO_ROOT)),
        "source_results_json": str(RESULTS_JSON.relative_to(REPO_ROOT)),
        "git_commit": git_commit(), "git_dirty": git_dirty(),
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
