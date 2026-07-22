#!/usr/bin/env python3
"""Offline (no training, no GPU) privacy-utility LADDER for the P2 architecture.

Context: the earlier epsilon<=8 cap (scripts/26) was a working target, not a
thesis requirement. This performs a wider search -- sigma in {0.25, 0.35, 0.5,
0.75, 1.0}, logical_batch in {8, 16, 32, 64} (physical_batch capped at 8 via
the already-validated BatchMemoryManager path whenever logical_batch > 8),
rounds in [5, 40] -- and, for each of five approximate target privacy regions
(epsilon_max ~= 8, 12, 20, 30, 40), finds the (sigma, batch, rounds) triple
whose epsilon_max is closest to that target, using the SAME PRV accountant as
every other projected-epsilon table in this project
(fedxpalm.privacy.accounting.compute_epsilon). Epsilon is monotonically
increasing in rounds for a fixed (sigma, batch), so a binary search over
rounds finds the closest match without scanning all 36 round values.

Also reports an ESTIMATED (not measured) noise/signal ratio for every grid
cell, derived from two facts already established empirically in this
project's own zero-weight-update probes:
  1. noise_norm scales EXACTLY linearly in sigma (not just approximately --
     this is a direct mathematical consequence of the Gaussian mechanism,
     std = sigma * C, and was independently confirmed by the P2 probes:
     noise=482.124 @ sigma=0.5 vs noise=964.247 @ sigma=1.0, batch=8 --
     482.124 * 2 = 964.248, matching to 4 significant figures).
  2. signal_norm (the clipped-gradient-sum norm BEFORE noise) is measured to
     be IDENTICAL regardless of sigma at a fixed batch (both sigma=0.5 and
     sigma=1.0 probes at batch=8 report signal=2.827 exactly) -- expected,
     since clipping happens before noise is added and does not depend on the
     noise mechanism. So signal_norm depends only on batch, and this project
     has ALREADY MEASURED it at all four batch values in this grid (8, 16,
     32, 64), via the sigma=1.0 logical-batch probes.

Given these two facts, expected_ratio(sigma, batch) = (964.247 * sigma) /
signal_measured(batch) is a MATHEMATICALLY JUSTIFIED ESTIMATE for any sigma
in the search grid at any of the four measured batch sizes -- not a new
measurement. It is reported as such, never conflated with a probed value.
Sigma values already directly probed (0.5, 1.0) are flagged so their
estimate is known to match a real measurement; sigma values only estimated
(0.25, 0.35, 0.75) are flagged as candidates for an optional confirmatory
zero-weight probe if selected as a representative operating point.

Output: results/dp_privacy_utility_ladder_P2.json (full grid + per-region
selection) and a printed table.

This is a PLANNING artifact only. No training occurs. No selection here
implies a pilot will be run -- that remains a separate, explicit decision.
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.privacy.accounting import compute_epsilon, estimate_steps, training_sample_rate  # noqa: E402

FALLBACK_CLIENT_SIZES = {"0": 860, "1": 775, "2": 5305, "3": 1997}  # established K=4 seed42 partition

# measured (not estimated) signal_norm at sigma=1.0, step0, per the P2 logical-batch probes
MEASURED_SIGNAL_BY_BATCH = {8: 2.827, 16: 5.455, 32: 12.254, 64: 19.267}
# measured (not estimated) noise_norm at sigma=1.0, batch=8, step0 -- the reference point for
# the exact linear-in-sigma scaling (noise_norm = NOISE_BASE_SIGMA1 * sigma)
NOISE_BASE_SIGMA1 = 964.247
MEASURED_SIGMAS = {0.5, 1.0}  # sigmas with a DIRECT zero-weight-update measurement on record


def load_client_sizes():
    manifest_path = Path("data/splits_v2/federated_partitions/manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            sizes = json.load(f)["4"]["sizes"]
        return {str(k): int(v) for k, v in sizes.items()}, str(manifest_path)
    print(f"[!] {manifest_path} not found on this machine -- falling back to the already-established "
          f"K=4 seed42 client sizes {FALLBACK_CLIENT_SIZES}. Re-run this script on the machine with the "
          f"real dataset to confirm these match manifest.json exactly before trusting the table.")
    return dict(FALLBACK_CLIENT_SIZES), "FALLBACK (manifest.json not found)"


def epsilon_max_at(clients, batch, sigma, rounds, epochs_per_round, delta):
    per_client = {}
    for cid, n in clients.items():
        q = training_sample_rate(n, batch)
        steps = estimate_steps(n, batch, epochs_per_round, rounds)
        eps = compute_epsilon(sigma, q, steps, delta)
        per_client[cid] = {"n": n, "q": q, "steps": steps, "epsilon": eps}
    eps_max = max(v["epsilon"] for v in per_client.values())
    return eps_max, per_client


def closest_round_to_target(clients, batch, sigma, target_eps, epochs_per_round, delta, r_min=5, r_max=40):
    """Binary-search the round count in [r_min, r_max] whose epsilon_max is
    closest to target_eps (epsilon_max is monotonically increasing in rounds
    for a fixed (batch, sigma), so this is well-defined). Returns
    (rounds, eps_max, per_client, distance_to_target)."""
    eps_lo, pc_lo = epsilon_max_at(clients, batch, sigma, r_min, epochs_per_round, delta)
    eps_hi, pc_hi = epsilon_max_at(clients, batch, sigma, r_max, epochs_per_round, delta)
    if target_eps <= eps_lo:
        return r_min, eps_lo, pc_lo, abs(eps_lo - target_eps)
    if target_eps >= eps_hi:
        return r_max, eps_hi, pc_hi, abs(eps_hi - target_eps)
    lo, hi = r_min, r_max  # invariant: eps(lo) <= target_eps <= eps(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        eps_mid, _ = epsilon_max_at(clients, batch, sigma, mid, epochs_per_round, delta)
        if eps_mid <= target_eps:
            lo = mid
        else:
            hi = mid
    eps_lo2, pc_lo2 = epsilon_max_at(clients, batch, sigma, lo, epochs_per_round, delta)
    eps_hi2, pc_hi2 = epsilon_max_at(clients, batch, sigma, hi, epochs_per_round, delta)
    if abs(eps_lo2 - target_eps) <= abs(eps_hi2 - target_eps):
        return lo, eps_lo2, pc_lo2, abs(eps_lo2 - target_eps)
    return hi, eps_hi2, pc_hi2, abs(eps_hi2 - target_eps)


def main() -> int:
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    epochs_per_round = fl_cfg["local_training"]["epochs_per_round"]  # fixed at 2
    delta = 1e-5
    r_min, r_max = 5, 40
    sigmas = [0.25, 0.35, 0.5, 0.75, 1.0]
    batches = [8, 16, 32, 64]
    targets = [8, 12, 20, 30, 40]

    clients, client_source = load_client_sizes()
    print(f"client sizes ({client_source}): {clients}")
    print(f"fixed: local_epochs={epochs_per_round}  delta={delta}  round_search_range=[{r_min},{r_max}]  "
          f"P2 architecture (epsilon is n_trainable-independent, already verified)\n")

    # full grid: for every (sigma, batch), the closest-achievable round count per target region
    grid = []
    for sigma in sigmas:
        for batch in batches:
            physical = min(8, batch)
            expected_ratio = (NOISE_BASE_SIGMA1 * sigma) / MEASURED_SIGNAL_BY_BATCH[batch]
            sigma_is_measured = sigma in MEASURED_SIGMAS
            for target in targets:
                rounds, eps_max, per_client, dist = closest_round_to_target(
                    clients, batch, sigma, target, epochs_per_round, delta, r_min, r_max)
                grid.append({
                    "target_epsilon_region": target, "sigma": sigma, "logical_batch": batch,
                    "physical_batch": physical, "rounds": rounds, "epsilon_max_projected": eps_max,
                    "distance_to_target": dist, "per_client": per_client,
                    "expected_relative_noise_scale": expected_ratio,
                    "sigma_directly_measured": sigma_is_measured,
                })

    print(f"{'target':>7}{'sigma':>7}{'lbatch':>8}{'pbatch':>8}{'rounds':>8}{'eps_proj':>10}{'|dist|':>8}"
          f"{'est_ratio':>11}{'sigma_meas':>11}")
    for row in grid:
        print(f"{row['target_epsilon_region']:>7}{row['sigma']:>7}{row['logical_batch']:>8}"
              f"{row['physical_batch']:>8}{row['rounds']:>8}{row['epsilon_max_projected']:>10.3f}"
              f"{row['distance_to_target']:>8.3f}{row['expected_relative_noise_scale']:>11.2f}"
              f"{str(row['sigma_directly_measured']):>11}")

    # per-region candidate ranking: closer to target first, then MORE rounds
    # (learning signal) weighted against BETTER (lower) expected noise scale --
    # not a single automatic winner, all candidates within a tolerance band
    # are surfaced so the actual selection reasoning (not just the argmin) is
    # visible and can be argued in the report
    print("\n--- per-region candidates within +/-25% of target, sorted by rounds desc then est_ratio asc ---")
    per_region_candidates = {}
    for target in targets:
        rows = [r for r in grid if r["target_epsilon_region"] == target
                and abs(r["epsilon_max_projected"] - target) <= 0.25 * target]
        rows.sort(key=lambda r: (-r["rounds"], r["expected_relative_noise_scale"]))
        per_region_candidates[target] = rows
        print(f"\ntarget ~= {target}:")
        for r in rows:
            print(f"  sigma={r['sigma']:<5} batch={r['logical_batch']:<4} rounds={r['rounds']:<3} "
                  f"eps_proj={r['epsilon_max_projected']:.3f}  est_ratio={r['expected_relative_noise_scale']:.2f}  "
                  f"sigma_measured={r['sigma_directly_measured']}")

    out = {
        "diagnostic": "dp_privacy_utility_ladder_offline",
        "note": "OFFLINE PLANNING ONLY -- no training occurred; epsilon values are PROJECTED via the same "
                "PRV accountant used elsewhere in this project, not measured from an actual run. "
                "expected_relative_noise_scale is a MATHEMATICALLY JUSTIFIED ESTIMATE derived from measured "
                "sigma-linearity and measured per-batch signal norms, not a new probe measurement.",
        "architecture": "P2", "client_sizes": clients, "client_source": client_source,
        "fixed": {"local_epochs": epochs_per_round, "delta": delta, "k": 4, "c": 1.0, "lr0": 0.01,
                  "momentum": 0.9, "weight_decay": 0.0005},
        "measured_signal_by_batch": MEASURED_SIGNAL_BY_BATCH,
        "measured_noise_base_sigma1": NOISE_BASE_SIGMA1,
        "directly_measured_sigmas": sorted(MEASURED_SIGMAS),
        "targets": targets,
        "grid": grid,
        "per_region_candidates_within_25pct": {str(k): v for k, v in per_region_candidates.items()},
    }
    out_json = "results/dp_privacy_utility_ladder_P2.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
