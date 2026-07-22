#!/usr/bin/env python3
"""Offline (no training, no GPU) privacy-utility LADDER for the P2 architecture.

Context: the earlier epsilon<=8 cap (scripts/26) was a working target, not a
thesis requirement. This performs a wider search -- sigma in {0.25, 0.35, 0.5,
0.75, 1.0}, logical_batch in {8, 16, 32, 64} (physical_batch capped at 8 via
the already-validated BatchMemoryManager path whenever logical_batch > 8),
rounds in [5, 40] -- and, for each of five approximate target privacy regions
(epsilon_max ~= 8, 12, 20, 30, 40), finds the (sigma, batch, rounds) triple
closest to that target, via the SAME PRV accountant used everywhere else in
this project (fedxpalm.privacy.accounting.compute_epsilon).

PERFORMANCE NOTE (this is why the script is structured the way it is): a
single PRVAccountant.get_epsilon() call costs ~1-70 seconds depending on the
accumulated step count (measured directly: 68.4s for the largest client at
batch=8/40 rounds/sigma=0.25) -- the accountant.step() calls themselves are
essentially free, so the cost is driven entirely by how many get_epsilon()
calls are made and at what step counts, NOT by which client. An earlier
version of this search did a full binary search per (sigma, batch, target)
using ALL FOUR clients (~2400 get_epsilon calls) and was killed for taking
too long. This version instead:
  1. Explores using ONLY the smallest client (n=775, "client1" in this
     project's K=4 seed42 partition) as a proxy for epsilon_max. This is not
     an arbitrary shortcut: client1 has the SMALLEST n, hence (a) the FEWEST
     total accountant steps for any fixed (batch, rounds) -- so it is the
     CHEAPEST client to evaluate -- and (b) the HIGHEST Poisson sample_rate
     q = 1/ceil(n/batch) -- so it has been the binding (highest-epsilon)
     client in every single per-client table computed in this project so far
     (every projected/offline table and every real pilot's accountant
     output). Using it for exploration is simultaneously the cheapest and
     the most representative choice.
  2. Per (sigma, batch), evaluates only 2 anchor points (r=5, r=40) then
     interpolates a power-law fit (epsilon ~ A * r^p, a reasonable local
     approximation for how a fixed-(sigma,q) DP-SGD composition grows with
     step count) to estimate which round count lands closest to each target
     region -- then makes exactly ONE confirming real accountant call at
     that estimated round (not a repeated binary search).
  3. Only for the small number of configurations actually selected as
     per-region REPRESENTATIVE candidates (at most one per target region) is
     the EXACT epsilon_max recomputed using all four real clients -- this is
     the only place the expensive largest-client (n=5305) computation runs,
     bounded to <= 5 calls total instead of hundreds.

Also reports an ESTIMATED (not measured) noise/signal ratio for every grid
cell, derived from two facts already established empirically in this
project's own zero-weight-update probes:
  1. noise_norm scales EXACTLY linearly in sigma (confirmed: 482.124 @
     sigma=0.5 vs 964.247 @ sigma=1.0, batch=8 -- 482.124*2=964.248).
  2. signal_norm is IDENTICAL regardless of sigma at a fixed batch (both
     probes at batch=8 report signal=2.827 exactly), and has already been
     MEASURED at all four batch values in this grid (8, 16, 32, 64).

Output: results/dp_privacy_utility_ladder_P2.json + a printed table. This is
a PLANNING artifact only -- no training occurs, and no selection here
implies a pilot will be run.
"""
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.privacy.accounting import compute_epsilon, estimate_steps, training_sample_rate  # noqa: E402

FALLBACK_CLIENT_SIZES = {"0": 860, "1": 775, "2": 5305, "3": 1997}  # established K=4 seed42 partition
PROXY_CLIENT = "1"  # smallest n -- cheapest to evaluate AND empirically the binding (max-epsilon) client

MEASURED_SIGNAL_BY_BATCH = {8: 2.827, 16: 5.455, 32: 12.254, 64: 19.267}  # sigma=1.0, step0, measured
NOISE_BASE_SIGMA1 = 964.247  # measured noise_norm at sigma=1.0, batch=8, step0
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


def eps_for_client(clients, cid, batch, sigma, rounds, epochs_per_round, delta):
    n = clients[cid]
    q = training_sample_rate(n, batch)
    steps = estimate_steps(n, batch, epochs_per_round, rounds)
    return compute_epsilon(sigma, q, steps, delta), q, steps


def estimate_round_for_target(r_lo, eps_lo, r_hi, eps_hi, target):
    """Power-law interpolation (eps ~= A * r^p) between two anchor points,
    solved for the r giving eps(r) ~= target. Falls back to clamping at the
    boundary if target is outside [eps_lo, eps_hi]. This is an ESTIMATE used
    only to pick ONE round to confirm with a real accountant call -- the
    reported epsilon always comes from that real call, never from this
    interpolation."""
    if target <= eps_lo:
        return r_lo
    if target >= eps_hi:
        return r_hi
    p = math.log(eps_hi / eps_lo) / math.log(r_hi / r_lo)
    A = eps_lo / (r_lo ** p)
    r_est = (target / A) ** (1.0 / p)
    return int(round(min(max(r_est, r_lo), r_hi)))


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
    print(f"proxy client for exploration: client{PROXY_CLIENT} (n={clients[PROXY_CLIENT]}) -- smallest n, "
          f"cheapest to evaluate, empirically the binding client in every prior table\n")
    print(f"fixed: local_epochs={epochs_per_round}  delta={delta}  round_search_range=[{r_min},{r_max}]  "
          f"P2 architecture (epsilon is n_trainable-independent, already verified)\n")

    # --- Phase A: cheap proxy-client exploration (2 anchors + <=5 confirmations per combo) ---
    combo_results = []  # one entry per (sigma, batch), each with per-target proxy round/eps
    for sigma in sigmas:
        for batch in batches:
            eps_lo, q_lo, steps_lo = eps_for_client(clients, PROXY_CLIENT, batch, sigma, r_min, epochs_per_round, delta)
            eps_hi, q_hi, steps_hi = eps_for_client(clients, PROXY_CLIENT, batch, sigma, r_max, epochs_per_round, delta)
            print(f"[anchor] sigma={sigma} batch={batch}: eps(r={r_min})={eps_lo:.3f}  eps(r={r_max})={eps_hi:.3f}")
            per_target = {}
            for target in targets:
                if target < eps_lo * 0.5 or target > eps_hi * 2.0:
                    # wildly outside this combo's achievable range -- not worth a confirming call
                    per_target[target] = {"reachable": False, "boundary_eps": eps_lo if target < eps_lo else eps_hi,
                                          "boundary_round": r_min if target < eps_lo else r_max}
                    continue
                r_est = estimate_round_for_target(r_min, eps_lo, r_max, eps_hi, target)
                eps_confirmed, q_c, steps_c = eps_for_client(clients, PROXY_CLIENT, batch, sigma, r_est,
                                                             epochs_per_round, delta)
                per_target[target] = {"reachable": True, "rounds": r_est, "eps_proxy": eps_confirmed,
                                      "distance": abs(eps_confirmed - target)}
            combo_results.append({"sigma": sigma, "batch": batch, "eps_at_rmin": eps_lo, "eps_at_rmax": eps_hi,
                                  "per_target": per_target})

    # --- rank per-region candidates using the proxy grid ---
    expected_ratio = lambda sigma, batch: (NOISE_BASE_SIGMA1 * sigma) / MEASURED_SIGNAL_BY_BATCH[batch]  # noqa: E731
    print("\n--- per-region candidates (proxy client1 epsilon), sorted by rounds desc then est_ratio asc ---")
    region_candidates = {}
    for target in targets:
        rows = []
        for cr in combo_results:
            pt = cr["per_target"][target]
            if not pt["reachable"]:
                continue
            rows.append({"sigma": cr["sigma"], "batch": cr["batch"], "rounds": pt["rounds"],
                        "eps_proxy": pt["eps_proxy"], "distance": pt["distance"],
                        "expected_ratio": expected_ratio(cr["sigma"], cr["batch"])})
        rows = [r for r in rows if r["distance"] <= 0.35 * target]  # within +/-35% of target
        rows.sort(key=lambda r: (-r["rounds"], r["expected_ratio"]))
        region_candidates[target] = rows
        print(f"\ntarget ~= {target}:")
        for r in rows[:6]:
            print(f"  sigma={r['sigma']:<5} batch={r['batch']:<4} rounds={r['rounds']:<3} "
                  f"eps_proxy={r['eps_proxy']:.3f}  dist={r['distance']:.3f}  est_ratio={r['expected_ratio']:.2f}")
        if not rows:
            print("  (none within tolerance -- see full combo_results in the saved JSON)")

    # --- Phase B: exact epsilon_max (all 4 real clients) for the TOP candidate per region only ---
    print("\n--- Phase B: exact epsilon_max (all 4 clients) for the top candidate per region ---")
    representative = {}
    for target in targets:
        rows = region_candidates[target]
        if not rows:
            representative[target] = None
            continue
        top = rows[0]
        per_client = {}
        for cid in clients:
            eps_c, q_c, steps_c = eps_for_client(clients, cid, top["batch"], top["sigma"], top["rounds"],
                                                 epochs_per_round, delta)
            per_client[cid] = {"n": clients[cid], "q": q_c, "steps": steps_c, "epsilon": eps_c}
        eps_max_exact = max(v["epsilon"] for v in per_client.values())
        representative[target] = {**top, "eps_max_exact": eps_max_exact, "per_client": per_client,
                                  "physical_batch": min(8, top["batch"]),
                                  "sigma_directly_measured": top["sigma"] in MEASURED_SIGMAS}
        print(f"target ~= {target}: sigma={top['sigma']} batch={top['batch']} rounds={top['rounds']}  "
              f"eps_max_EXACT={eps_max_exact:.4f}  (proxy was {top['eps_proxy']:.4f})")
        for cid, v in sorted(per_client.items()):
            print(f"    client{cid} n={v['n']:5d} q={v['q']:.5f} steps={v['steps']:5d} eps={v['epsilon']:.4f}")

    out = {
        "diagnostic": "dp_privacy_utility_ladder_offline",
        "note": "OFFLINE PLANNING ONLY -- no training occurred. 'eps_proxy' fields use client1 (smallest n) "
                "only, as a cheap exploration proxy. 'eps_max_exact' fields (Phase B) are computed from all "
                "four real clients via the same PRV accountant used elsewhere in this project. "
                "expected_ratio/est_ratio is a MATHEMATICALLY JUSTIFIED ESTIMATE derived from measured "
                "sigma-linearity and measured per-batch signal norms, not a new probe measurement.",
        "architecture": "P2", "client_sizes": clients, "client_source": client_source,
        "proxy_client": PROXY_CLIENT,
        "fixed": {"local_epochs": epochs_per_round, "delta": delta, "k": 4, "c": 1.0, "lr0": 0.01,
                  "momentum": 0.9, "weight_decay": 0.0005},
        "measured_signal_by_batch": MEASURED_SIGNAL_BY_BATCH,
        "measured_noise_base_sigma1": NOISE_BASE_SIGMA1,
        "directly_measured_sigmas": sorted(MEASURED_SIGMAS),
        "targets": targets,
        "combo_results_proxy": combo_results,
        "region_candidates_proxy": {str(k): v for k, v in region_candidates.items()},
        "representative_exact": {str(k): v for k, v in representative.items()},
    }
    out_json = "results/dp_privacy_utility_ladder_P2.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
