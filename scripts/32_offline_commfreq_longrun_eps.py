#!/usr/bin/env python3
"""STEP 2 (offline part) + STEP 4 (planning part) of the locked roadmap.

PART 1 -- Experiment G offline analysis: communication frequency.
Compares candidate A (40 rounds x 2 local epochs) vs B (80 rounds x 1 local
epoch) using the PRODUCTION step logic, not the naive rounds*epochs product:
  - q = 1/ceil(n/batch)  (Opacus DPDataLoader, same as training)
  - expected optimizer steps per local epoch = ceil(n/batch)  (the loop
    iterates the Poisson loader epochs_per_round times; empty/zero-box-batch
    skips are data-dependent runtime effects not knowable offline, so these
    are expected upper bounds, same convention as every projection so far)
  - epsilon from the validated PRV accountant on the actual (sigma, q,
    total_steps) triple.
Key structural fact this script verifies numerically: steps per local epoch
do not depend on epochs_per_round, so A and B have IDENTICAL q AND identical
expected total steps -> identical projected epsilon. The pair is exactly
privacy-matched by construction; any utility difference isolates aggregation
frequency alone.

Also prints the recommended SHORT pilot pair on the same principle:
  F-pilot (existing/next):  5 rounds x 2 epochs
  G-pilot (proposed):      10 rounds x 1 epoch
identical expected exposure (10 x steps/epoch at the same q/sigma).

PART 2 -- long E2 run privacy projection (PLANNING ONLY): projected
epsilon_max at checkpoint rounds {20, 40, 80, 120, 160, 200} for the locked
reference configuration sigma=0.75, logical batch 64, 2 local epochs,
delta=1e-5, all 4 real clients. Labeled PROJECTED/OFFLINE -- the final
thesis numbers must come from the persisted accountant of the actual run.

Output: results/offline_commfreq_longrun_eps.json

  python scripts/32_offline_commfreq_longrun_eps.py

No training. Nothing launched.
"""
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedxpalm.privacy.accounting import compute_epsilon, estimate_steps, training_sample_rate  # noqa: E402

FALLBACK_CLIENT_SIZES = {"0": 860, "1": 775, "2": 5305, "3": 1997}
SIGMA = 0.75
BATCH = 64
DELTA = 1e-5


def load_client_sizes():
    manifest_path = Path("data/splits_v2/federated_partitions/manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            sizes = json.load(f)["4"]["sizes"]
        return {str(k): int(v) for k, v in sizes.items()}, str(manifest_path)
    print(f"[!] {manifest_path} not found -- falling back to established sizes {FALLBACK_CLIENT_SIZES}; "
          f"re-run on the machine with the real dataset to confirm.")
    return dict(FALLBACK_CLIENT_SIZES), "FALLBACK (manifest.json not found)"


def main() -> int:
    clients, source = load_client_sizes()
    print(f"client sizes ({source}): {clients}")
    print(f"fixed: sigma={SIGMA}  logical_batch={BATCH}  delta={DELTA}  PRV accountant  "
          f"(P2 subset; epsilon is n_trainable-independent, verified)\n")

    # ---- PART 1: Experiment G offline (A: 40x2 vs B: 80x1) ----
    print("=== Experiment G offline: A (40 rounds x 2 epochs) vs B (80 rounds x 1 epoch) ===")
    print(f"{'client':>7}{'n':>7}{'q':>10}{'steps/ep':>10}{'A st/rnd':>10}{'B st/rnd':>10}"
          f"{'A total':>9}{'B total':>9}{'match':>7}")
    g_rows = {}
    identical = True
    for cid, n in sorted(clients.items()):
        q = training_sample_rate(n, BATCH)
        spe = math.ceil(n / BATCH)
        a_total = estimate_steps(n, BATCH, 2, 40)
        b_total = estimate_steps(n, BATCH, 1, 80)
        match = a_total == b_total
        identical = identical and match
        print(f"{cid:>7}{n:>7}{q:>10.5f}{spe:>10}{spe*2:>10}{spe:>10}{a_total:>9}{b_total:>9}{str(match):>7}")
        g_rows[cid] = {"n": n, "q": q, "steps_per_epoch": spe,
                       "A_steps_per_round": spe * 2, "B_steps_per_round": spe,
                       "A_total_steps": a_total, "B_total_steps": b_total,
                       "totals_identical": match}
    print(f"\nA and B expected totals identical for every client: {identical}")
    eps_g = {}
    for cid, n in sorted(clients.items()):
        q = training_sample_rate(n, BATCH)
        eps = compute_epsilon(SIGMA, q, g_rows[cid]["A_total_steps"], DELTA)
        eps_g[cid] = eps
        g_rows[cid]["projected_epsilon_A_and_B"] = eps
        print(f"  client{cid}: projected epsilon (A == B) = {eps:.4f}")
    print(f"  epsilon_max (A == B) = {max(eps_g.values()):.4f}")
    print("\nRecommended G short pilot (same principle): 10 rounds x 1 epoch, vs the matched "
          "5 rounds x 2 epochs reference -- identical expected exposure (10 x steps/epoch), "
          "isolating aggregation frequency alone.")
    pilot_eps = {}
    for cid, n in sorted(clients.items()):
        q = training_sample_rate(n, BATCH)
        st = estimate_steps(n, BATCH, 1, 10)
        assert st == estimate_steps(n, BATCH, 2, 5)
        pilot_eps[cid] = compute_epsilon(SIGMA, q, st, DELTA)
    print(f"  pilot projected epsilon_max (both arms) = {max(pilot_eps.values()):.4f}")

    # ---- PART 2: long-run projection ----
    checkpoints = [20, 40, 80, 120, 160, 200]
    print(f"\n=== Long E2 run projection (sigma={SIGMA}, b{BATCH}, 2 epochs/round) -- "
          f"PROJECTED/OFFLINE, not final ===")
    long_rows = []
    print(f"{'round':>7}" + "".join(f"{'c'+c:>10}" for c in sorted(clients)) + f"{'eps_max':>10}")
    for r in checkpoints:
        row = {"round": r, "per_client": {}}
        for cid, n in sorted(clients.items()):
            q = training_sample_rate(n, BATCH)
            st = estimate_steps(n, BATCH, 2, r)
            eps = compute_epsilon(SIGMA, q, st, DELTA)
            row["per_client"][cid] = {"q": q, "steps": st, "epsilon": eps}
        row["epsilon_max"] = max(v["epsilon"] for v in row["per_client"].values())
        long_rows.append(row)
        print(f"{r:>7}" + "".join(f"{row['per_client'][c]['epsilon']:>10.3f}" for c in sorted(clients))
              + f"{row['epsilon_max']:>10.3f}")

    out = {
        "note": "OFFLINE PLANNING ONLY -- all epsilon values PROJECTED via the production PRV "
                "accountant with expected step counts (upper bounds; runtime batch skips are "
                "data-dependent). Final numbers must come from the actual persisted accountant.",
        "fixed": {"sigma": SIGMA, "logical_batch": BATCH, "delta": DELTA, "k": 4},
        "client_sizes": clients, "client_source": source,
        "experiment_g_offline": {
            "candidate_A": "40 rounds x 2 local epochs", "candidate_B": "80 rounds x 1 local epoch",
            "per_client": g_rows, "totals_identical_all_clients": identical,
            "epsilon_max_A_and_B": max(eps_g.values()),
            "recommended_pilot": {"arms": ["5 rounds x 2 epochs (reference)", "10 rounds x 1 epoch"],
                                  "pilot_epsilon_max_projected": max(pilot_eps.values())},
        },
        "long_run_projection": long_rows,
    }
    Path("results").mkdir(exist_ok=True)
    out_json = "results/offline_commfreq_longrun_eps.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
