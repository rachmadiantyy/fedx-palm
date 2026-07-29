#!/usr/bin/env python3
"""Offline (no training, no GPU) DP-SGD privacy-feasibility grid search for the
P2 architecture, using the SAME PRV accountant as every other projected-epsilon
table in this project (fedxpalm.privacy.accounting.compute_epsilon).

Context: P2's zero-weight-update probes confirmed sigma=1.0 at batch=8 is
WORSE than sigma=0.5 (noise/signal ratio ~2x higher, signal itself unchanged),
so a single-hyperparameter walk (bump sigma, bump batch, ...) is no longer a
useful search strategy. This performs a constrained grid search over
(logical_batch, sigma, rounds) instead, purely via the accountant -- no
training, no GPU, no dataset read beyond the four fixed client sample counts
(from data/splits_v2/federated_partitions/manifest.json when available on
this machine; falls back to the already-established constants
{860, 775, 5305, 1997} if that file isn't present, e.g. when this script is
run somewhere without the dataset -- the fallback values are printed loudly
so they're never silently assumed).

Search space:
  logical_batch: 8, 16, 32, 64, 128   (physical_batch capped at 8 for GPU
                                        memory in the real probe/training
                                        scripts via Opacus BatchMemoryManager
                                        -- irrelevant to the accountant itself,
                                        which only sees the LOGICAL batch)
  sigma:         0.5, 0.75, 1.0, 1.5
  rounds:        5..40 (fixed: local_epochs=2, delta=1e-5, K=4, same client
                  sizes, same Opacus sample-rate/expected-step formula this
                  project has used throughout)

For each (batch, sigma) pair, epsilon_max is monotonically increasing in
rounds (more steps -> more privacy loss), so instead of evaluating all 36
round values we binary-search the LARGEST round count in [5, 40] with
epsilon_max <= 8 (the constraint), which is also the most useful candidate
per pair for learning (more communication rounds at the same sigma/batch is
strictly more training signal, never a downside once the epsilon constraint
is satisfied).

Output: results/dp_feasibility_grid_P2.json (full grid, every batch x sigma
pair, feasible or not) + a printed ranked table of feasible candidates.

This is a DIAGNOSTIC/PLANNING artifact: it proposes candidate configurations
to zero-weight-probe next. It does not itself justify running a training
pilot -- that decision is made from the probes' MEASURED noise/signal ratios,
per the pre-agreed decision rule.
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.privacy.accounting import compute_epsilon, estimate_steps, training_sample_rate  # noqa: E402

FALLBACK_CLIENT_SIZES = {"0": 860, "1": 775, "2": 5305, "3": 1997}  # established K=4 seed42 partition


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


def max_feasible_rounds(clients, batch, sigma, epochs_per_round, delta, eps_cap, r_min=5, r_max=40):
    eps_at_min, _ = epsilon_max_at(clients, batch, sigma, r_min, epochs_per_round, delta)
    if eps_at_min > eps_cap:
        return None  # infeasible even at the minimum required round count
    eps_at_max, _ = epsilon_max_at(clients, batch, sigma, r_max, epochs_per_round, delta)
    if eps_at_max <= eps_cap:
        return r_max  # feasible all the way to the round-count ceiling
    lo, hi = r_min, r_max  # invariant: lo feasible, hi infeasible
    while hi - lo > 1:
        mid = (lo + hi) // 2
        eps_mid, _ = epsilon_max_at(clients, batch, sigma, mid, epochs_per_round, delta)
        if eps_mid <= eps_cap:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> int:
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    epochs_per_round = fl_cfg["local_training"]["epochs_per_round"]  # fixed at 2, per instruction
    delta = 1e-5
    eps_cap = 8.0
    r_min, r_max = 5, 40
    batches = [8, 16, 32, 64, 128]
    sigmas = [0.5, 0.75, 1.0, 1.5]

    clients, client_source = load_client_sizes()
    print(f"client sizes ({client_source}): {clients}")
    print(f"fixed: local_epochs={epochs_per_round}  delta={delta}  eps_cap={eps_cap}  "
          f"round_search_range=[{r_min},{r_max}]  P2 architecture (epsilon is n_trainable-independent, "
          f"already verified)\n")

    grid = []
    for batch in batches:
        for sigma in sigmas:
            r_feas = max_feasible_rounds(clients, batch, sigma, epochs_per_round, delta, eps_cap, r_min, r_max)
            if r_feas is None:
                grid.append({"batch": batch, "sigma": sigma, "feasible": False, "rounds": None,
                             "epsilon_max": None, "per_client": None})
                continue
            eps_max, per_client = epsilon_max_at(clients, batch, sigma, r_feas, epochs_per_round, delta)
            grid.append({"batch": batch, "sigma": sigma, "feasible": True, "rounds": r_feas,
                         "epsilon_max": eps_max, "per_client": per_client})

    print(f"{'batch':>6}{'sigma':>7}{'feasible':>10}{'max_rounds':>12}{'epsilon_max':>13}")
    for row in grid:
        if row["feasible"]:
            print(f"{row['batch']:>6}{row['sigma']:>7}{'yes':>10}{row['rounds']:>12}{row['epsilon_max']:>13.4f}")
        else:
            print(f"{row['batch']:>6}{row['sigma']:>7}{'no':>10}{'--':>12}{f'>{eps_cap} @ r={r_min}':>13}")

    feasible = [r for r in grid if r["feasible"] and r["rounds"] >= r_min]
    # rank: most communication rounds first (learning signal), then SNR
    # favorability (larger logical batch, then lower sigma) as the tie-break --
    # per the agreed decision rule, NOT a claim about measured utility
    feasible_ranked = sorted(feasible, key=lambda r: (-r["rounds"], -r["batch"], r["sigma"]))

    print(f"\n{len(feasible_ranked)}/{len(grid)} grid cells satisfy epsilon_max<=8 AND rounds>=5.")
    print("\n--- ranked feasible candidates (rounds desc, then batch desc, then sigma asc) ---")
    print(f"{'rank':>5}{'batch':>7}{'sigma':>7}{'rounds':>8}{'eps_max':>10}   per-client (q / steps / eps)")
    for i, r in enumerate(feasible_ranked, 1):
        pc_str = "; ".join(f"c{cid}: q={v['q']:.5f} steps={v['steps']} eps={v['epsilon']:.3f}"
                           for cid, v in sorted(r["per_client"].items()))
        print(f"{i:>5}{r['batch']:>7}{r['sigma']:>7}{r['rounds']:>8}{r['epsilon_max']:>10.4f}   {pc_str}")

    top_n = feasible_ranked[:5]
    print(f"\n--- top {len(top_n)} candidates: zero-weight P2 probe commands (physical_batch capped at 8 "
          f"for GPU memory via BatchMemoryManager, matching every prior logical-batch probe) ---")
    for i, r in enumerate(top_n, 1):
        physical = min(8, r["batch"])
        print(f"\n#{i}: batch={r['batch']} sigma={r['sigma']} rounds={r['rounds']} (projected eps_max={r['epsilon_max']:.3f})")
        print("python scripts/25_diag_noise_signal_probe.py --device 0 --client 0 "
              "--freeze-stages 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 20 21 --subset-label P2 "
              f"--sigma {r['sigma']} --logical-batch {r['batch']} --physical-batch {physical} --lr0 0.01 --steps 2")

    out = {
        "diagnostic": "dp_feasibility_grid_offline",
        "note": "OFFLINE PLANNING ONLY -- no training occurred; epsilon values are PROJECTED via the same "
                "PRV accountant used elsewhere in this project, not measured from an actual run",
        "architecture": "P2", "client_sizes": clients, "client_source": client_source,
        "fixed": {"local_epochs": epochs_per_round, "delta": delta, "k": 4},
        "constraints": {"epsilon_max_cap": eps_cap, "min_rounds": r_min, "round_search_ceiling": r_max},
        "grid": grid,
        "ranked_feasible_candidates": feasible_ranked,
    }
    out_json = "results/dp_feasibility_grid_P2.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
