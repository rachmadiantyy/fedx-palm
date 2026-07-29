#!/usr/bin/env python3
"""PHASE C2 -- offline PRV accounting grid for the large-logical-batch P2
DP-SGD candidates (no training, no GPU).

Scope (per the agreed redesign protocol): logical batch in {64, 128, 256}
(512 REJECTED structurally before any computation: at n=860/775, batch 512
gives Poisson q = 1/ceil(n/512) = 0.5 -- each draw samples HALF the client
dataset, privacy amplification essentially vanishes -- and only 4 optimizer
steps/round, too few to train; per protocol it is not forced), sigma in
{0.5, 0.75, 1.0, 1.5, 2.0}, rounds in {5, 10, 20, 40}, local_epochs=2,
delta=1e-5, same PRV accountant and Poisson q = 1/ceil(n/batch) as
production, real client sizes from manifest.json (loud fallback to the
established constants when absent).

Also reports an ESTIMATED noise/signal ratio column:
  - batch 64: from the MEASURED sigma=1 probe signal (19.267) scaled
    linearly in sigma (exact for the Gaussian mechanism, verified);
  - batch 128/256: a PROJECTION BRACKET [linear, sqrt] extrapolated from
    the measured batch-8..64 signal trajectory (per-sample effective signal
    ~0.32-0.40 x expected batch so far, i.e. near-linear, but cancellation
    could degrade toward sqrt scaling at larger batches) -- clearly labeled;
    the Phase C1 zero-weight probes at 128/256 SUPERSEDE these projections
    the moment they are measured. No projection is ever a measurement.

Per candidate row: batch, sigma, rounds, per-client q / expected steps /
projected epsilon, epsilon_max, estimated ratio (or bracket), and
steps-per-round per client (the "enough optimization left to learn" signal
-- reported, not auto-classified; epsilon<=8 is NOT treated as mandatory).

Output: results/dp_largebatch_accounting_grid_P2.json

  python scripts/30_offline_largebatch_accounting.py

All epsilon values are PROJECTED/OFFLINE, not measured. This is a PLANNING
artifact; it launches nothing and selects nothing by itself.
"""
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.privacy.accounting import compute_epsilon, estimate_steps, training_sample_rate  # noqa: E402

FALLBACK_CLIENT_SIZES = {"0": 860, "1": 775, "2": 5305, "3": 1997}

# measured at sigma=1.0, client0, physical batch 8 (zero-weight probes, step0)
MEASURED_SIGNAL = {8: 2.827, 16: 5.455, 32: 12.254, 64: 19.267}
MEASURED_EXPECTED_BATCH = {8: 7, 16: 15, 32: 31, 64: 61}  # int(n/ceil(n/B)) for n=860
NOISE_BASE_SIGMA1 = 964.247


def load_client_sizes():
    manifest_path = Path("data/splits_v2/federated_partitions/manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            sizes = json.load(f)["4"]["sizes"]
        return {str(k): int(v) for k, v in sizes.items()}, str(manifest_path)
    print(f"[!] {manifest_path} not found -- falling back to established sizes {FALLBACK_CLIENT_SIZES}; "
          f"re-run on the machine with the real dataset to confirm.")
    return dict(FALLBACK_CLIENT_SIZES), "FALLBACK (manifest.json not found)"


def est_ratio_bracket(sigma: float, batch: int, n_probe_client: int = 860):
    """Estimated noise/signal ratio. Returns (best, worst, kind):
    measured-anchored single value for batch<=64, [linear, sqrt] bracket
    beyond."""
    noise = NOISE_BASE_SIGMA1 * sigma
    if batch in MEASURED_SIGNAL:
        return noise / MEASURED_SIGNAL[batch], noise / MEASURED_SIGNAL[batch], "measured_signal"
    eb = int(n_probe_client / math.ceil(n_probe_client / batch))
    eb64 = MEASURED_EXPECTED_BATCH[64]
    sig_linear = MEASURED_SIGNAL[64] * (eb / eb64)
    sig_sqrt = MEASURED_SIGNAL[64] * math.sqrt(eb / eb64)
    return noise / sig_linear, noise / sig_sqrt, "projected_bracket[linear,sqrt]"


def main() -> int:
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    epochs_per_round = fl_cfg["local_training"]["epochs_per_round"]
    delta = 1e-5
    batches = [64, 128, 256]
    sigmas = [0.5, 0.75, 1.0, 1.5, 2.0]
    rounds_list = [5, 10, 20, 40]

    clients, client_source = load_client_sizes()
    print(f"client sizes ({client_source}): {clients}")
    print(f"fixed: local_epochs={epochs_per_round} delta={delta} C=1 flat, P2 (929,522 trainable)\n")

    rows = []
    print(f"{'batch':>6}{'sigma':>7}{'rounds':>7}{'eps_max':>10}{'est_ratio':>22}{'steps/rnd c0/c1/c2/c3':>24}")
    for batch in batches:
        spr = {cid: math.ceil(n / batch) * epochs_per_round for cid, n in clients.items()}
        for sigma in sigmas:
            r_best, r_worst, kind = est_ratio_bracket(sigma, batch)
            for rounds in rounds_list:
                per_client = {}
                for cid, n in clients.items():
                    q = training_sample_rate(n, batch)
                    steps = estimate_steps(n, batch, epochs_per_round, rounds)
                    eps = compute_epsilon(sigma, q, steps, delta)
                    per_client[cid] = {"n": n, "q": q, "steps": steps, "epsilon": eps}
                eps_max = max(v["epsilon"] for v in per_client.values())
                ratio_str = (f"{r_best:.1f}" if kind == "measured_signal"
                             else f"[{r_best:.1f},{r_worst:.1f}]proj")
                spr_str = "/".join(str(spr[c]) for c in sorted(spr))
                print(f"{batch:>6}{sigma:>7}{rounds:>7}{eps_max:>10.3f}{ratio_str:>22}{spr_str:>24}")
                rows.append({
                    "batch": batch, "sigma": sigma, "rounds": rounds,
                    "epsilon_max_projected": eps_max, "per_client": per_client,
                    "steps_per_round_per_client": spr,
                    "est_noise_signal_ratio_best": r_best,
                    "est_noise_signal_ratio_worst": r_worst,
                    "est_ratio_kind": kind,
                })

    out = {
        "diagnostic": "dp_largebatch_accounting_grid_offline",
        "note": "OFFLINE PLANNING ONLY -- epsilon PROJECTED via the production PRV accountant, "
                "not measured. est ratios for batch 128/256 are PROJECTION BRACKETS "
                "[linear-scaling, sqrt-scaling] from the measured 8..64 trajectory and are "
                "superseded by the Phase C1 zero-weight probes; batch<=64 ratios are anchored "
                "to measured signal norms. batch 512 rejected structurally (q=0.5 at n<=860, "
                "4 steps/round).",
        "architecture": "P2_flat_C1", "client_sizes": clients, "client_source": client_source,
        "fixed": {"local_epochs": epochs_per_round, "delta": delta, "C": 1.0, "k": 4},
        "measured_signal_by_batch_sigma1": MEASURED_SIGNAL,
        "noise_base_sigma1": NOISE_BASE_SIGMA1,
        "grid": rows,
    }
    out_json = "results/dp_largebatch_accounting_grid_P2.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
