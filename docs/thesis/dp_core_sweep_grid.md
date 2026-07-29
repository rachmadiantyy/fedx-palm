# Pre-registered DP core sweep grid: sigma in {0.5, 1.0, 3.0}

Status: **LOCKED before any 40-round DP core sweep result exists.** Committed
so the git timestamp proves the grid and its rationale predate every sweep
result. The sigma* selection rule (docs/thesis/sigma_star_selection_rule.md,
commit d398674, Amendment 2 in effect) is unchanged by this document.

## Final grid

| configuration      | sigma | expected eps_max (offline PRV) |
|--------------------|-------|--------------------------------|
| higher-epsilon     | 0.5   | ~38.5                          |
| intermediate-eps   | 1.0   | ~5.6                           |
| lower-epsilon      | 3.0   | ~1.2                           |

Rationale (recorded verbatim from the decision):

- sigma 0.5: higher-epsilon configuration -- the lowest-noise anchor, needed
  to test whether E2's ~0.85 utility target can be sustained at all.
- sigma 1.0: intermediate-epsilon configuration.
- sigma 3.0: lower-epsilon configuration representing a much stricter
  privacy setting.
- The grid spans a wide privacy-budget range, eps_max ~38.5 -> ~5.6 -> ~1.2
  (roughly uniform on a log scale).
- Final epsilons are ALWAYS taken from each run's actual persistent
  per-client PRV accountant history, never from the offline estimate alone.

## Offline PRV table backing the choice (training-matched semantics)

Computed with the repo's accounting helper after it was reconciled EXACTLY
against the real 2-round smoke history (commit f265114): PRV accountant,
q = 1/ceil(n/batch) (Opacus DPDataLoader semantics), expected steps =
ceil(n/batch) x epochs x rounds (conservative upper bound; actual is
slightly lower due to skipped empty/zero-box Poisson draws).

batch=8, local_epochs=2, rounds=40, delta=1e-5, clients C0=860, C1=775,
C2=5305, C3=1997 (partition seed 42):

| sigma | eps_C0 | eps_C1 | eps_C2 | eps_C3 | eps_max |
|-------|--------|--------|--------|--------|---------|
| 0.50  | 36.495 | 38.479 | 14.443 | 23.925 | 38.479  |
| 0.75  |  9.985 | 10.647 |  3.404 |  6.053 | 10.647  |
| 1.00  |  5.220 |  5.563 |  1.834 |  3.196 |  5.563  |
| 1.50  |  2.702 |  2.872 |  0.983 |  1.684 |  2.872  |
| 2.00  |  1.841 |  1.955 |  0.679 |  1.157 |  1.955  |
| 3.00  |  1.129 |  1.198 |  0.423 |  0.715 |  1.198  |

eps_max is always C1 (smallest client, largest sample rate). Sigmas 0.75,
1.5, 2.0 are reserved interpolation points, to be run only if later analysis
requires finer resolution (separate decision).

## Core sweep protocol (6 runs)

E1 Full DP and E2 Partial DP, each at sigma in {0.5, 1.0, 3.0}:
training seed 42, partition seed 42, K=4, alpha=0.5, 40 communication
rounds, 2 local epochs, batch=8, imgsz=960, workers=0, validation-only,
delta=1e-5, persistent per-client PRV accountant. Sigma is passed explicitly
per run (configs/dp_config.yaml's sigma list is not consulted by these
commands). The held-out test remains locked; sigma* is then chosen by the
Amendment-2 rule, and multi-seed confirmation follows at sigma* only.
