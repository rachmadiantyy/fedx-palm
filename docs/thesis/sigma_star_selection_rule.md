# Pre-registered rule: shared operating point sigma* for E1/E2

Status: **LOCKED before any core-sweep result exists.** This file is committed
to git precisely so the commit timestamp proves the selection rule predates
the sweep results it will be applied to. Do not edit after sweep results are
produced; if a change is ever unavoidable, record it as an explicit amendment
with its own rationale, never as a silent rewrite.

## Utility threshold (operational research threshold)

    tau = 0.80 x (mean best-validation mAP@0.5 of B2, seeds 42/123/2026)
        = 0.80 x 0.8775
        ~= 0.702  ->  operational value: tau = 0.70

tau is an **operational research threshold** (~80% of the B2 multi-seed mean),
NOT a universal "acceptable/excellent" category. No universal epsilon or
utility categories are used anywhere in this thesis.

## Selection rule

sigma* = the LARGEST sigma in the candidate grid satisfying ALL of:

1. E2 best-validation mAP@0.5 >= 0.70
2. training completed normally
3. nan_inf = False
4. validation metrics finite/valid

Notes:

- E1 is NOT part of the threshold conditions -- E1 is the full-DP comparator.
  E1 IS evaluated at the SAME sigma* as E2 (matched privacy setting: same
  sigma, same partition, same q_k, same step counts, same delta, same
  accountant => identical per-client epsilon between E1 and E2).
- All core-sweep results (every sigma, both variants) are still reported in
  full; sigma* selection complements, and does not replace, the
  privacy-utility curve analysis.

## Pre-registered fallbacks

- If NO sigma reaches E2 mAP@0.5 >= 0.70: pick the sigma with the highest E2
  best-validation mAP@0.5, and state explicitly that no configuration reached
  the utility threshold.
- Tie (at reporting precision): pick the LARGER sigma.

## Multi-seed confirmation (after sigma* is fixed)

E1 at sigma* and E2 at sigma*, seeds 42 / 123 / 2026, partition seed fixed at
42, validation-only. The held-out test is evaluated once, after every
configuration is locked, on explicit instruction -- never during selection.

## Clipping-only diagnostic (sigma = 0)

"DP mechanism ablation: clipping-only diagnostic". Not a DP scenario: no
finite (epsilon, delta) guarantee, epsilon reported as null
(privacy_guarantee = not_applicable_no_noise), excluded from the
privacy-utility curve. Utility metrics and clipping diagnostics may be
reported. Used only to decompose utility cost: B1->B2 (federation),
B2->clipping-only (clipping), clipping-only->E1(sigma) (noise).
