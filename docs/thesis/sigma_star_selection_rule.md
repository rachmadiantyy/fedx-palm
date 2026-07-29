# Pre-registered rule: shared operating point sigma* for E1/E2

Status: **LOCKED before any core-sweep result exists.** This file is committed
to git precisely so the commit timestamp proves the selection rule predates
the sweep results it will be applied to. Do not edit after sweep results are
produced; if a change is ever unavoidable, record it as an explicit amendment
with its own rationale, never as a silent rewrite.

> **AMENDMENT 2 IS IN EFFECT** (see below). The original threshold (tau =
> 0.70) and Amendment 1 (tau = 0.83) are both superseded by tau = 0.85.
> All earlier texts are preserved unchanged for the amendment trail; git
> history shows every version was committed BEFORE any 40-round DP sweep
> run existed.

## Utility threshold (operational research threshold) — ORIGINAL (superseded)

    tau = 0.80 x (mean best-validation mAP@0.5 of B2, seeds 42/123/2026)
        = 0.80 x 0.8775
        ~= 0.702  ->  operational value: tau = 0.70

tau is an **operational research threshold** (~80% of the B2 multi-seed mean),
NOT a universal "acceptable/excellent" category. No universal epsilon or
utility categories are used anywhere in this thesis.

## Selection rule — ORIGINAL (superseded by Amendment 1)

sigma* = the LARGEST sigma in the candidate grid satisfying ALL of:

1. E2 best-validation mAP@0.5 >= 0.70
2. training completed normally
3. nan_inf = False
4. validation metrics finite/valid

---

# AMENDMENT 1 (superseded by Amendment 2) — threshold revised 0.70 -> 0.83

Recorded and committed BEFORE any 40-round DP sweep run was executed.
Rationale: tau = 0.70 sits too far below the research's utility goal. The
primary target is for E2 (Partial DP-SGD) to stay close to mAP@0.5 = 0.85,
given the B2 non-DP multi-seed mean of 0.8775.

## Utility targets (Amendment 1)

- **Ideal utility target for E2:** best-validation mAP@0.5 of approximately
  0.85 or higher.
- **Operational minimum threshold:**

      tau = 0.95 x (mean best-validation mAP@0.5 of B2, seeds 42/123/2026)
          = 0.95 x 0.8775
          = 0.8336  ->  operational value: tau = 0.83

tau = 0.83 is an **operational research threshold** (~95% utility retention
against the B2 multi-seed mean), NOT a universal model-quality category. Its
purpose is to keep DP utility close to the federated non-DP baseline.

## Selection rule (Amendment 1 — superseded)

sigma* = the LARGEST sigma in the candidate grid satisfying ALL of:

1. E2 best-validation mAP@0.5 >= 0.83
2. training completed normally
3. nan_inf = False
4. validation metrics finite/valid

## Pre-registered fallbacks (Amendment 1 — superseded)

- If NO sigma reaches E2 mAP@0.5 >= 0.83: pick the sigma with the highest E2
  best-validation mAP@0.5, and state explicitly that the operational utility
  threshold was not reached.
- Tie (at reporting precision): pick the LARGER sigma.

---

# AMENDMENT 2 (in effect) — threshold revised 0.83 -> 0.85

Recorded and committed BEFORE any 40-round DP core sweep was executed.

Rationale: the research target is to keep Partial DP-SGD (E2) utility close
to the federated non-private baseline B2, whose multi-seed mean
best-validation mAP@0.5 is 0.8775. mAP@0.5 >= 0.85 is therefore set as the
research's operational utility target/criterion.

Important framing (binding for the thesis text):

- tau = 0.85 is NOT a universal object-detection quality standard.
- The threshold is NO LONGER derived from the "95% retention = 0.83"
  formula. Numerically 0.85 does represent very high retention against B2,
  but the primary reason for choosing it is the research's utility target of
  keeping performance near 0.85.

## Selection rule (Amendment 2, IN EFFECT)

sigma* = the LARGEST sigma in the candidate grid satisfying ALL of:

1. E2 best-validation mAP@0.5 >= 0.85
2. training completed normally
3. nan_inf = False
4. validation metrics finite/valid

E1 is NOT part of the threshold conditions; E1 IS evaluated at the SAME
sigma* as E2 (matched privacy comparison).

## Pre-registered fallbacks (Amendment 2)

- If NO sigma reaches E2 mAP@0.5 >= 0.85: pick the sigma with the highest E2
  best-validation mAP@0.5, and state explicitly that the operational utility
  target of 0.85 was not reached.
- Tie (at reporting precision): pick the LARGER sigma.

All E1 and E2 sweep results at every sigma remain fully reported.

Notes:

- E1 is NOT part of the threshold conditions -- E1 is the full-DP comparator.
  E1 IS evaluated at the SAME sigma* as E2 (matched privacy setting: same
  sigma, same partition, same q_k, same step counts, same delta, same
  accountant => identical per-client epsilon between E1 and E2).
- All core-sweep results (every sigma, both variants) are still reported in
  full; sigma* selection complements, and does not replace, the
  privacy-utility curve analysis.

---

# General protocol (applies under every amendment)

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
