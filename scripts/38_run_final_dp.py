#!/usr/bin/env python3
"""FINAL THESIS RUNNER -- E1_full / E2_partial_P2, YOLO11n, 20 continuous
rounds. This is NOT a diagnostic/pilot/probe/smoke-test/clipping-only
control: its output is the locked, reportable privacy-utility result for
the main thesis. The 5-round pilots produced by scripts/21 and /23 remain
untouched under their own results/diag_*.json paths and are never read,
overwritten, or continued by this script.

Reuses the SAME production DP-SGD mechanism every prior E1/E2 diagnostic
used (fedxpalm.privacy.dp_sgd.train_client_round_dp -- per-sample gradients,
flat clipping, Gaussian noise, PRV accountant, BatchMemoryManager, the
n_logical_steps/RDP-fallback fixes already audited 14/14) and the same
FedAvg server loop (fedxpalm.federated.server.run_federated_training).
Nothing about the DP-SGD mechanism itself is reimplemented here -- this
script only adds protocol-locking, pre/post-training audits, per-round
privacy bookkeeping, and phase summaries around that existing mechanism.

Almost nothing below is a CLI flag. --variant, --device, --training-seed,
--partition-seed, and --rounds are the only arguments; every other
hyperparameter (rounds count aside, which is still checked) is a fixed
module-level constant with NO override flag at all, so an accidental
different value cannot be passed by mistake -- it can only be changed by
editing this file, which is a new protocol decision, not a runtime option.

--partition-seed and --training-seed are DELIBERATELY separate concepts:
partition-seed identifies WHICH client split is used (locked to 42 --
the only partition ever materialized for this project); training-seed
drives local shuffle/augmentation randomness per (round, client) via
federated.client.effective_seed(), independent of which data a client
owns. Older scripts (e.g. 06_train_b2_federated.py) default
--partition-seed from --seed when --partition-seed is omitted -- this
script never does that: the two are always independent arguments here.

Resume is NOT implemented. A 20-round run's per-client Opacus accountant
state only ever exists in this process's memory (threaded through a
plain Python dict across rounds, never serialized to disk per round) --
so there is no way to reload it after an interruption without either
losing privacy-accounting fidelity or fabricating a plausible-looking but
unverified state. Better to have no resume than a resume that silently
under- or over-reports cumulative epsilon. If a run is interrupted, it
must be restarted from round 0 (models/base_groupnorm.pt); this script
refuses to overwrite an existing final output directory/JSON so a restart
can never silently clobber a completed run.

    python scripts/38_run_final_dp.py --variant E1_full --device 0 --training-seed 42 --partition-seed 42 --rounds 20
    python scripts/38_run_final_dp.py --variant E2_partial_P2 --device 0 --training-seed 42 --partition-seed 42 --rounds 20

Outputs:
    runs/final_dp/e1_full/seed{training_seed}_{rounds}r/
    runs/final_dp/e2_partial_p2/seed{training_seed}_{rounds}r/
    results/final_dp/e1_full_seed{training_seed}_{rounds}r.json
    results/final_dp/e2_partial_p2_seed{training_seed}_{rounds}r.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.server import run_federated_training  # noqa: E402
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import audit_freeze, load_state  # noqa: E402

# ============================================================
# LOCKED FINAL PROTOCOL -- fixed by explicit decision, 2026-07-26.
# NOT CLI-overridable. Changing any of these is a new protocol decision:
# edit this file (and note WHY in the commit), don't add a flag.
# ============================================================
EXPECTED_BRANCH = "final-tesis-2026"

MODEL_WEIGHTS = "models/base_groupnorm.pt"
EXPECTED_TOTAL_PARAMS = 2_591_010
EXPECTED_BATCHNORM = 0
EXPECTED_GROUPNORM = 81

OPTIMIZER = "SGD"
LR0 = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 0.0  # matches every A0/P2 diagnostic that confirmed healthy training;
                      # NOT stated explicitly in the protocol spec but kept consistent
                      # with the reference configuration rather than silently reverting
                      # to Ultralytics' own default (3.0), which would be a different,
                      # unvetted operating point

ROUNDS = 20
EPOCHS_PER_ROUND = 2
LOGICAL_BATCH = 64
PHYSICAL_BATCH = 8
IMGSZ = 960
MAX_GRAD_NORM = 1.0
SIGMA = 0.75
DELTA = 1.0e-5
ACCOUNTANT = "prv"
SECURE_MODE = False  # Opacus PrivacyEngine's own default; recorded, not set explicitly
CLIPPING_MODE = "flat"
# canonical loss-normalization generation -- see dp_sgd.py's make_private() comment
# for the full derivation of why this differs from the legacy (pre-fix) and the
# rejected direct-sum generations. This script has NEVER been run for either of
# those two prior generations, so there is no collision risk on that front, but
# the output namespace is still bumped below (final_dp_canonical/) to keep any
# eventual re-run of this exact protocol unambiguous about which generation
# produced it, matching scripts/23's convention.
PROTOCOL_STATUS = "canonical_loss_normalization"
DP_LOSS_REDUCTION = "mean"
UPSTREAM_LOSS_CONVENTION = "ultralytics_sum"
EXPLICIT_LOSS_NORMALIZATION = "actual_microbatch_mean"
WORKERS = 0
K = 4
EXPECTED_PARTITION_SEED = 42
DIRICHLET_ALPHA = 0.5

EXPECTED_SPLIT_COUNTS = {"train": 8937, "val": 826, "test": 1051}
EXPECTED_CLIENT_SIZES = {"0": 860, "1": 775, "2": 5305, "3": 1997}

assert LOGICAL_BATCH == 64 and PHYSICAL_BATCH == 8, "locked constants edited inconsistently"
assert ROUNDS == 20 and EPOCHS_PER_ROUND == 2, "locked constants edited inconsistently"

VARIANTS = {
    "E1_full": {
        "freeze_stages": [],
        "expected_trainable_params": 2_590_994,
        "expected_frozen_params": 16,
    },
    "E2_partial_P2": {
        "freeze_stages": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21],
        "expected_trainable_params": 929_522,
        "expected_frozen_params": 1_661_488,
    },
}
VARIANT_DIRS = {"E1_full": "e1_full", "E2_partial_P2": "e2_partial_p2"}

# internal round_idx ranges (0-based, half-open) -> human round numbers 1-20
PHASES = {
    "early_phase": (0, 5),          # human rounds 1-5
    "intermediate_phase": (5, 10),  # human rounds 6-10
    "extended_phase": (10, 20),     # human rounds 11-20
}


def locked_protocol_dict(variant: str) -> dict:
    """Everything a configuration_fingerprint should change on -- deliberately
    EXCLUDES training_seed (multiple seeds of the SAME locked protocol should
    share one fingerprint) and device/output paths (not part of the
    scientific configuration)."""
    v = VARIANTS[variant]
    return {
        "variant": variant, "model_weights": MODEL_WEIGHTS, "total_params": EXPECTED_TOTAL_PARAMS,
        "optimizer": OPTIMIZER, "lr0": LR0, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS, "rounds": ROUNDS, "epochs_per_round": EPOCHS_PER_ROUND,
        "logical_batch_size": LOGICAL_BATCH, "physical_batch_size": PHYSICAL_BATCH, "imgsz": IMGSZ,
        "max_grad_norm": MAX_GRAD_NORM, "sigma": SIGMA, "delta": DELTA, "accountant": ACCOUNTANT,
        "secure_mode": SECURE_MODE, "clipping_mode": CLIPPING_MODE, "workers": WORKERS, "k": K,
        "partition_seed": EXPECTED_PARTITION_SEED, "dirichlet_alpha": DIRICHLET_ALPHA,
        "freeze_stages": v["freeze_stages"],
        "dp_loss_reduction": DP_LOSS_REDUCTION,
        "upstream_loss_convention": UPSTREAM_LOSS_CONVENTION,
        "explicit_loss_normalization": EXPLICIT_LOSS_NORMALIZATION,
    }


def compute_fingerprint(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


PROTOCOL_FINGERPRINTS = {v: compute_fingerprint(locked_protocol_dict(v)) for v in VARIANTS}


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True,
        ).strip()
    except Exception as e:
        return f"unknown ({e})"


def get_git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent.parent, text=True,
        )
        return bool(out.strip())
    except Exception:
        return True  # unknown git state is treated as dirty, never silently as clean


def compute_run_fingerprint(variant: str, training_seed: int, manifest_path: Path) -> dict:
    """Unlike protocol_fingerprint (shared across seeds of the SAME locked
    protocol), this identifies THIS specific invocation: seed 42, 123, and
    2026 runs of the same variant share protocol_fingerprint but each gets
    its own run_fingerprint."""
    repo_root = Path(__file__).resolve().parent.parent
    this_file = Path(__file__).resolve()
    dp_sgd_file = repo_root / "src" / "fedxpalm" / "privacy" / "dp_sgd.py"
    dataset_manifest_file = repo_root / "configs" / "dataset.yaml"
    fields = {
        "variant": variant, "training_seed": training_seed,
        "git_commit": get_git_commit(), "git_dirty": get_git_dirty(),
        "script_38_sha256": file_sha256(str(this_file)),
        "dp_sgd_py_sha256": file_sha256(str(dp_sgd_file)),
        "dataset_manifest_sha256": file_sha256(str(dataset_manifest_file)),
        "client_partition_manifest_sha256": file_sha256(str(manifest_path)),
    }
    return {**fields, "run_fingerprint": compute_fingerprint(fields)}


def check_branch() -> None:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent, text=True,
        ).strip()
    except Exception as e:
        print(f"[!] WARNING: could not determine git branch ({e}) -- skipping branch check")
        return
    if branch != EXPECTED_BRANCH:
        print(f"FAIL: current git branch is {branch!r}, expected {EXPECTED_BRANCH!r} -- this "
              f"final-protocol runner must only be used on {EXPECTED_BRANCH!r} (e.g. "
              f"exp-yolo11s-dp is reserved for YOLO11s experiments and must not run this)")
        raise SystemExit(1)


def compute_trainable_frozen(model, freeze_stages: list[int]) -> tuple[int, int]:
    """Same freeze-name-matching logic dp_sgd.py itself uses post-filter --
    NOT read from requires_grad on a fresh, never-frozen copy of the model
    (this model hasn't been through train_client_round_dp yet)."""
    freeze_names = [f"model.{s}." for s in freeze_stages] + [".dfl"]
    trainable = frozen = 0
    for name, p in model.named_parameters():
        if any(x in name for x in freeze_names):
            frozen += p.numel()
        else:
            trainable += p.numel()
    return trainable, frozen


def expected_sample_rate_and_steps(n_samples: int, logical_batch: int, epochs_per_round: int) -> tuple[float, int]:
    """Matches Opacus's ACTUAL convention exactly (verified against installed
    opacus.data_loader.DPDataLoader.from_data_loader source): sample_rate =
    1 / len(original_data_loader), and a plain PyTorch DataLoader's __len__
    with drop_last=False is ceil(N / batch_size) -- NOT batch_size / N. Steps
    per epoch under Poisson sampling default to int(1 / sample_rate), which
    is exactly that same ceil(N / batch_size) value (its own reciprocal)."""
    import math
    steps_per_epoch = math.ceil(n_samples / logical_batch)
    sample_rate = 1.0 / steps_per_epoch
    return sample_rate, steps_per_epoch * epochs_per_round


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--training-seed", type=int, default=42,
                        help="training/randomness seed -- drives per-(round,client) shuffle "
                             "and augmentation via federated.client.effective_seed(). "
                             "Independent of --partition-seed: this does NOT change which "
                             "data a client owns, only the local training randomness")
    parser.add_argument("--partition-seed", type=int, default=42,
                        help=f"must equal {EXPECTED_PARTITION_SEED} -- the only federated "
                             f"partition materialized for this protocol. Kept as an explicit "
                             f"argument (rather than silently assumed) so a future different "
                             f"partition can never be used by accident")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help=f"must equal {ROUNDS} -- this protocol requires one continuous "
                             f"{ROUNDS}-round run, not several shorter ones")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run every pre-training audit (model/BN-GN/ModuleValidator/"
                             "trainable-frozen/dataset/partition/fingerprint/output-collision) "
                             "and print the report, then exit BEFORE any local training, "
                             "validation, FedAvg, noise addition, or accountant step")
    args = parser.parse_args()

    check_branch()

    failures = []  # pre-training hard-stop reasons, collected then reported together

    if args.rounds != ROUNDS:
        print(f"FAIL: --rounds {args.rounds} != locked value {ROUNDS}")
        return 1
    if args.partition_seed != EXPECTED_PARTITION_SEED:
        print(f"FAIL: --partition-seed {args.partition_seed} != locked value {EXPECTED_PARTITION_SEED}")
        return 1

    variant = args.variant
    vcfg = VARIANTS[variant]
    freeze_stages = vcfg["freeze_stages"]
    protocol_fingerprint = PROTOCOL_FINGERPRINTS[variant]

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    if fl_cfg["local_training"].get("optimizer") != OPTIMIZER:
        failures.append(f"configs/fl_config.yaml local_training.optimizer="
                        f"{fl_cfg['local_training'].get('optimizer')!r} != {OPTIMIZER!r}")
    if dp_cfg["dp_sgd"].get("accountant") != ACCOUNTANT:
        failures.append(f"configs/dp_config.yaml dp_sgd.accountant="
                        f"{dp_cfg['dp_sgd'].get('accountant')!r} != {ACCOUNTANT!r}")

    if not Path(MODEL_WEIGHTS).exists():
        failures.append(f"{MODEL_WEIGHTS} not found")
        for m in failures:
            print(f"FAIL: {m}")
        return 1

    # ---- structural audit on the (never-trained) base checkpoint ----
    base_sd, base_model = load_state(MODEL_WEIGHTS)
    total_params = sum(p.numel() for p in base_model.parameters())
    if total_params != EXPECTED_TOTAL_PARAMS:
        failures.append(f"total_params={total_params} != expected {EXPECTED_TOTAL_PARAMS} "
                        f"(model is not YOLO11n nc=6, or {MODEL_WEIGHTS} is not the locked checkpoint)")

    import torch.nn as nn
    n_bn = sum(1 for m in base_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in base_model.modules() if isinstance(m, nn.GroupNorm))
    if n_bn != EXPECTED_BATCHNORM:
        failures.append(f"BatchNorm={n_bn} != expected {EXPECTED_BATCHNORM}")
    if n_gn != EXPECTED_GROUPNORM:
        failures.append(f"GroupNorm={n_gn} != expected {EXPECTED_GROUPNORM}")

    from opacus.validators import ModuleValidator
    validator_errors = ModuleValidator.validate(copy.deepcopy(base_model), strict=False)
    if validator_errors:
        failures.append(f"Opacus ModuleValidator: {len(validator_errors)} error(s): "
                        f"{[str(e) for e in validator_errors]}")

    trainable_measured, frozen_measured = compute_trainable_frozen(base_model, freeze_stages)
    if trainable_measured != vcfg["expected_trainable_params"]:
        failures.append(f"{variant} trainable_params={trainable_measured} != expected "
                        f"{vcfg['expected_trainable_params']}")
    if frozen_measured != vcfg["expected_frozen_params"]:
        failures.append(f"{variant} frozen_params={frozen_measured} != expected "
                        f"{vcfg['expected_frozen_params']}")

    # ---- dataset + partition audit ----
    splits_dir = Path(ds_cfg["output_dir"])
    for split, expected_n in EXPECTED_SPLIT_COUNTS.items():
        split_dir = splits_dir / split / "images"
        if not split_dir.exists():
            failures.append(f"{split_dir} not found")
            continue
        actual_n = sum(1 for p in split_dir.iterdir() if p.is_file())
        if actual_n != expected_n:
            failures.append(f"{split} image count={actual_n} != expected {expected_n}")

    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        failures.append(f"{manifest_path} not found")
        manifest = None
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)[str(K)]
        if len(manifest["sizes"]) != K:
            failures.append(f"client count={len(manifest['sizes'])} != {K}")
        for cid, expected_n in EXPECTED_CLIENT_SIZES.items():
            actual_n = manifest["sizes"].get(cid)
            if actual_n != expected_n:
                failures.append(f"client{cid} sample count={actual_n} != expected {expected_n}")

    # ---- output-path collision check (no overwrite, no auto-resume) ----
    # namespace bumped to final_dp_canonical/ -- the existing runs/final_dp/
    # and results/final_dp/ paths hold the LEGACY (pre-canonical-fix) E1/E2
    # 20-round results and are never read or written by this generation
    variant_dir = VARIANT_DIRS[variant]
    out_dir = Path(f"runs/final_dp_canonical/{variant_dir}/seed{args.training_seed}_{ROUNDS}r")
    out_json = Path(f"results/final_dp_canonical/{variant_dir}_seed{args.training_seed}_{ROUNDS}r.json")
    if out_dir.exists() or out_json.exists():
        failures.append(f"final output already exists ({out_dir if out_dir.exists() else out_json}) "
                        f"-- resume is not supported; move/rename the existing output first if you "
                        f"intend to redo this run")

    if failures:
        print(f"\n=== PRE-TRAINING AUDIT: {len(failures)} FAILURE(S) -- refusing to train ===")
        for m in failures:
            print(f"FAIL: {m}")
        return 1

    print("=== PRE-TRAINING AUDIT: ALL CHECKS PASSED ===")
    print(f"variant={variant}  protocol_fingerprint={protocol_fingerprint}")
    print(f"total_params={total_params}  BatchNorm={n_bn}  GroupNorm={n_gn}  "
          f"trainable={trainable_measured}  frozen={frozen_measured}")
    print(f"train={EXPECTED_SPLIT_COUNTS['train']}  val={EXPECTED_SPLIT_COUNTS['val']}  "
          f"test={EXPECTED_SPLIT_COUNTS['test']} (never read below)  "
          f"clients={EXPECTED_CLIENT_SIZES}")

    clients_dir = splits_dir / "federated_partitions" / f"k{K}_clients"
    client_data_yamls = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}
    client_sample_counts = dict(manifest["sizes"])
    data_yaml = str(splits_dir / "data.yaml")

    expected_q_steps = {
        cid: expected_sample_rate_and_steps(n, LOGICAL_BATCH, EPOCHS_PER_ROUND)
        for cid, n in client_sample_counts.items()
    }
    print("expected (sample_rate q, private steps/round):",
          {cid: (round(q, 7), s) for cid, (q, s) in expected_q_steps.items()})

    run_fp = compute_run_fingerprint(variant, args.training_seed, manifest_path)
    print(f"run_fingerprint={run_fp['run_fingerprint']}  git_commit={run_fp['git_commit']}  "
          f"git_dirty={run_fp['git_dirty']}")

    if args.preflight_only:
        print("\n=== --preflight-only: stopping here. No local training, validation, FedAvg, "
              "noise addition, or accountant step has run. ===")
        return 0

    hyp = dict(
        epochs_per_round=EPOCHS_PER_ROUND, batch_size=LOGICAL_BATCH, imgsz=IMGSZ,
        optimizer=OPTIMIZER, lr0=LR0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS, workers=WORKERS, seed=args.training_seed,
    )
    dp_hyp = dict(sigma=SIGMA, max_grad_norm=MAX_GRAD_NORM, delta=DELTA, accountant=ACCOUNTANT)

    accountant_states: dict = {}

    def client_round_fn(client_id, data_yaml_c, global_weights_path, round_idx, out_dir_c):
        state_dict, info = train_client_round_dp(
            global_weights_path, data_yaml_c, hyp, dp_hyp, round_idx, client_id, out_dir_c,
            device=args.device, freeze_stages=freeze_stages,
            accountant_state=accountant_states.get(client_id),
            physical_batch_size=PHYSICAL_BATCH,
        )
        accountant_states[client_id] = info.pop("accountant_state")
        # RUNTIME optimizer audit (every round, every client -- not just once on
        # the static base checkpoint before training): confirms Opacus's
        # DPOptimizer actually held EXACTLY the intended trainable set for
        # THIS round's rebuilt trainer/optimizer, not just that the checkpoint
        # was structurally correct before training started. Mirrors the same
        # hard assertion 23_diag_b_clipping_only.py's per-layer-clipping mode
        # already makes; extended here to the flat-clipping path E1/E2 use.
        # Fails fast (raises) rather than silently continuing 20 rounds on a
        # possibly-corrupted optimizer and only noticing at the end.
        if info.get("missing_trainable_params") or info.get("unexpected_frozen_params"):
            raise RuntimeError(
                f"round {round_idx} client {client_id}: RUNTIME optimizer/trainable set "
                f"mismatch -- missing_trainable={info['missing_trainable_params'][:5]} "
                f"unexpected_frozen_in_optimizer={info['unexpected_frozen_params'][:5]}")
        # RUNTIME canonical-loss-normalization audit (every round, every client):
        # confirms dp_sgd.py actually used the canonical fix for THIS round, not
        # a regressed legacy or rejected-direct-sum generation
        if (info.get("loss_reduction") != DP_LOSS_REDUCTION
                or info.get("upstream_loss_convention") != UPSTREAM_LOSS_CONVENTION
                or info.get("explicit_loss_normalization") != EXPLICIT_LOSS_NORMALIZATION):
            raise RuntimeError(
                f"round {round_idx} client {client_id}: canonical loss-normalization mismatch -- "
                f"loss_reduction={info.get('loss_reduction')!r} (expected {DP_LOSS_REDUCTION!r}), "
                f"upstream_loss_convention={info.get('upstream_loss_convention')!r} "
                f"(expected {UPSTREAM_LOSS_CONVENTION!r}), "
                f"explicit_loss_normalization={info.get('explicit_loss_normalization')!r} "
                f"(expected {EXPLICIT_LOSS_NORMALIZATION!r})")
        return state_dict, info

    def eval_fn(weights_path):
        return evaluate_detector(weights_path, data_yaml, split="val", imgsz=IMGSZ, device=args.device)

    print(f"\n=== TRAINING: {variant}  {ROUNDS} continuous rounds  K={K}  "
          f"freeze={freeze_stages}  sigma={SIGMA}  C={MAX_GRAD_NORM}  "
          f"logical_batch={LOGICAL_BATCH}  physical_batch={PHYSICAL_BATCH}  "
          f"training_seed={args.training_seed}  partition_seed={args.partition_seed} ===")
    result = run_federated_training(
        client_round_fn, client_data_yamls, client_sample_counts,
        init_weights_path=MODEL_WEIGHTS, rounds=ROUNDS, out_dir=str(out_dir),
        eval_fn=eval_fn, eval_every=1,
    )

    # ============================================================
    # POST-TRAINING AUDITS
    # ============================================================
    history = result["history"]
    invalid_reasons = []

    if len(history) != ROUNDS:
        invalid_reasons.append(f"validation history has {len(history)} rounds, expected {ROUNDS}")

    client_ids = list(client_sample_counts)
    epsilon_per_client_per_round = []
    epsilon_max_per_round = []
    accountant_used_per_client_per_round = []
    prv_failed_per_client_per_round = []
    private_steps_per_client_per_round = []
    sample_rate_per_client = {}
    nan_inf_any = False
    fallback_reasons: dict = {}

    prev_cumulative: dict = {}
    for h in history:
        clients = h["clients"]
        eps_this = {cid: clients[cid].get("epsilon") for cid in client_ids}
        eps_vals = [v for v in eps_this.values() if v is not None]
        epsilon_per_client_per_round.append(eps_this)
        epsilon_max_per_round.append(max(eps_vals) if eps_vals else None)
        accountant_used_per_client_per_round.append({cid: clients[cid].get("accountant_used") for cid in client_ids})
        prv_failed_per_client_per_round.append({cid: clients[cid].get("prv_failed", False) for cid in client_ids})
        steps_this = {cid: clients[cid].get("cumulative_steps") for cid in client_ids}
        private_steps_per_client_per_round.append(steps_this)
        for cid in client_ids:
            info = clients[cid]
            nan_inf_any = nan_inf_any or info.get("nan_inf", False)
            if info.get("epsilon_fallback_reason"):
                fallback_reasons[cid] = info["epsilon_fallback_reason"]
            cur = steps_this[cid]
            prev = prev_cumulative.get(cid)
            if prev is not None and cur is not None and cur < prev:
                invalid_reasons.append(f"client{cid} cumulative_steps DECREASED round-over-round "
                                       f"({prev} -> {cur}) -- accountant state may have reset")
            prev_cumulative[cid] = cur
            if "accountant_step_mismatch" in info:
                invalid_reasons.append(f"client{cid} round {h['round']}: accountant_step_mismatch "
                                       f"{info['accountant_step_mismatch']}")
        if SIGMA > 0 and any(v is None for v in eps_this.values()):
            invalid_reasons.append(f"round {h['round']}: missing epsilon for at least one client "
                                   f"while sigma={SIGMA} > 0")

    for cid in client_ids:
        first_round_info = history[0]["clients"][cid]
        sample_rate_per_client[cid] = first_round_info.get("sample_rate_q")

    best_round_index = result["best_round"]
    final_round_index = ROUNDS - 1
    best_round_number = best_round_index + 1
    final_round_number = ROUNDS

    epsilon_per_client_at_best_round = epsilon_per_client_per_round[best_round_index]
    epsilon_max_at_best_round = epsilon_max_per_round[best_round_index]
    private_steps_per_client_at_best_round = private_steps_per_client_per_round[best_round_index]
    epsilon_per_client_at_final_round = epsilon_per_client_per_round[final_round_index]
    epsilon_max_at_final_round = epsilon_max_per_round[final_round_index]
    private_steps_per_client_at_final_round = private_steps_per_client_per_round[final_round_index]

    accountant_used_per_client_final = accountant_used_per_client_per_round[final_round_index]
    prv_failed_per_client_final = prv_failed_per_client_per_round[final_round_index]

    val_per_round = [
        {"round": h["round"], "map50": (h.get("val") or {}).get("map50"),
         "map50_95": (h.get("val") or {}).get("map50_95"),
         "precision": (h.get("val") or {}).get("precision"), "recall": (h.get("val") or {}).get("recall")}
        for h in history
    ]

    def phase_summary(lo: int, hi: int) -> dict:
        rows = val_per_round[lo:hi]
        map50s = [r["map50"] for r in rows if r["map50"] is not None]
        return {
            "internal_round_range": [lo, hi - 1],
            "human_round_range": [lo + 1, hi],
            "best_map50_in_phase": max(map50s) if map50s else None,
            "final_map50_in_phase": rows[-1]["map50"] if rows else None,
            "epsilon_max_at_phase_end": epsilon_max_per_round[hi - 1],
            "private_steps_per_client_at_phase_end": private_steps_per_client_per_round[hi - 1],
        }

    phases_summary = {name: phase_summary(lo, hi) for name, (lo, hi) in PHASES.items()}

    # ---- checkpoints: best (validation-selected) and final-round, named explicitly ----
    # filename uses the HUMAN-readable round number (20), not the 0-based
    # internal index (19) -- final_round_index/final_round_number in the JSON
    # keep both explicit and distinct.
    best_ckpt = out_dir / "best_validation.pt"
    final_ckpt = out_dir / f"final_round_{final_round_number}.pt"
    shutil.copy2(result["best_weights"], best_ckpt)
    shutil.copy2(result["final_weights"], final_ckpt)

    frozen_changed, trainable_changed, n_bn_final, n_gn_final, dfl_changed = audit_freeze(
        MODEL_WEIGHTS, str(final_ckpt), freeze_stages=freeze_stages)

    if frozen_changed:
        invalid_reasons.append(f"frozen region changed (freeze_stages={freeze_stages})")
    if not trainable_changed:
        invalid_reasons.append("trainable region did NOT change")
    if dfl_changed:
        invalid_reasons.append("DFL changed")
    if n_bn_final != EXPECTED_BATCHNORM:
        invalid_reasons.append(f"final BatchNorm={n_bn_final} != {EXPECTED_BATCHNORM}")
    if n_gn_final != EXPECTED_GROUPNORM:
        invalid_reasons.append(f"final GroupNorm={n_gn_final} != {EXPECTED_GROUPNORM}")
    if nan_inf_any:
        invalid_reasons.append("NaN/Inf detected in at least one client-round")
    if not best_ckpt.exists():
        invalid_reasons.append("best checkpoint missing")
    if not final_ckpt.exists():
        invalid_reasons.append("final checkpoint missing")

    run_valid = not invalid_reasons

    best_val = history[best_round_index].get("val") or {}
    final_val = history[final_round_index].get("val") or {}

    record = {
        "protocol_status": "final_primary",
        "loss_normalization_generation": PROTOCOL_STATUS,  # "canonical_loss_normalization"
        "dp_loss_reduction": DP_LOSS_REDUCTION,
        "upstream_loss_convention": UPSTREAM_LOSS_CONVENTION,
        "explicit_loss_normalization": EXPLICIT_LOSS_NORMALIZATION,
        "prior_generations_note": (
            "This script has never been run for the legacy (pre-any-loss-reduction-fix) or "
            "rejected_direct_sum_loss_reduction generations -- those exist only under "
            "runs/final_dp/ and results/final_dp/ (E1/E2 20-round, produced before this "
            "canonical fix existed) and under scripts/23's diag_b_clipping_only namespace "
            "(5-round pilots). See dp_sgd.py's make_private() comment and "
            "scripts/23_diag_b_clipping_only.py's prior_generations_note for the full history."),
        "experiment_family": "final_dp_20round",
        "privacy_operating_point": f"sigma={SIGMA}_C={MAX_GRAD_NORM}",
        "variant": variant,
        "model_weights": MODEL_WEIGHTS, "model_variant": "yolo11n",
        "total_params": int(total_params),
        "n_trainable_params": int(trainable_measured), "n_frozen_params": int(frozen_measured),
        "trainable_stage_indices": sorted(set(range(24)) - set(freeze_stages)),
        "freeze_stages": freeze_stages, "dfl_fixed": True,
        "batchnorm_count": n_bn, "groupnorm_count": n_gn,
        "optimizer": OPTIMIZER, "lr0": LR0, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "rounds": ROUNDS, "epochs_per_round": EPOCHS_PER_ROUND,
        "logical_batch_size": LOGICAL_BATCH, "physical_batch_size": PHYSICAL_BATCH,
        "imgsz": IMGSZ, "workers": WORKERS,
        "training_seed": args.training_seed, "partition_seed": args.partition_seed,
        "dataset_manifest_path": str(manifest_path),
        "client_partition_manifest_path": str(manifest_path),
        "client_sample_counts": client_sample_counts, "dirichlet_alpha": DIRICHLET_ALPHA,
        "sigma": SIGMA, "max_grad_norm": MAX_GRAD_NORM,
        "absolute_noise_std": SIGMA * MAX_GRAD_NORM, "delta": DELTA,
        "clipping_mode": CLIPPING_MODE, "secure_mode": SECURE_MODE,
        "accountant_requested": ACCOUNTANT,
        "accountant_used_per_client_per_round": accountant_used_per_client_per_round,
        "accountant_used_per_client_final": accountant_used_per_client_final,
        "prv_failed_per_client_per_round": prv_failed_per_client_per_round,
        "prv_failed_per_client_final": prv_failed_per_client_final,
        "epsilon_fallback_reason_per_client": fallback_reasons,
        "epsilon_per_client_per_round": epsilon_per_client_per_round,
        "epsilon_max_per_round": epsilon_max_per_round,
        "epsilon_per_client_at_best_round": epsilon_per_client_at_best_round,
        "epsilon_max_at_best_round": epsilon_max_at_best_round,
        "epsilon_per_client_at_final_round": epsilon_per_client_at_final_round,
        "epsilon_max_at_final_round": epsilon_max_at_final_round,
        "sample_rate_per_client": sample_rate_per_client,
        "expected_sample_rate_and_steps_per_client": {
            cid: {"expected_sample_rate_q": q, "expected_private_steps_per_round": s}
            for cid, (q, s) in expected_q_steps.items()
        },
        "private_steps_per_client_per_round": private_steps_per_client_per_round,
        "private_steps_per_client_at_best_round": private_steps_per_client_at_best_round,
        "private_steps_per_client_at_final_round": private_steps_per_client_at_final_round,
        "validation_per_round": val_per_round,
        "best_round_index": best_round_index, "best_round_number": best_round_number,
        "best_validation_map50": best_val.get("map50"), "best_validation_map50_95": best_val.get("map50_95"),
        "best_validation_precision": best_val.get("precision"), "best_validation_recall": best_val.get("recall"),
        "final_round_index": final_round_index, "final_round_number": final_round_number,
        "final_round_map50": final_val.get("map50"), "final_round_map50_95": final_val.get("map50_95"),
        "final_round_precision": final_val.get("precision"), "final_round_recall": final_val.get("recall"),
        "best_checkpoint_path": str(best_ckpt), "final_checkpoint_path": str(final_ckpt),
        "frozen_region_changed": frozen_changed, "trainable_region_changed": trainable_changed,
        "dfl_changed": dfl_changed, "nan_inf_any": nan_inf_any,
        "test_evaluated": False,
        "early_phase": phases_summary["early_phase"],
        "intermediate_phase": phases_summary["intermediate_phase"],
        "extended_phase": phases_summary["extended_phase"],
        "run_valid": run_valid, "invalid_reasons": invalid_reasons,
        "protocol_fingerprint": protocol_fingerprint,
        "run_fingerprint": run_fp["run_fingerprint"],
        "run_fingerprint_components": {k: v for k, v in run_fp.items() if k != "run_fingerprint"},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)

    # ---- console summary ----
    print("\n--- validation progression ---")
    print(f"{'round':>6}{'mAP50':>10}{'mAP50-95':>11}{'precision':>11}{'recall':>9}{'eps_max':>10}"
          f"{'steps c0':>9}{'steps c1':>9}{'steps c2':>9}{'steps c3':>9}")
    for i, h in enumerate(history):
        v = h.get("val") or {}
        def _f(x):
            return x if x is not None else float("nan")
        steps = private_steps_per_client_per_round[i]
        print(f"{h['round'] + 1:>6}{_f(v.get('map50')):>10.4f}{_f(v.get('map50_95')):>11.4f}"
              f"{_f(v.get('precision')):>11.4f}{_f(v.get('recall')):>9.4f}"
              f"{_f(epsilon_max_per_round[i]):>10.4f}"
              + "".join(f"{_f(steps.get(cid)):>9.0f}" for cid in client_ids))

    print(f"\nbest_round_number={best_round_number}  best_validation_map50={best_val.get('map50')}  "
          f"epsilon_max_at_best_round={epsilon_max_at_best_round}")
    print(f"final_round_number={final_round_number}  final_round_map50={final_val.get('map50')}  "
          f"epsilon_max_at_final_round={epsilon_max_at_final_round}")
    print(f"\nrun_valid={run_valid}")
    for r in invalid_reasons:
        print(f"INVALID: {r}")
    print(f"Saved {out_json}")
    print(f"best checkpoint: {best_ckpt}")
    print(f"final checkpoint: {final_ckpt}")
    return 0 if run_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
