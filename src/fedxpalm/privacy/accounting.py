"""Standalone privacy-budget (epsilon) computation, independent of a training run.

Useful to pre-compute the sigma -> epsilon table (mirrors thesis Table 4.4)
before committing GPU time to a full sweep, or to sanity-check the epsilon
that `dp_sgd.train_client_round_dp` reports for a given (K, sigma, rounds).
"""
from __future__ import annotations

from opacus.accountants import PRVAccountant


def compute_epsilon(
    sigma: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
    accountant: str = "prv",
) -> float:
    """epsilon for `steps` Poisson-subsampled Gaussian-mechanism DP-SGD steps."""
    if accountant != "prv":
        raise NotImplementedError("only the PRV accountant is wired up here; "
                                   "pass accountant='prv' or extend this function")
    acct = PRVAccountant()
    for _ in range(steps):
        acct.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return acct.get_epsilon(delta=delta)


def estimate_steps(n_samples: int, batch_size: int, epochs_per_round: int, rounds: int) -> int:
    """Total local SGD steps a client takes over the whole federated run."""
    steps_per_epoch = max(1, round(n_samples / batch_size))
    return steps_per_epoch * epochs_per_round * rounds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-compute an epsilon table for a K x sigma grid.")
    parser.add_argument("--n-samples", type=int, required=True, help="per-client sample count")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    args = parser.parse_args()

    steps = estimate_steps(args.n_samples, args.batch_size, args.epochs_per_round, args.rounds)
    sample_rate = args.batch_size / args.n_samples
    print(f"n_samples={args.n_samples} batch_size={args.batch_size} -> sample_rate={sample_rate:.4f}, "
          f"{steps} total local steps over {args.rounds} rounds")
    for sigma in args.sigmas:
        eps = compute_epsilon(sigma, sample_rate, steps, args.delta)
        print(f"  sigma={sigma:>4} -> epsilon={eps:.3f} (delta={args.delta})")
