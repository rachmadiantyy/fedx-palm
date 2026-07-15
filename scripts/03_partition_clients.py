#!/usr/bin/env python3
"""Step 3: Dirichlet Non-IID partition of the train split across K clients,
for every K in configs/fl_config.yaml's clients.k_values, then materialize
each client's symlinked data folder + data.yaml.

--seed writes to data/splits_v2/federated_partitions_seed{S}/ instead of the
primary federated_partitions/, so multi-seed reproducibility partitions never
overwrite the primary one."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.data.partition import partition_and_write  # noqa: E402
from fedxpalm.federated.client_data import materialize_clients  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="partition only this K (default: sweep all k_values)")
    parser.add_argument("--seed", type=int, default=None,
                        help="partition seed override (default: fl_config clients.partition_seed)")
    args = parser.parse_args()

    out_root = partition_and_write("configs/dataset.yaml", "configs/fl_config.yaml",
                                   k=args.k, seed=args.seed)

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    k_values = [args.k] if args.k is not None else fl_cfg["clients"]["k_values"]
    for k in k_values:
        partition_json = out_root / f"k{k}.json"
        out_dir = out_root / f"k{k}_clients"
        materialize_clients(str(partition_json), str(splits_dir), str(out_dir), ds_cfg["nc"], ds_cfg["names"])
