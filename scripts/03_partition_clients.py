#!/usr/bin/env python3
"""Step 3: Dirichlet Non-IID partition of the train split across K clients,
for every K in configs/fl_config.yaml's clients.k_values, then materialize
each client's symlinked data folder + data.yaml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.data.partition import partition_and_write  # noqa: E402
from fedxpalm.federated.client_data import materialize_clients  # noqa: E402

if __name__ == "__main__":
    partition_and_write("configs/dataset.yaml", "configs/fl_config.yaml")

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    for k in fl_cfg["clients"]["k_values"]:
        partition_json = splits_dir / "federated_partitions" / f"k{k}.json"
        out_dir = splits_dir / "federated_partitions" / f"k{k}_clients"
        materialize_clients(str(partition_json), str(splits_dir), str(out_dir), ds_cfg["nc"], ds_cfg["names"])
