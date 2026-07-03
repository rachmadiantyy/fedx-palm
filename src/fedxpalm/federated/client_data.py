"""Materialize per-client train folders (symlinked/hardlinked/copied) from a Dirichlet partition manifest.

Built once per K before federated training starts; every client re-uses the
same shared val/test split (federated evaluation happens on the server's
global held-out set, never on client-local data, matching the original
FedX-Palm design).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import shutil

import yaml


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink if possible, else hardlink, else fall back to a real copy.

    os.symlink() raises OSError [WinError 1314] on Windows unless the
    process runs elevated or Developer Mode is enabled -- not something we
    can assume of every machine this pipeline runs on. Hardlinks need no
    special privilege and work as long as src/dst are on the same volume
    (true here, both under data/splits/); a plain copy is the universal
    fallback if even that fails (e.g. different drives).
    """
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def materialize_clients(
    partition_json: str,
    splits_dir: str,
    out_dir: str,
    nc: int,
    names: list[str],
) -> list[str]:
    with open(partition_json) as f:
        assignment: dict[str, list[str]] = json.load(f)

    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)
    train_images = splits_dir / "train" / "images"
    train_labels = splits_dir / "train" / "labels"

    data_yaml_paths = []
    for client_id, stems in assignment.items():
        client_dir = out_dir / f"client{client_id}"
        img_link_dir = client_dir / "images"
        lbl_link_dir = client_dir / "labels"
        img_link_dir.mkdir(parents=True, exist_ok=True)
        lbl_link_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            src_img = next(train_images.glob(f"{stem}.*"))
            dst_img = img_link_dir / src_img.name
            if not dst_img.exists():
                _link_or_copy(src_img.resolve(), dst_img)
            src_lbl = train_labels / f"{stem}.txt"
            dst_lbl = lbl_link_dir / f"{stem}.txt"
            if src_lbl.exists() and not dst_lbl.exists():
                _link_or_copy(src_lbl.resolve(), dst_lbl)

        data_yaml = {
            "path": str(client_dir.resolve()),
            "train": "images",
            "val": str((splits_dir / "val" / "images").resolve()),
            "test": str((splits_dir / "test" / "images").resolve()),
            "nc": nc,
            "names": names,
        }
        yaml_path = client_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        data_yaml_paths.append(str(yaml_path))

    print(f"Materialized {len(assignment)} client folders under {out_dir}")
    return data_yaml_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    args = parser.parse_args()

    with open(args.dataset_config) as f:
        ds_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    partition_json = splits_dir / "federated_partitions" / f"k{args.k}.json"
    out_dir = splits_dir / "federated_partitions" / f"k{args.k}_clients"
    materialize_clients(str(partition_json), str(splits_dir), str(out_dir), ds_cfg["nc"], ds_cfg["names"])
