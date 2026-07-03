"""Non-IID partitioning of the training split across K federated clients.

Uses the latent-Dirichlet-allocation partition scheme (Hsu et al., 2019):
for each class c, draw a proportion vector over the K clients from
Dirichlet(alpha, ..., alpha) and split that class's images across clients
according to the draw. Small alpha => a class's images concentrate on a
few clients (high heterogeneity, mimicking distinct plantations); large
alpha => near-uniform (IID-like).

Object-detection images can contain boxes of several classes, so each
image is assigned a single "primary class" = the most frequent class
among its YOLO-format label boxes, which is what the Dirichlet draw
partitions on.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def _primary_class(label_path: Path) -> int | None:
    if not label_path.exists():
        return None
    counts = Counter()
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts[int(line.split()[0])] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def dirichlet_partition(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    k: int,
    alpha: float,
    seed: int,
) -> dict[int, list[str]]:
    """Returns {client_id: [image_stem, ...]}."""
    rng = np.random.default_rng(seed)

    by_class: dict[int, list[str]] = defaultdict(list)
    unlabeled: list[str] = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cls = _primary_class(labels_dir / (img_path.stem + ".txt"))
        if cls is None:
            unlabeled.append(img_path.stem)
        else:
            by_class[cls].append(img_path.stem)

    client_samples: dict[int, list[str]] = {c: [] for c in range(k)}

    for cls in range(num_classes):
        stems = by_class.get(cls, [])
        if not stems:
            continue
        rng.shuffle(stems)
        proportions = rng.dirichlet(alpha=np.full(k, alpha))
        # cumulative split points over len(stems)
        split_points = (np.cumsum(proportions) * len(stems)).astype(int)[:-1]
        chunks = np.split(stems, split_points)
        for client_id, chunk in enumerate(chunks):
            client_samples[client_id].extend(chunk.tolist())

    # background/unlabeled images: round-robin so no client is starved
    for i, stem in enumerate(unlabeled):
        client_samples[i % k].append(stem)

    return client_samples


def partition_and_write(
    cfg_path: str = "configs/dataset.yaml",
    fl_cfg_path: str = "configs/fl_config.yaml",
    k: int | None = None,
) -> Path:
    with open(cfg_path) as f:
        ds_cfg = yaml.safe_load(f)
    with open(fl_cfg_path) as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    train_images = splits_dir / "train" / "images"
    train_labels = splits_dir / "train" / "labels"

    alpha = fl_cfg["clients"]["dirichlet_alpha"]
    seed = fl_cfg["clients"]["partition_seed"]
    k_values = [k] if k is not None else fl_cfg["clients"]["k_values"]

    out_root = splits_dir / "federated_partitions"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for kk in k_values:
        assignment = dirichlet_partition(
            train_images, train_labels, ds_cfg["nc"], kk, alpha, seed
        )
        sizes = {cid: len(stems) for cid, stems in assignment.items()}
        out_path = out_root / f"k{kk}.json"
        with open(out_path, "w") as f:
            json.dump(assignment, f)
        print(f"K={kk}: client sample counts = {sizes}")
        manifest[kk] = {"path": str(out_path), "sizes": sizes}

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--fl-config", default="configs/fl_config.yaml")
    parser.add_argument("--k", type=int, default=None, help="partition only this K (default: sweep all k_values)")
    args = parser.parse_args()
    partition_and_write(args.config, args.fl_config, args.k)
