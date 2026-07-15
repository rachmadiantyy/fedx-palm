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


def _client_stats(stems: list[str], labels_dir: Path, num_classes: int) -> dict:
    """Per-client audit info: bounding-box count per class, total boxes, and
    the (unique) bunch_ids covered -- so partition skew is inspectable from
    the manifest alone, without re-reading thousands of label files."""
    from fedxpalm.data.split import _parse_bunch_id  # local import: avoids a cycle at module load

    class_counts = Counter()
    for stem in stems:
        lbl = labels_dir / f"{stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if line:
                class_counts[int(line.split()[0])] += 1
    bunch_ids = sorted({_parse_bunch_id(Path(stem + ".jpg")) for stem in stems})
    return {
        "num_images": len(stems),
        "num_boxes": sum(class_counts.values()),
        "class_boxes": {c: class_counts.get(c, 0) for c in range(num_classes)},
        "classes_missing": [c for c in range(num_classes) if class_counts.get(c, 0) == 0],
        "num_bunches": len(bunch_ids),
        "bunch_ids": bunch_ids,
    }


def partition_and_write(
    cfg_path: str = "configs/dataset.yaml",
    fl_cfg_path: str = "configs/fl_config.yaml",
    k: int | None = None,
    seed: int | None = None,
) -> Path:
    """`seed=None` uses configs/fl_config.yaml's clients.partition_seed and the
    default `federated_partitions/` output root (backward compatible with
    partitions already materialized on disk). An explicit different seed writes
    to `federated_partitions_seed{seed}/` instead, so multi-seed reproducibility
    runs never overwrite the primary partition."""
    with open(cfg_path) as f:
        ds_cfg = yaml.safe_load(f)
    with open(fl_cfg_path) as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    train_images = splits_dir / "train" / "images"
    train_labels = splits_dir / "train" / "labels"

    alpha = fl_cfg["clients"]["dirichlet_alpha"]
    cfg_seed = fl_cfg["clients"]["partition_seed"]
    seed = cfg_seed if seed is None else seed
    k_values = [k] if k is not None else fl_cfg["clients"]["k_values"]

    root_name = "federated_partitions" if seed == cfg_seed else f"federated_partitions_seed{seed}"
    out_root = splits_dir / root_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Merge with an existing manifest instead of starting from {}: running
    # e.g. `--k 4` used to rewrite manifest.json with ONLY the k=4 entry,
    # silently dropping every other K's entry from a prior full sweep.
    manifest_path = out_root / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    for kk in k_values:
        assignment = dirichlet_partition(
            train_images, train_labels, ds_cfg["nc"], kk, alpha, seed
        )
        sizes = {cid: len(stems) for cid, stems in assignment.items()}
        out_path = out_root / f"k{kk}.json"
        with open(out_path, "w") as f:
            json.dump(assignment, f)
        print(f"K={kk}: client sample counts = {sizes}")

        clients = {cid: _client_stats(stems, train_labels, ds_cfg["nc"])
                   for cid, stems in assignment.items()}
        for cid, st in clients.items():
            if st["classes_missing"]:
                print(f"  WARNING K={kk} client {cid}: zero boxes for class(es) "
                      f"{st['classes_missing']} -- alpha={alpha} may be too extreme for this K")
        # str key: json.load round-trips keys as strings, so int keys here
        # would coexist with (not replace) the same K's old string entry
        manifest[str(kk)] = {
            "path": str(out_path),
            "sizes": sizes,
            "dirichlet_alpha": alpha,
            "partition_seed": seed,
            "clients": clients,
        }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--fl-config", default="configs/fl_config.yaml")
    parser.add_argument("--k", type=int, default=None, help="partition only this K (default: sweep all k_values)")
    parser.add_argument("--seed", type=int, default=None,
                        help="partition seed override (default: fl_config clients.partition_seed; "
                             "a non-default seed writes to federated_partitions_seed{seed}/)")
    args = parser.parse_args()
    partition_and_write(args.config, args.fl_config, args.k, seed=args.seed)
