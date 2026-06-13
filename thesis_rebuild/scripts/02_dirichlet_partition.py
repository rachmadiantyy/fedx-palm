"""
[02] Non-IID Dirichlet partition into K=4 federated clients.

Step 2 of the rebuild pipeline. Splits the (clean, leakage-free) TRAIN
set into 4 client shards with different Dirichlet alpha, representing
heterogeneous plantations (thesis Table 3.3). Validation and test sets
are SHARED across all clients so every client is evaluated on the same
held-out data — a fair comparison.

  Client 1: alpha=0.1  (highly skewed — 1-2 classes dominate)
  Client 2: alpha=0.3  (moderately skewed)
  Client 3: alpha=0.5  (mildly skewed)
  Client 4: alpha=0.7  (near-uniform / IID-like)

Deterministic: seed=42.

Usage:
    python thesis_rebuild/scripts/02_dirichlet_partition.py

Input:  data/resplit/{train,valid,test}/{images,labels}
Output: data/clients/client_{1..4}/{images,labels}  (train shards)
        data/clients/client_{1..4}/data.yaml         (points val->shared valid)
        data/global_val.yaml, data/global_test.yaml
"""
import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

NC = 6
CLASS_NAMES = {
    0: "Abnormal", 1: "Empty Bunch", 2: "Overripe",
    3: "Ripe", 4: "Underripe", 5: "Unripe",
}
ALPHA_PER_CLIENT = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def primary_class(lbl_path: Path) -> int:
    if not lbl_path.exists():
        return 0
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                return int(parts[0])
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="[02] Dirichlet Non-IID partition")
    p.add_argument("--resplit", type=str, default=str(REPO_ROOT / "data" / "resplit"))
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "clients"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resplit_dir = Path(args.resplit)
    clients_root = Path(args.out)
    data_root = clients_root.parent

    if not resplit_dir.exists():
        sys.exit(f"ERROR: {resplit_dir} not found. Run 01_resplit_bunch_id.py first.")

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 70)
    print("[02] DIRICHLET NON-IID PARTITION (K=4 clients)")
    print("=" * 70)

    train_img = resplit_dir / "train" / "images"
    train_lbl = resplit_dir / "train" / "labels"

    # Group train images by primary class
    by_class = defaultdict(list)
    for img in train_img.iterdir():
        if img.suffix.lower() in IMG_EXTS:
            by_class[primary_class(train_lbl / (img.stem + ".txt"))].append(img)
    for c in by_class:
        random.shuffle(by_class[c])
    print("  Train images per class:",
          {CLASS_NAMES[c]: len(by_class[c]) for c in sorted(by_class) if 0 <= c < NC})

    # Sample Dirichlet proportions per client
    client_props = {}
    for k, alpha in ALPHA_PER_CLIENT.items():
        client_props[k] = np.random.dirichlet([alpha] * NC)
        print(f"  Client {k} (alpha={alpha}): {np.round(client_props[k], 3)}")

    # Allocate images per class to clients by proportion
    client_files = {k: [] for k in ALPHA_PER_CLIENT}
    for cls in sorted(by_class):
        if not (0 <= cls < NC):
            continue
        imgs = by_class[cls]
        n = len(imgs)
        props = np.array([client_props[k][cls] for k in ALPHA_PER_CLIENT])
        props = props / props.sum() if props.sum() > 0 else props
        allocated = 0
        keys = list(ALPHA_PER_CLIENT)
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                n_alloc = n - allocated            # last client takes remainder
            else:
                n_alloc = int(round(props[i] * n))
                n_alloc = min(n_alloc, n - allocated)
            client_files[k].extend(imgs[allocated:allocated + n_alloc])
            allocated += n_alloc

    # Materialize client train shards
    if clients_root.exists():
        shutil.rmtree(clients_root)
    for k in ALPHA_PER_CLIENT:
        (clients_root / f"client_{k}" / "images").mkdir(parents=True, exist_ok=True)
        (clients_root / f"client_{k}" / "labels").mkdir(parents=True, exist_ok=True)
        cnt = Counter()
        for img in client_files[k]:
            lbl = train_lbl / (img.stem + ".txt")
            shutil.copy2(img, clients_root / f"client_{k}" / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, clients_root / f"client_{k}" / "labels" / lbl.name)
            cnt[primary_class(lbl)] += 1
        pretty = {CLASS_NAMES[c]: cnt[c] for c in sorted(cnt) if 0 <= c < NC}
        print(f"  Client {k} ({len(client_files[k])} img): {pretty}")

        # Per-client data.yaml: local train, SHARED valid/test
        with open(clients_root / f"client_{k}" / "data.yaml", "w") as f:
            yaml.safe_dump({
                "path": str(resplit_dir),
                "train": str((clients_root / f"client_{k}" / "images").resolve()),
                "val": "valid/images",
                "test": "test/images",
                "nc": NC,
                "names": CLASS_NAMES,
            }, f, sort_keys=False)

    # Shared global val / test yamls
    for tag, split in (("global_val", "valid"), ("global_test", "test")):
        with open(data_root / f"{tag}.yaml", "w") as f:
            yaml.safe_dump({
                "path": str(resplit_dir),
                "train": "train/images",
                "val": f"{split}/images",
                "nc": NC,
                "names": CLASS_NAMES,
            }, f, sort_keys=False)

    print(f"\n  Clients -> {clients_root}")
    print(f"  Global val  -> {data_root / 'global_val.yaml'}")
    print(f"  Global test -> {data_root / 'global_test.yaml'}")
    print("\nNEXT: training")
    print("  B1: python thesis_rebuild/scripts/train_b1_centralized.py "
          "--data data/resplit/data.yaml")
    print("  E1: python thesis_rebuild/scripts/train_e1_dp_sgd_full.py "
          "--data data/resplit/data.yaml --sweep")


if __name__ == "__main__":
    main()
