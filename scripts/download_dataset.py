"""
Dataset Download Script for FedX-PALM.

Downloads the Palm Fruit Ripeness Detection dataset from Roboflow
and splits it across 4 federated learning clients using Non-IID
Dirichlet distribution as specified in the thesis.

6 Classes: Unripe, Underripe, Ripe, Overripe, Abnormal, Empty Bunch

Non-IID Distribution (Dirichlet α per client):
  - Client A (α=0.1): Dominated by Unripe & Underripe
  - Client B (α=0.3): Dominated by Underripe & Ripe
  - Client C (α=0.5): Fairly balanced distribution
  - Client D (α=0.7): Dominated by Abnormal & Empty Bunch

Usage:
    python scripts/download_dataset.py --api-key YOUR_API_KEY
    python scripts/download_dataset.py --api-key YOUR_API_KEY --split-strategy non_iid
"""

import os
import sys
import shutil
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 6 classes as defined in thesis Section 2.1.2
CLASS_NAMES = {
    0: "Unripe",
    1: "Underripe",
    2: "Ripe",
    3: "Overripe",
    4: "Abnormal",
    5: "Empty Bunch"
}

# Dirichlet α per client as defined in thesis Table 3.3
DIRICHLET_ALPHA_PER_CLIENT = {
    "client_1": 0.1,  # Client A: highly skewed → Unripe/Underripe dominant
    "client_2": 0.3,  # Client B: moderately skewed → Underripe/Ripe dominant
    "client_3": 0.5,  # Client C: fairly balanced
    "client_4": 0.7,  # Client D: near-balanced → Abnormal/Empty Bunch emphasis
}


def download_dataset(api_key: str, output_dir: str = "./data/raw") -> str:
    """
    Download Palm Fruit Ripeness Detection dataset from Roboflow.

    Args:
        api_key: Roboflow API key
        output_dir: Directory to save the dataset

    Returns:
        Path to the downloaded dataset
    """
    from roboflow import Roboflow

    logger.info("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    logger.info("Accessing workspace and project...")
    project = rf.workspace("dydy-worker").project("palm-fruit-ripeness-detection-f6sac-ccb2z")

    logger.info("Downloading dataset version 2 in YOLOv11 format...")
    version = project.version(2)
    dataset = version.download("yolov11", location=output_dir)

    logger.info(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def get_image_class(label_path: Path) -> int:
    """
    Get the primary class of an image from its YOLO label file.

    Args:
        label_path: Path to the .txt label file

    Returns:
        Primary class index (0-5), or 0 if label not found
    """
    if not label_path.exists():
        return 0

    with open(label_path) as f:
        lines = f.readlines()
        if lines:
            # Use the first annotation's class as primary
            return int(lines[0].split()[0])
    return 0


def dirichlet_non_iid_split(
    images_by_class: dict,
    num_clients: int = 4,
    alpha_per_client: dict = None,
    seed: int = 42
) -> list:
    """
    Split dataset using Dirichlet distribution for Non-IID allocation.

    As described in thesis Section 3.3.4:
    pk ~ Dir(α) where pk is the class proportion vector for client k.

    Small α → highly skewed (Non-IID)
    Large α → near uniform (IID-like)

    Args:
        images_by_class: Dict mapping class_id -> list of image paths
        num_clients: Number of FL clients
        alpha_per_client: Dict mapping client_name -> α value
        seed: Random seed

    Returns:
        List of lists, where splits[k] contains image paths for client k
    """
    np.random.seed(seed)
    random.seed(seed)

    if alpha_per_client is None:
        alpha_per_client = DIRICHLET_ALPHA_PER_CLIENT

    num_classes = len(images_by_class)
    splits = [[] for _ in range(num_clients)]

    # For each client, sample a proportion vector from Dir(α)
    # Then allocate images from each class according to those proportions
    client_proportions = {}
    for k, (client_name, alpha) in enumerate(alpha_per_client.items()):
        # Sample class proportions from Dirichlet distribution
        # Use alpha as concentration parameter for all classes
        proportions = np.random.dirichlet([alpha] * num_classes)
        client_proportions[k] = proportions
        logger.info(
            f"  {client_name} (α={alpha}): "
            f"class proportions = [{', '.join(f'{p:.3f}' for p in proportions)}]"
        )

    # Normalize proportions so they sum to 1 across clients for each class
    for class_id, class_images in images_by_class.items():
        random.shuffle(class_images)
        n_images = len(class_images)

        if n_images == 0:
            continue

        # Get each client's proportion for this class
        props = np.array([client_proportions[k][class_id] for k in range(num_clients)])
        props = props / props.sum()  # Normalize

        # Allocate images according to proportions
        allocated = 0
        for k in range(num_clients):
            if k == num_clients - 1:
                # Last client gets the remainder
                n_alloc = n_images - allocated
            else:
                n_alloc = int(round(props[k] * n_images))
                n_alloc = min(n_alloc, n_images - allocated)

            splits[k].extend(class_images[allocated:allocated + n_alloc])
            allocated += n_alloc

    return splits


def iid_split(images: list, num_clients: int = 4, seed: int = 42) -> list:
    """Split images uniformly (IID) across clients."""
    random.seed(seed)
    random.shuffle(images)

    chunk_size = len(images) // num_clients
    splits = []
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(images)
        splits.append(images[start:end])
    return splits


def split_dataset_for_clients(
    dataset_dir: str,
    num_clients: int = 4,
    output_base: str = "./data",
    split_strategy: str = "non_iid",
    seed: int = 42
):
    """
    Split dataset across multiple FL clients.

    Supports:
    - non_iid: Dirichlet-based Non-IID split (thesis default)
    - iid: Uniform random split

    Args:
        dataset_dir: Path to the downloaded dataset
        num_clients: Number of FL clients (default: 4)
        output_base: Base output directory
        split_strategy: 'non_iid' (Dirichlet) or 'iid' (uniform)
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_dir)
    output_base = Path(output_base)

    # Find train images and labels
    train_images_dir = dataset_path / "train" / "images"
    train_labels_dir = dataset_path / "train" / "labels"
    val_images_dir = dataset_path / "valid" / "images"
    val_labels_dir = dataset_path / "valid" / "labels"

    if not train_images_dir.exists():
        # Try alternative structure
        train_images_dir = dataset_path / "images" / "train"
        train_labels_dir = dataset_path / "labels" / "train"
        val_images_dir = dataset_path / "images" / "val"
        val_labels_dir = dataset_path / "labels" / "val"

    if not train_images_dir.exists():
        logger.error(f"Could not find training images in {dataset_path}")
        logger.info("Available directories:")
        for p in dataset_path.rglob("*"):
            if p.is_dir():
                logger.info(f"  {p}")
        return

    # Get all training images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    train_images = [
        f for f in train_images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    logger.info(f"Found {len(train_images)} training images")

    # Group images by class for Non-IID split
    images_by_class = defaultdict(list)
    for img_path in train_images:
        label_path = train_labels_dir / (img_path.stem + ".txt")
        class_id = get_image_class(label_path)
        images_by_class[class_id].append(img_path)

    # Log class distribution
    logger.info("Dataset class distribution:")
    for cls_id in sorted(images_by_class.keys()):
        cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        logger.info(f"  {cls_name} (C{cls_id+1}): {len(images_by_class[cls_id])} images")

    # Split images across clients
    if split_strategy == "non_iid":
        logger.info(f"\nApplying Non-IID Dirichlet split (4 clients):")
        splits = dirichlet_non_iid_split(
            images_by_class, num_clients, seed=seed
        )
    else:
        logger.info(f"\nApplying IID uniform split ({num_clients} clients):")
        splits = iid_split(train_images, num_clients, seed=seed)

    # Get validation images (shared across all clients)
    val_images = []
    if val_images_dir.exists():
        val_images = [
            f for f in val_images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

    # Create client directories and copy data
    for client_idx in range(num_clients):
        client_name = f"client_{client_idx + 1}"
        client_dir = output_base / client_name
        client_train_img = client_dir / "images" / "train"
        client_train_lbl = client_dir / "labels" / "train"
        client_val_img = client_dir / "images" / "val"
        client_val_lbl = client_dir / "labels" / "val"

        # Create directories
        client_train_img.mkdir(parents=True, exist_ok=True)
        client_train_lbl.mkdir(parents=True, exist_ok=True)
        client_val_img.mkdir(parents=True, exist_ok=True)
        client_val_lbl.mkdir(parents=True, exist_ok=True)

        # Copy training data for this client
        client_images = splits[client_idx]
        for img_path in client_images:
            # Copy image
            shutil.copy2(img_path, client_train_img / img_path.name)

            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            label_path = train_labels_dir / label_name
            if label_path.exists():
                shutil.copy2(label_path, client_train_lbl / label_name)

        # Copy validation data (all clients share the same val set)
        for img_path in val_images:
            shutil.copy2(img_path, client_val_img / img_path.name)

            label_name = img_path.stem + ".txt"
            label_path = val_labels_dir / label_name
            if label_path.exists():
                shutil.copy2(label_path, client_val_lbl / label_name)

        # Create data.yaml for this client
        data_yaml_content = _generate_data_yaml(client_dir)
        with open(client_dir / "data.yaml", "w") as f:
            f.write(data_yaml_content)

        # Log class distribution for this client
        client_class_dist = defaultdict(int)
        for img_path in client_images:
            label_path = train_labels_dir / (img_path.stem + ".txt")
            cls_id = get_image_class(label_path)
            client_class_dist[cls_id] += 1

        alpha = list(DIRICHLET_ALPHA_PER_CLIENT.values())[client_idx]
        logger.info(
            f"\n{client_name} (α={alpha}): {len(client_images)} train, "
            f"{len(val_images)} val images"
        )
        for cls_id in sorted(client_class_dist.keys()):
            cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            count = client_class_dist[cls_id]
            pct = count / max(len(client_images), 1) * 100
            logger.info(f"    {cls_name}: {count} ({pct:.1f}%)")

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset split complete: {num_clients} clients ({split_strategy})")
    logger.info(f"{'='*50}")


def _generate_data_yaml(client_dir: Path) -> str:
    """Generate data.yaml content for a client (6 classes)."""
    return f"""# Palm Fruit Ripeness Detection - Client Data Configuration
# Auto-generated by FedX-PALM dataset split script
# 6 Classes as per thesis specification

path: {client_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 6

# Class names for Palm Fruit Ripeness (6 classes)
names:
  0: Unripe
  1: Underripe
  2: Ripe
  3: Overripe
  4: Abnormal
  5: Empty Bunch
"""


def main():
    parser = argparse.ArgumentParser(
        description="Download and split Palm Fruit Ripeness dataset for Federated Learning"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Directory to download raw dataset"
    )
    parser.add_argument(
        "--split-clients",
        type=int,
        default=4,
        help="Number of FL clients to split data for"
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="non_iid",
        choices=["iid", "non_iid"],
        help="Data split strategy: non_iid (Dirichlet, thesis default) or iid (uniform)"
    )
    parser.add_argument(
        "--client-data-dir",
        type=str,
        default="./data",
        help="Base directory for client data splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use existing data in output-dir)"
    )

    args = parser.parse_args()

    if not args.api_key and not args.skip_download:
        logger.error(
            "Roboflow API key required. Provide via --api-key or ROBOFLOW_API_KEY env var"
        )
        sys.exit(1)

    # Step 1: Download dataset
    if not args.skip_download:
        dataset_dir = download_dataset(args.api_key, args.output_dir)
    else:
        dataset_dir = args.output_dir
        logger.info(f"Skipping download, using existing data at: {dataset_dir}")

    # Step 2: Split for FL clients using Non-IID Dirichlet
    if args.split_clients > 0:
        split_dataset_for_clients(
            dataset_dir=dataset_dir,
            num_clients=args.split_clients,
            output_base=args.client_data_dir,
            split_strategy=args.split_strategy,
            seed=args.seed
        )

    logger.info("\nDone! Dataset ready for federated learning.")
    logger.info("Distribution strategy: Non-IID Dirichlet (α varies per client)")
    logger.info("  Client A (α=0.1): Highly skewed → Unripe/Underripe dominant")
    logger.info("  Client B (α=0.3): Moderately skewed → Underripe/Ripe dominant")
    logger.info("  Client C (α=0.5): Fairly balanced")
    logger.info("  Client D (α=0.7): Near-balanced → Abnormal/Empty Bunch emphasis")


if __name__ == "__main__":
    main()
