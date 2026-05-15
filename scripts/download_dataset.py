"""
Dataset Download Script for FedX-PALM.

Downloads the Palm Fruit Ripeness Detection dataset from Roboflow
and optionally splits it across multiple federated learning clients.

Usage:
    python scripts/download_dataset.py --api-key YOUR_API_KEY
    python scripts/download_dataset.py --api-key YOUR_API_KEY --split-clients 3
"""

import os
import sys
import shutil
import random
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(api_key: str, output_dir: str = "./data") -> str:
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


def split_dataset_for_clients(
    dataset_dir: str,
    num_clients: int = 3,
    output_base: str = "./data",
    split_strategy: str = "iid",
    seed: int = 42
):
    """
    Split dataset across multiple FL clients.

    Supports:
    - IID: Each client gets a random uniform split
    - Non-IID: Each client gets a skewed class distribution

    Args:
        dataset_dir: Path to the downloaded dataset
        num_clients: Number of FL clients to split data for
        output_base: Base output directory
        split_strategy: 'iid' or 'non_iid'
        seed: Random seed for reproducibility
    """
    random.seed(seed)
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
    random.shuffle(train_images)

    logger.info(f"Found {len(train_images)} training images")

    # Split images across clients
    if split_strategy == "iid":
        splits = _iid_split(train_images, num_clients)
    else:
        splits = _non_iid_split(train_images, train_labels_dir, num_clients)

    # Get validation images
    val_images = []
    if val_images_dir.exists():
        val_images = [
            f for f in val_images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

    # Create client directories and copy data
    for client_idx in range(num_clients):
        client_dir = output_base / f"client_{client_idx + 1}"
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

        logger.info(
            f"Client {client_idx + 1}: {len(client_images)} train images, "
            f"{len(val_images)} val images -> {client_dir}"
        )

    logger.info(f"Dataset split complete for {num_clients} clients ({split_strategy})")


def _iid_split(images: list, num_clients: int) -> list:
    """Split images uniformly (IID) across clients."""
    chunk_size = len(images) // num_clients
    splits = []
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(images)
        splits.append(images[start:end])
    return splits


def _non_iid_split(images: list, labels_dir: Path, num_clients: int) -> list:
    """
    Split images non-IID based on class distribution.
    Each client gets a skewed subset of classes.
    """
    # Group images by their primary class
    class_images = {}
    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        primary_class = 0
        if label_path.exists():
            with open(label_path) as f:
                lines = f.readlines()
                if lines:
                    primary_class = int(lines[0].split()[0])

        if primary_class not in class_images:
            class_images[primary_class] = []
        class_images[primary_class].append(img_path)

    # Distribute classes unevenly across clients
    all_classes = sorted(class_images.keys())
    splits = [[] for _ in range(num_clients)]

    for i, cls in enumerate(all_classes):
        # Primary client gets 60% of this class, rest shared
        primary_client = i % num_clients
        cls_imgs = class_images[cls]
        random.shuffle(cls_imgs)

        primary_share = int(len(cls_imgs) * 0.6)
        splits[primary_client].extend(cls_imgs[:primary_share])

        # Distribute remaining across other clients
        remaining = cls_imgs[primary_share:]
        other_clients = [c for c in range(num_clients) if c != primary_client]
        for j, img in enumerate(remaining):
            splits[other_clients[j % len(other_clients)]].append(img)

    return splits


def _generate_data_yaml(client_dir: Path) -> str:
    """Generate data.yaml content for a client."""
    return f"""# Palm Fruit Ripeness Detection - Client Data Configuration
# Auto-generated by FedX-PALM dataset split script

path: {client_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 4

# Class names for Palm Fruit Ripeness
names:
  0: unripe
  1: underripe
  2: ripe
  3: overripe
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
        default="iid",
        choices=["iid", "non_iid"],
        help="Data split strategy: iid (uniform) or non_iid (skewed)"
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

    # Step 2: Split for FL clients
    if args.split_clients > 0:
        split_dataset_for_clients(
            dataset_dir=dataset_dir,
            num_clients=args.split_clients,
            output_base=args.client_data_dir,
            split_strategy=args.split_strategy,
            seed=args.seed
        )

    logger.info("Done! Dataset ready for federated learning.")


if __name__ == "__main__":
    main()
