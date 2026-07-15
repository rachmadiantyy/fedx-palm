"""Download the oil palm FFB ripeness dataset from Roboflow.

Requires network access to api.roboflow.com -- this only works on the
machine you actually train on (Colab / your GPU box), not inside a
network-restricted sandbox.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def download_dataset(cfg_path: str = "configs/dataset.yaml") -> Path:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    rf_cfg = cfg["roboflow"]
    api_key = os.environ.get("ROBOFLOW_API_KEY") or rf_cfg.get("api_key")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Export it first, e.g.\n"
            "  Windows:  set ROBOFLOW_API_KEY=<your key>\n"
            "  Linux:    export ROBOFLOW_API_KEY=<your key>\n"
            "(the key is deliberately no longer stored in configs/dataset.yaml)")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(rf_cfg["workspace"]).project(rf_cfg["project"])
    version = project.version(rf_cfg["version"])

    download_dir = Path(cfg["download_dir"])
    download_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = version.download(rf_cfg["format"], location=str(download_dir))

    data_yaml = Path(dataset.location) / "data.yaml"
    with open(data_yaml) as f:
        downloaded_names = yaml.safe_load(f).get("names")

    expected_names = cfg["names"]
    if downloaded_names != expected_names:
        print(
            "WARNING: class names in the downloaded data.yaml do not match "
            f"configs/dataset.yaml.\n  downloaded: {downloaded_names}\n"
            f"  configured: {expected_names}\n"
            "Update configs/dataset.yaml `names` (and `nc`) to match the "
            "downloaded order before running the rest of the pipeline -- "
            "class ids are positional."
        )

    print(f"Dataset downloaded to: {dataset.location}")
    return Path(dataset.location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()
    download_dataset(args.config)
