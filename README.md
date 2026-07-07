# FedX-Palm

A YOLOv11-based Federated Learning framework with Differential Privacy and
Explainable AI for oil palm fresh fruit bunch (FFB) ripeness detection.

- **Federated Learning**: FedAvg across K simulated client nodes, Non-IID
  data via Dirichlet partitioning over Roboflow's own train/valid/test split.
- **Differential Privacy**: per-sample DP-SGD (Opacus), full and
  backbone-frozen variants, epsilon via the PRV accountant.
- **Explainable AI**: Grad-CAM++ with Average Drop / Focus Retention Rate
  faithfulness metrics.
- **Deployment**: Dockerized CPU-only Flask inference service with a
  Grad-CAM++ overlay on every prediction.

## Layout

```
configs/            dataset / federated learning / differential privacy config (YAML)
src/fedxpalm/         core Python package (data, models, federated, privacy, xai, eval)
scripts/             numbered pipeline entrypoints, 01 (download) through 10 (export results)
deployment/          Dockerfile, Flask inference service, HTML template
notebooks/           end-to-end Colab/Jupyter notebook
docs/thesis/          thesis chapters (Bahasa Indonesia) -- see docs/thesis/NOTES_FOR_RACHMA.md
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/01_download_dataset.py
python scripts/02_prepare_splits.py
python scripts/03_partition_clients.py
python scripts/04_prepare_base_model.py
python scripts/05_train_b1_centralized.py --device 0
python scripts/06_train_b2_federated.py --device 0
python scripts/07_train_e1_dp_full.py --device 0
python scripts/08_train_e2_dp_partial.py --device 0
python scripts/09_evaluate_xai.py --weights <checkpoint.pt> --tag <name> --save-overlays
python scripts/10_export_results.py
```

Or run `notebooks/FedXPalm_v2_Colab.ipynb` top to bottom (works on Colab
GPU or any Jupyter server with a GPU).

Requires a GPU for training at any reasonable speed; the Docker deployment
service (`deployment/`) is CPU-only by design.
