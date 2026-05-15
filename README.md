# FedX-PALM: Federated Learning with Explainable AI & Differential Privacy

A privacy-preserving federated learning system for object detection using **YOLOv11**, with built-in **Explainable AI (XAI)** and **Differential Privacy (DP)** mechanisms, orchestrated with **Docker**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FedX-PALM System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   FL Server (Aggregator)                   │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │  FedAvg /   │  │  Central DP  │  │  Model Version │  │   │
│  │  │  FedProx    │  │  (Noise Add) │  │  Management    │  │   │
│  │  └─────────────┘  └──────────────┘  └────────────────┘  │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │ REST API                           │
│           ┌─────────────────┼─────────────────┐                 │
│           │                 │                 │                  │
│  ┌────────▼───────┐ ┌──────▼────────┐ ┌──────▼────────┐       │
│  │   FL Client 1  │ │  FL Client 2  │ │  FL Client 3  │       │
│  │ ┌────────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │       │
│  │ │  YOLOv11   │ │ │ │  YOLOv11  │ │ │ │  YOLOv11  │ │       │
│  │ │  (Local)   │ │ │ │  (Local)  │ │ │ │  (Local)  │ │       │
│  │ ├────────────┤ │ │ ├───────────┤ │ │ ├───────────┤ │       │
│  │ │  Local DP  │ │ │ │ Local DP  │ │ │ │ Local DP  │ │       │
│  │ │  (Clip+    │ │ │ │ (Clip+   │ │ │ │ (Clip+   │ │       │
│  │ │   Noise)   │ │ │ │  Noise)  │ │ │ │  Noise)  │ │       │
│  │ ├────────────┤ │ │ ├───────────┤ │ │ ├───────────┤ │       │
│  │ │    XAI     │ │ │ │   XAI    │ │ │ │   XAI    │ │       │
│  │ │ (Grad-CAM  │ │ │ │(Grad-CAM │ │ │ │(Grad-CAM │ │       │
│  │ │  + SHAP)   │ │ │ │ + SHAP)  │ │ │ │ + SHAP)  │ │       │
│  │ └────────────┘ │ │ └───────────┘ │ │ └───────────┘ │       │
│  │  Private Data  │ │ Private Data  │ │ Private Data  │       │
│  └────────────────┘ └───────────────┘ └───────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Federated Learning
- **FedAvg**: Federated Averaging for model aggregation
- **FedProx**: Proximal term for heterogeneous data handling
- **Flexible client selection**: Configurable fraction of clients per round
- **Model checkpointing**: Save and resume training state

### Differential Privacy
- **Local DP**: Gradient clipping + Gaussian noise on client side
- **Central DP**: Server-side noise addition after aggregation
- **Privacy budget tracking**: Epsilon/delta accounting per round
- **Moments Accountant**: Tight privacy analysis using RDP (Renyi DP)
- **Configurable parameters**: Epsilon, delta, noise multiplier, clipping norm

### Explainable AI (XAI)
- **Grad-CAM**: Visual explanations of model focus regions
- **SHAP**: Shapley value-based feature importance
- **Privacy-aware explanations**: DP noise added to prevent information leakage
- **Feature importance tracking**: Monitor layer importance across FL rounds
- **Comprehensive reports**: JSON reports with overlay visualizations

### YOLOv11 Integration
- **Ultralytics YOLOv11**: State-of-the-art object detection
- **Multiple variants**: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- **Transfer learning**: Backbone freezing support
- **FL-ready wrapper**: Parameter extraction, delta computation, serialization

### Docker Deployment
- **Multi-container orchestration**: Docker Compose for server + clients
- **GPU support**: NVIDIA CUDA-enabled containers
- **CPU fallback**: Separate compose file for CPU-only environments
- **Health checks**: Automatic container health monitoring
- **Volume management**: Persistent checkpoints and results

---

## Project Structure

```
fedx-palm/
├── server/
│   ├── __init__.py
│   ├── fed_server.py          # FL server logic (aggregation, DP)
│   └── grpc_server.py         # REST API server (Flask)
├── client/
│   ├── __init__.py
│   └── fed_client.py          # FL client (local training, DP)
├── models/
│   ├── __init__.py
│   └── yolov11_wrapper.py     # YOLOv11 federated wrapper
├── utils/
│   ├── __init__.py
│   └── differential_privacy.py # DP mechanisms & accounting
├── xai/
│   ├── __init__.py
│   └── explainer.py           # Grad-CAM, SHAP, privacy-aware XAI
├── configs/
│   ├── server_config.yaml     # Server configuration
│   ├── client_config.yaml     # Client configuration
│   └── data_sample.yaml       # Sample data config (COCO format)
├── docker/
│   ├── Dockerfile.server      # Server container
│   └── Dockerfile.client      # Client container
├── data/                       # Dataset directory (not tracked)
├── docker-compose.yml          # Full deployment (GPU)
├── docker-compose.cpu.yml      # CPU-only deployment
├── requirements.txt            # Python dependencies
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU mode)
- Python 3.10+ (for local development)

### 1. Clone the Repository

```bash
git clone https://github.com/rachmadiantyy/fedx-palm.git
cd fedx-palm
```

### 2. Prepare Your Data

Organize your data for each client in YOLO format:

```
data/
├── client_1/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── client_2/
│   └── ...
└── client_3/
    └── ...
```

Each `data.yaml` should follow the format in `configs/data_sample.yaml`.

### 3. Deploy with Docker

**GPU Mode (recommended):**
```bash
docker-compose up --build
```

**CPU Mode (development/testing):**
```bash
docker-compose -f docker-compose.cpu.yml up --build
```

### 4. Monitor Training

Check server status:
```bash
curl http://localhost:8080/status
```

View privacy report:
```bash
curl http://localhost:8080/privacy_report
```

View round history:
```bash
curl http://localhost:8080/round_history
```

---

## Configuration

### Server Configuration (`configs/server_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rounds` | 100 | Total federated learning rounds |
| `min_clients` | 2 | Minimum clients per round |
| `fraction_fit` | 1.0 | Client selection fraction |
| `aggregation_strategy` | fedavg | Aggregation method (fedavg/fedprox) |
| `dp_epsilon` | 1.0 | Total privacy budget |
| `dp_delta` | 1e-5 | Privacy failure probability |
| `dp_max_grad_norm` | 1.0 | Gradient clipping norm |
| `dp_noise_multiplier` | 1.0 | Noise scale multiplier |

### Client Configuration (`configs/client_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `local_epochs` | 5 | Training epochs per round |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 0.01 | Learning rate |
| `dp_epsilon` | 1.0 | Local privacy budget |
| `dp_max_grad_norm` | 1.0 | Local gradient clipping |
| `fedprox_mu` | 0.01 | FedProx proximal coefficient |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| POST | `/register` | Register a new FL client |
| GET | `/global_model` | Get current global model |
| POST | `/submit_update` | Submit client model update |
| POST | `/aggregate` | Force aggregation (admin) |
| GET | `/status` | Get server status |
| GET | `/privacy_report` | Get DP privacy report |
| GET | `/round_history` | Get training history |
| POST | `/checkpoint` | Save server checkpoint |

---

## Differential Privacy Details

The system implements a two-level DP approach:

### Local DP (Client-side)
1. **Gradient Clipping**: Bounds the L2 norm of model updates
2. **Gaussian Noise**: Adds calibrated noise to clipped updates
3. **Privacy Budget**: Each client tracks its own epsilon consumption

### Central DP (Server-side)
1. **Aggregation**: Weighted average of client updates
2. **Noise Addition**: Server adds noise to aggregated model
3. **Moments Accountant**: Tight privacy analysis using RDP

### Privacy Guarantee
- Achieves (epsilon, delta)-differential privacy
- Configurable privacy-utility trade-off
- Automatic training halt when budget is exhausted

---

## Explainable AI (XAI) Details

### Grad-CAM
Generates heatmaps showing which image regions most influence detection decisions. Useful for verifying the model focuses on relevant objects.

### SHAP (Kernel SHAP)
Computes Shapley values via image segmentation and perturbation. Shows positive (red) and negative (blue) contributions of image regions.

### Privacy-Aware Explanations
Adds Laplacian noise to explanation heatmaps to prevent membership inference attacks through explanation outputs.

---

## Scaling Clients

To add more clients, extend `docker-compose.yml`:

```yaml
fl-client-N:
  build:
    context: .
    dockerfile: docker/Dockerfile.client
  volumes:
    - ./data/client_N:/app/data
  networks:
    - fl-network
  depends_on:
    fl-server:
      condition: service_healthy
  command: >
    --client-id client_N
    --server-url http://fl-server:8080
    --data-config /app/data/data.yaml
    --local-epochs 5
    --dp-enabled
```

---

## Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server locally
python -m server.grpc_server --port 8080 --dp-enabled

# Run client locally (in another terminal)
python -m client.fed_client \
  --client-id dev_client \
  --server-url http://localhost:8080 \
  --data-config ./data/data.yaml \
  --local-epochs 2 \
  --dp-enabled
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
