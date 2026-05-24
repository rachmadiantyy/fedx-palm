#!/usr/bin/env bash
# FedX-Palm FAST DEMO orchestrator untuk VPS.
# Jalankan dari root repo: bash scripts/run_fast_demo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "=========================================="
echo " FedX-Palm FAST DEMO (2 client, 1 ronde)"
echo "=========================================="

if [[ ! -d data/client_1 || ! -d data/client_2 ]]; then
  echo "[setup] Dataset belum ada — siapkan via prepare_demo_data.sh"
  bash scripts/prepare_demo_data.sh
fi

echo "[build] Build CPU image (server + client) — sekali aja, ~10-15 menit"
docker compose -f docker-compose.fast-demo.yml build

echo "[up] Jalankan demo (logs streaming)…"
docker compose -f docker-compose.fast-demo.yml up --abort-on-container-exit

echo
echo "Selesai. Lihat checkpoint di ./checkpoints_demo dan results/"
