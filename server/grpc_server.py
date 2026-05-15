"""
gRPC Server for Federated Learning Communication.

Provides the network interface for FL clients to:
- Register with the server
- Receive global model parameters
- Submit local model updates
- Query server status
"""

import grpc
from concurrent import futures
import torch
import pickle
import logging
import time
import json
from typing import Optional

from server.fed_server import FederatedServer

logger = logging.getLogger(__name__)


# Since we're defining our own protocol, we'll use a simple
# Flask-based REST API as an alternative to gRPC for easier Docker deployment

from flask import Flask, request, jsonify
import io
import base64


def create_fl_server_app(server: FederatedServer) -> Flask:
    """
    Create Flask application for the FL server API.
    
    Endpoints:
    - POST /register: Register a new client
    - GET /global_model: Get current global model parameters
    - POST /submit_update: Submit a client model update
    - POST /aggregate: Trigger aggregation (admin)
    - GET /status: Get server status
    - GET /privacy_report: Get privacy report
    """
    app = Flask(__name__)

    # Store pending updates for current round
    pending_updates = []

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "round": server.current_round})

    @app.route("/register", methods=["POST"])
    def register_client():
        """Register a new FL client."""
        data = request.json
        client_id = data.get("client_id")
        data_size = data.get("data_size", 0)

        if not client_id:
            return jsonify({"error": "client_id required"}), 400

        result = server.register_client(client_id, data_size)

        # Serialize model parameters for transfer
        model_params = result.pop("global_model_params")
        buffer = io.BytesIO()
        torch.save(model_params, buffer)
        result["global_model_params_b64"] = base64.b64encode(buffer.getvalue()).decode()

        return jsonify(result)

    @app.route("/global_model", methods=["GET"])
    def get_global_model():
        """Get current global model parameters."""
        params = server.get_global_model_params()
        buffer = io.BytesIO()
        torch.save(params, buffer)

        return jsonify({
            "round": server.current_round,
            "model_params_b64": base64.b64encode(buffer.getvalue()).decode(),
            "is_complete": server.is_training_complete()
        })

    @app.route("/submit_update", methods=["POST"])
    def submit_update():
        """Submit a client model update."""
        data = request.json
        client_id = data.get("client_id")
        data_size = data.get("data_size", 0)
        metrics = data.get("metrics", {})
        update_b64 = data.get("model_update_b64")

        if not client_id or not update_b64:
            return jsonify({"error": "client_id and model_update_b64 required"}), 400

        # Deserialize model update
        buffer = io.BytesIO(base64.b64decode(update_b64))
        model_update = torch.load(buffer, map_location="cpu")

        # Receive the update
        result = server.receive_client_update(client_id, model_update, metrics, data_size)

        # Add to pending updates
        pending_updates.append((client_id, model_update, data_size))

        # Check if we have enough updates to aggregate
        if len(pending_updates) >= server.min_clients:
            logger.info(f"Triggering aggregation with {len(pending_updates)} updates")
            agg_result = server.aggregate_round(pending_updates.copy())
            pending_updates.clear()

            result["aggregation"] = {
                "status": "completed",
                "round": agg_result.get("round"),
                "privacy_cost": agg_result.get("privacy_cost")
            }

        return jsonify(result)

    @app.route("/aggregate", methods=["POST"])
    def force_aggregate():
        """Force aggregation with current pending updates (admin endpoint)."""
        if not pending_updates:
            return jsonify({"error": "No pending updates"}), 400

        agg_result = server.aggregate_round(pending_updates.copy())
        pending_updates.clear()

        # Serialize new global model
        params = server.get_global_model_params()
        buffer = io.BytesIO()
        torch.save(params, buffer)

        return jsonify({
            "status": "aggregated",
            "round": agg_result.get("round"),
            "privacy_cost": agg_result.get("privacy_cost"),
            "model_params_b64": base64.b64encode(buffer.getvalue()).decode()
        })

    @app.route("/status", methods=["GET"])
    def get_status():
        """Get server status."""
        return jsonify(server.get_server_status())

    @app.route("/privacy_report", methods=["GET"])
    def privacy_report():
        """Get differential privacy report."""
        if server.dp_engine:
            report = server.dp_engine.get_privacy_report()
            if server.moments_accountant:
                report["moments_accountant"] = server.moments_accountant.get_privacy_spent(
                    delta=server.dp_engine.budget.delta
                )
            return jsonify(report)
        return jsonify({"message": "Differential privacy not enabled"})

    @app.route("/checkpoint", methods=["POST"])
    def save_checkpoint():
        """Save a server checkpoint."""
        server.save_checkpoint()
        return jsonify({"status": "checkpoint_saved", "round": server.current_round})

    @app.route("/round_history", methods=["GET"])
    def round_history():
        """Get history of all rounds."""
        history = []
        for r in server.round_history:
            history.append({
                "round": r.round_number,
                "clients": r.participating_clients,
                "metrics": r.aggregated_metrics,
                "privacy_cost": r.privacy_cost,
                "duration": r.duration_seconds,
                "timestamp": r.timestamp.isoformat()
            })
        return jsonify({"rounds": history})

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    model_variant: str = "yolo11n.pt",
    num_classes: int = 80,
    num_rounds: int = 100,
    min_clients: int = 2,
    dp_enabled: bool = True,
    dp_epsilon: float = 1.0,
    dp_delta: float = 1e-5,
    dp_max_grad_norm: float = 1.0,
    dp_noise_multiplier: float = 1.0,
    aggregation_strategy: str = "fedavg",
    **kwargs
):
    """
    Run the federated learning server.
    
    Args:
        host: Server host address
        port: Server port
        model_variant: YOLOv11 model variant
        num_classes: Number of classes
        num_rounds: Total FL rounds
        min_clients: Minimum clients per round
        dp_enabled: Enable differential privacy
        dp_epsilon: Privacy budget
        dp_delta: Privacy delta
        dp_max_grad_norm: Max gradient norm for DP
        dp_noise_multiplier: Noise multiplier
        aggregation_strategy: Aggregation method
    """
    # Initialize FL server
    fl_server = FederatedServer(
        model_variant=model_variant,
        num_classes=num_classes,
        num_rounds=num_rounds,
        min_clients=min_clients,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
        dp_max_grad_norm=dp_max_grad_norm,
        dp_noise_multiplier=dp_noise_multiplier,
        aggregation_strategy=aggregation_strategy,
        **kwargs
    )

    # Create Flask app
    app = create_fl_server_app(fl_server)

    logger.info(f"Starting FL Server on {host}:{port}")
    logger.info(f"Configuration: rounds={num_rounds}, min_clients={min_clients}, DP={dp_enabled}")

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedX-PALM Federated Learning Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLOv11 variant")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients per round")
    parser.add_argument("--aggregation", default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--dp-enabled", action="store_true", default=True)
    parser.add_argument("--dp-epsilon", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_server(
        host=args.host,
        port=args.port,
        model_variant=args.model,
        num_classes=args.num_classes,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        dp_enabled=args.dp_enabled,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        aggregation_strategy=args.aggregation,
    )
