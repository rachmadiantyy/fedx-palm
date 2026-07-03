"""FedX-Palm v2: YOLOv11 federated learning + differential privacy + XAI for oil palm FFB ripeness detection."""
from fedxpalm.models.groupnorm import patch_fuse_for_groupnorm as _patch_fuse_for_groupnorm

_patch_fuse_for_groupnorm()
