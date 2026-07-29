#!/usr/bin/env python3
"""Aggregate every results/*.json produced by scripts 05-09 into flat CSV
summary tables, one per experiment block -- these map directly onto thesis
Tables 4.1 (B1), 4.2 (B2), 4.3-4.4 (E1), 4.5 (E2), 4.9 (XAI)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

RESULTS_DIR = Path("results")


def _load(pattern):
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob(pattern))]


def export_b1():
    p = RESULTS_DIR / "b1_centralized.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    df = pd.DataFrame([{"map50": r["metrics"]["map50"], "map50_95": r["metrics"]["map50_95"],
                         "precision": r["metrics"]["precision"], "recall": r["metrics"]["recall"]}])
    df.to_csv(RESULTS_DIR / "table_4_1_b1_centralized.csv", index=False)
    return df


def export_b2():
    rows = [{"k": r["k"], "rounds": r["rounds"], "map50": r["metrics"]["map50"],
             "map50_95": r["metrics"]["map50_95"], "precision": r["metrics"]["precision"],
             "recall": r["metrics"]["recall"]} for r in _load("b2_k*.json")]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("k")
    df.to_csv(RESULTS_DIR / "table_4_2_b2_federated.csv", index=False)
    return df


def export_dp(tag, out_name):
    rows = [{"k": r["k"], "sigma": r["sigma"], "epsilon": r["epsilon_max_over_clients"],
             "map50": r["metrics"]["map50"], "map50_95": r["metrics"]["map50_95"]}
            for r in _load(f"{tag}_k*_sigma*.json")]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["k", "sigma"])
    df.to_csv(RESULTS_DIR / out_name, index=False)
    return df


def export_xai():
    frames = []
    for p in sorted(RESULTS_DIR.glob("xai_*.json")):
        tag = p.stem.replace("xai_", "")
        data = json.loads(p.read_text())
        for cls_name, m in data.items():
            frames.append({"run": tag, "class": cls_name, "n": m["n"],
                            "average_drop": m["average_drop"], "focus_retention_rate": m["focus_retention_rate"]})
    if not frames:
        return None
    df = pd.DataFrame(frames)
    df.to_csv(RESULTS_DIR / "table_4_9_xai_faithfulness.csv", index=False)
    return df


if __name__ == "__main__":
    for name, fn in [("B1 centralized", export_b1), ("B2 federated", export_b2),
                      ("E1 DP full", lambda: export_dp("e1_dp_full", "table_4_3_e1_dp_full.csv")),
                      ("E2 DP partial", lambda: export_dp("e2_dp_partial", "table_4_5_e2_dp_partial.csv")),
                      ("XAI faithfulness", export_xai)]:
        df = fn()
        print(f"\n=== {name} ===")
        print(df.to_string(index=False) if df is not None else "(no results/*.json found yet -- run the matching script first)")
    print(f"\nCSV tables written to {RESULTS_DIR}/")
