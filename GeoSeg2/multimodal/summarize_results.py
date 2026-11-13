"""Create the final paper-target comparison after both height stages finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _final_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if "final_test" in payload:
        return payload["final_test"]
    if isinstance(payload.get("public"), dict) and "final_test" in payload["public"]:
        return payload["public"]["final_test"]
    raise KeyError("metrics file does not contain final_test")


def write_comparison(protocol: dict[str, Any]) -> None:
    configuration = protocol.get("comparison")
    if not isinstance(configuration, dict):
        return
    output_dir = Path(configuration["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [
        {
            "method": "RoGLSNet paper single-modal",
            "oa": 0.9328,
            "mf1": 0.9131,
            "miou": 0.8431,
            "tta": "paper",
            "dsm_effective": "n/a",
        }
    ]
    for source in configuration["sources"]:
        metrics_path = Path(source["path"]).resolve()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        final = _final_metrics(payload)
        diagnostics = payload.get("auxiliary_diagnostics", {})
        diagnostics_path = str(source.get("diagnostics_path", "")).strip()
        if diagnostics_path:
            diagnostics = json.loads(
                Path(diagnostics_path).resolve().read_text(encoding="utf-8")
            )
        rows.append(
            {
                "method": source["name"],
                "oa": float(final["oa"]),
                "mf1": float(final["mf1"]),
                "miou": float(final["miou"]),
                "tta": final.get("tta", "unknown"),
                "dsm_effective": diagnostics.get("effective_at_0.002", "not_measured"),
                "dsm_mf1_margin": diagnostics.get(
                    "mf1_margin_over_strongest_control", "not_measured"
                ),
                "metrics_path": str(metrics_path),
            }
        )
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Vaihingen 最终对比",
        "",
        "| Method | OA | mF1 | mIoU | DSM effective | DSM mF1 margin |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        margin = row.get("dsm_mf1_margin", "n/a")
        margin_text = f"{float(margin) * 100:.3f} pp" if isinstance(margin, (int, float)) else str(margin)
        markdown.append(
            f"| {row['method']} | {float(row['oa']) * 100:.2f} | "
            f"{float(row['mf1']) * 100:.2f} | {float(row['miou']) * 100:.2f} | "
            f"{row['dsm_effective']} | {margin_text} |"
        )
    markdown.extend(("", "mF1/mIoU 为前五类平均，OA 包含所有有效类别。", ""))
    (output_dir / "comparison.md").write_text("\n".join(markdown), encoding="utf-8")
