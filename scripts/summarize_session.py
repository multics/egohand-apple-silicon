"""Validate exported geometry and summarize a real CLI session."""

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in (args.session / "frames.jsonl").read_text().splitlines()
    ]
    if not records:
        raise RuntimeError("Session has no frames")
    for row in records:
        path = args.session / f"frame_{row['frame_id']:08d}.npz"
        if not row["hands"]:
            if path.exists():
                raise RuntimeError("Unexpected geometry for absent hands")
            continue
        with np.load(path) as arrays:
            for side in row["hands"]:
                for key, shape in (
                    ("pred_keypoints_3d", (25, 3)),
                    ("pred_vertices", (2859, 3)),
                    ("poses", (25, 3, 3)),
                    ("transl", (3,)),
                ):
                    array = arrays[f"{side}_{key}"]
                    if array.shape != shape or not np.isfinite(array).all():
                        raise RuntimeError(f"Invalid output: {path}/{side}/{key}")
    seconds = (
        records[-1]["read_monotonic_ns"] - records[0]["read_monotonic_ns"]
    ) / 1e9 + records[-1]["age_ms"] / 1000
    metadata = json.loads((args.session / "metadata.json").read_text())
    report = {
        "session": str(args.session),
        "frames": len(records),
        "profile_stages": metadata.get("profile_stages", True),
        "geometry_validated": True,
        "hand_combinations": dict(
            Counter("+".join(sorted(r["hands"])) or "none" for r in records)
        ),
        "processing_fps": len(records) / seconds,
        "elapsed_processing_seconds": seconds,
        "captured": records[-1]["captured"],
        "dropped": records[-1]["dropped"],
        "timing_ms": {},
        "notes": "Includes export; excludes model/detector load and warmup. Stage synchronization is controlled by profile_stages. Preview time is reflected in throughput, not age_ms. Camera read timestamps are not exposure timestamps.",
    }
    for key in ("detect_ms", "inference_ms", "export_ms", "age_ms"):
        values = [r[key] for r in records]
        report["timing_ms"][key] = dict(
            zip(("p50", "p95", "max"), map(float, np.percentile(values, [50, 95, 100])))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
