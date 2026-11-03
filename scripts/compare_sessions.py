"""Compare matched replay outputs and timings for one optimization round."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    args = parser.parse_args()
    sessions = [args.baseline, args.candidate]
    records = [
        [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
        for path in sessions
    ]
    if not records[0] or len(records[0]) != len(records[1]):
        raise RuntimeError("Matched replays must have the same nonzero frame count")
    max_geometry = 0.0
    max_rotation_error = 0.0
    for old, new in zip(*records, strict=True):
        if old["frame_id"] != new["frame_id"] or old["hands"] != new["hands"]:
            raise RuntimeError("Frame or handedness mismatch")
        if not old["hands"]:
            continue
        filename = f"frame_{old['frame_id']:08d}.npz"
        with (
            np.load(args.baseline / filename) as a,
            np.load(args.candidate / filename) as b,
        ):
            if set(a.files) != set(b.files):
                raise RuntimeError("Output keys differ")
            for key in a.files:
                if a[key].shape != b[key].shape or not np.isfinite(b[key]).all():
                    raise RuntimeError(f"Invalid output: {filename}/{key}")
                if any(
                    name in key
                    for name in ("pred_vertices", "pred_keypoints_3d", "transl")
                ):
                    max_geometry = max(
                        max_geometry,
                        float(np.linalg.norm(a[key] - b[key], axis=-1).max()),
                    )
                if key.endswith("_poses"):
                    rotations = b[key]
                    max_rotation_error = max(
                        max_rotation_error,
                        float(
                            np.abs(
                                rotations @ rotations.transpose(0, 2, 1) - np.eye(3)
                            ).max()
                        ),
                        float(np.abs(np.linalg.det(rotations) - 1).max()),
                    )
            for side in new["hands"]:
                old_camera = a[f"{side}_pred_keypoints_3d"] + a[f"{side}_transl"]
                new_camera = b[f"{side}_pred_keypoints_3d"] + b[f"{side}_transl"]
                max_geometry = max(
                    max_geometry,
                    float(np.linalg.norm(new_camera - old_camera, axis=-1).max()),
                )
    if max_geometry > args.tolerance_mm / 1000 or max_rotation_error > 1e-3:
        raise RuntimeError(
            f"Geometry/rotation tolerance exceeded: {max_geometry}, {max_rotation_error}"
        )
    report = {
        "frames": len(records[0]),
        "geometry_passed": True,
        "max_geometry_distance_mm": max_geometry * 1000,
        "max_rotation_error": max_rotation_error,
        "tolerance_mm": args.tolerance_mm,
        "sessions": [],
    }
    for path, rows in zip(sessions, records):
        elapsed = (
            rows[-1]["read_monotonic_ns"] - rows[0]["read_monotonic_ns"]
        ) / 1e9 + rows[-1]["age_ms"] / 1000
        stats = {
            "path": str(path),
            "processing_fps": len(rows) / elapsed,
            "by_hand_count": {},
        }
        for count in (0, 1, 2):
            group = [r for r in rows if len(r["hands"]) == count]
            if not group:
                continue
            entry = {"frames": len(group)}
            for key in (
                "detect_ms",
                "inference_ms",
                "soma_forward_ms",
                "mhr_ms",
                "export_ms",
                "age_ms",
            ):
                values = [
                    sum(r[key]) if isinstance(r.get(key), list) else r[key]
                    for r in group
                    if key in r
                ]
                if values:
                    entry[key] = dict(
                        zip(
                            ("p50", "p95", "max"),
                            map(float, np.percentile(values, [50, 95, 100])),
                        )
                    )
            stats["by_hand_count"][str(count)] = entry
        report["sessions"].append(stats)
    report["notes"] = (
        "Matched sequential replay. Includes cold shapes and export; no frames excluded. Median stage values are not additive. Geometry metric includes wrist-relative mesh/joints, translation and camera-space joints."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
