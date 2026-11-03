"""Measure CPU/Core ML detector execution and actual graph assignments."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from egohand_live.detector import prepare_coreml_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("--iterations must be at least 3")
    sys.path.insert(0, str(args.source.resolve()))
    from inference.hand_detector_2d.rtdetr.onnx_model import HandDetector

    image = Image.open(args.image).convert("RGB")
    feed = {
        "images": HandDetector._preprocess(image),
        "orig_target_sizes": np.array([[image.width, image.height]], dtype=np.int64),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    derived = prepare_coreml_model(args.detector)
    reports = []
    reference = None
    for name, path, providers in (
        ("cpu_original", args.detector, ["CPUExecutionProvider"]),
        (
            "coreml_batch1",
            derived,
            [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "1",
                        "ModelCacheDirectory": str((args.output / "cache").resolve()),
                    },
                ),
                "CPUExecutionProvider",
            ],
        ),
    ):
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output / name)
        started = time.perf_counter()
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        load_seconds = time.perf_counter() - started
        timings = []
        for _ in range(args.iterations + 2):
            started = time.perf_counter()
            result = session.run(["labels", "boxes", "scores"], feed)
            timings.append((time.perf_counter() - started) * 1000)
        if reference is None:
            reference = [array.copy() for array in result]
        events = json.loads(Path(session.end_profiling()).read_text())
        assignments = {
            event["name"]: event["args"]["provider"]
            for event in events
            if event.get("args", {}).get("provider")
        }
        differences = [
            float(np.abs(a.astype(float) - b).max()) for a, b in zip(result, reference)
        ]
        valid = all(np.isfinite(array).all() for array in result)
        if (
            not valid
            or differences[0] != 0
            or differences[1] > 1.0
            or differences[2] > 0.01
        ):
            raise RuntimeError(f"Detector output parity check failed: {differences}")
        report = {
            "name": name,
            "load_seconds": load_seconds,
            "warm_median_ms": float(np.median(timings[2:])),
            "warm_p95_ms": float(np.percentile(timings[2:], 95)),
            "first_ms": timings[0],
            "provider_node_counts": dict(Counter(assignments.values())),
            "max_abs_errors_labels_boxes_scores": differences,
            "parity_passed": True,
            "notes": "Raw session execution excludes image preprocessing. Core ML graph assignment does not prove ANE placement. Profile events add overhead.",
        }
        reports.append(report)
        (args.output / "summary.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
