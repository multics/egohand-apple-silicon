"""Run calibrated video or a live camera through the official EgoHand model."""

import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image
from egohand_live.backend import load_predictor
from egohand_live.capture import Frame, LatestCapture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", required=True, type=Path, help="Pinned EgoPost source directory"
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--detector", required=True, type=Path)
    calibration_args = p.add_mutually_exclusive_group(required=True)
    calibration_args.add_argument("--calibration", type=Path)
    calibration_args.add_argument(
        "--estimate-intrinsics",
        action="store_true",
        help="Use approximate pinhole intrinsics; metric depth is unverified",
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", type=Path)
    p.add_argument("--output", type=Path, required=True, help="New session directory")
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--detector-device", choices=["coreml", "cpu"], default="coreml")
    p.add_argument("--mhr-device", choices=["mps", "cpu"], default="mps")
    p.add_argument(
        "--profile-stages",
        action="store_true",
        help="Synchronize MPS stages for diagnostics; can reduce throughput",
    )
    p.add_argument("--batch-size", type=int, choices=[1, 2], default=2)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")
    predictor, metadata = load_predictor(
        args.source,
        args.checkpoint,
        args.assets,
        args.device,
        args.batch_size,
        profile_stages=args.profile_stages,
        mhr_device=args.mhr_device,
    )
    from inference.hand_detector_2d.rtdetr.onnx_model import LABEL_ID_TO_HAND
    from egohand_live.detector import load_detector, select_distinct_detections

    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    # Use the upstream camera parser to preserve projection conventions.
    from inference.camera.camera_models import load_camera_calibration

    camera = load_camera_calibration(calibration) if calibration else None
    detector, detector_metadata = load_detector(args.detector, args.detector_device)
    metadata.update(detector_metadata)
    metadata.update(
        {
            "calibration": calibration,
            "calibration_status": "estimated"
            if args.estimate_intrinsics
            else "supplied",
            "metric_depth_verified": False,
            "detector_providers": detector.providers,
            "source_revision": (args.source / "UPSTREAM_REVISION").read_text().strip(),
            "input": str(args.video) if args.video else f"camera:{args.camera}",
            "timestamp_note": "Camera timestamps mark read completion, not sensor exposure",
            "torch": torch.__version__,
            "geometry_units": "meters",
            "camera_axes": "x right, y down, z forward",
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    faces = {side: predictor.model.soma_hand.faces(side) for side in ("left", "right")}
    np.savez_compressed(args.output / "topology.npz", **faces)
    preview = None
    if not args.no_preview:
        from egohand_live.visualize import Preview

        preview = Preview()
    cap = (
        cv2.VideoCapture(str(args.video))
        if args.video
        else cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    )
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open video input")
    reader = None
    count = 0
    start = time.monotonic()
    try:
        if not args.video:
            reader = LatestCapture(cap).start()
        with (args.output / "frames.jsonl").open("w") as journal:
            while args.max_frames is None or count < args.max_frames:
                if reader:
                    frame = reader.next()
                else:
                    ok, image = cap.read()
                    if not ok:
                        break
                    frame = Frame(count, time.monotonic_ns(), time.time_ns(), image)
                height, width = frame.image.shape[:2]
                if calibration is None:
                    focal = float(max(width, height))
                    calibration = {
                        "model": "PINHOLE",
                        "image_width": width,
                        "image_height": height,
                        "camera_matrix": {
                            "matrix_3x3": [
                                [focal, 0.0, width / 2],
                                [0.0, focal, height / 2],
                                [0.0, 0.0, 1.0],
                            ]
                        },
                        "distortion_coefficients": [],
                    }
                    camera = load_camera_calibration(calibration)
                    metadata["calibration"] = calibration
                    metadata["intrinsics_estimate_rule"] = (
                        "fx=fy=max(width,height), centered principal point, zero distortion"
                    )
                    (args.output / "metadata.json").write_text(
                        json.dumps(metadata, indent=2)
                    )
                    print(
                        "UNCALIBRATED preview: estimated intrinsics; metric depth is unverified",
                        flush=True,
                    )
                if (width, height) != (
                    calibration["image_width"],
                    calibration["image_height"],
                ):
                    raise ValueError(
                        "Frame size differs from calibration; calibrate the selected camera mode"
                    )
                tick = time.perf_counter()
                detections = detector.detect(
                    Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
                )
                accepted_detections, rejected_detections = select_distinct_detections(
                    detections
                )
                boxes = {
                    LABEL_ID_TO_HAND[d.label_id]: list(d.cxcys)
                    for d in accepted_detections
                    if d.label_id in LABEL_ID_TO_HAND
                }
                detect_ms = (time.perf_counter() - tick) * 1000
                image_id = str(frame.index)
                tick = time.perf_counter()
                predictor.model.soma_hand.timings_ms.clear()
                predictor.model.soma_hand.mhr_timings_ms.clear()
                for values in predictor.stage_timings_ms.values():
                    values.clear()
                results = predictor.run(
                    [frame.image],
                    [image_id],
                    {image_id: boxes},
                    ["left", "right"],
                    intrinsics_per_image={image_id: camera},
                )
                if args.device == "mps":
                    torch.mps.synchronize()
                infer_ms = (time.perf_counter() - tick) * 1000
                hands = {
                    side: values[image_id]
                    for side, values in results.items()
                    if image_id in values
                }
                arrays = {}
                for side, result in hands.items():
                    for key in (
                        "poses",
                        "identity_coeffs",
                        "scale_params",
                        "global_scale",
                        "pred_keypoints_3d",
                        "pred_vertices",
                        "transl",
                    ):
                        if result.get(key) is not None:
                            value = np.asarray(result[key])
                            if not np.isfinite(value).all():
                                raise RuntimeError(
                                    f"Non-finite model output: {side}/{key}"
                                )
                            arrays[f"{side}_{key}"] = value
                tick = time.perf_counter()
                if arrays:
                    np.savez_compressed(
                        args.output / f"frame_{frame.index:08d}.npz", **arrays
                    )
                export_ms = (time.perf_counter() - tick) * 1000
                record = {
                    "frame_id": frame.index,
                    "read_unix_ns": frame.unix_ns,
                    "read_monotonic_ns": frame.monotonic_ns,
                    "video_time_ms": cap.get(cv2.CAP_PROP_POS_MSEC)
                    if args.video
                    else None,
                    "hands": list(hands),
                    "boxes": boxes,
                    "detections": [d.as_dict() for d in detections],
                    "rejected_duplicate_detections": [
                        d.as_dict() for d in rejected_detections
                    ],
                    "detect_ms": detect_ms,
                    "inference_ms": infer_ms,
                    "export_ms": export_ms,
                    "soma_forward_ms": predictor.model.soma_hand.timings_ms.copy(),
                    "mhr_ms": predictor.model.soma_hand.mhr_timings_ms.copy(),
                    "network_stages_ms": predictor.stage_timings_ms,
                    "age_ms": (time.monotonic_ns() - frame.monotonic_ns) / 1e6,
                    "dropped": reader.dropped if reader else 0,
                    "captured": reader.count if reader else count + 1,
                }
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                count += 1
                elapsed = time.monotonic() - start
                inference_label = (
                    f"Inference: {infer_ms:.1f} ms ({len(hands)} hand(s))"
                    if hands
                    else "Inference: idle (no hands)"
                )
                label = (
                    f"{inference_label} | Detection: {detect_ms:.1f} ms | "
                    f"Processed: {count / elapsed:.1f} FPS\n"
                    f"Frame age: {record['age_ms']:.0f} ms | "
                    f"Capture: {record['captured'] / elapsed:.1f} FPS | "
                    f"Dropped: {record['dropped']}"
                )
                if args.estimate_intrinsics:
                    label += " | UNCALIBRATED"
                print(label, flush=True)
                if preview and not preview.show(
                    frame.image, hands, faces, camera, label
                ):
                    break
    finally:
        if reader:
            reader.close()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
