"""Validate real checkpoint inference on a calibrated image and export evidence."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from egohand_live.backend import load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Cannot decode input image")
    calibration = json.loads(args.calibration.read_text())
    if image.shape[:2] != (calibration["image_height"], calibration["image_width"]):
        raise ValueError("Image dimensions do not match the calibration")
    started = time.perf_counter()
    predictor, metadata = load_predictor(
        args.source, args.checkpoint, args.assets, args.device, args.batch_size
    )
    metadata["load_seconds"] = time.perf_counter() - started
    from inference.camera.camera_models import load_camera_calibration
    from inference.hand_detector_2d.rtdetr.onnx_model import (
        HandDetector,
        LABEL_ID_TO_HAND,
    )

    camera = load_camera_calibration(calibration)
    detector = HandDetector(
        args.detector,
        providers=["CPUExecutionProvider"],
        top1_per_class=True,
        pad=0.15,
        square=True,
        clip=False,
    )
    started = time.perf_counter()
    detections = detector.detect(
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    )
    metadata["detector_ms"] = (time.perf_counter() - started) * 1000
    boxes = {
        LABEL_ID_TO_HAND[d.label_id]: list(d.cxcys)
        for d in detections
        if d.label_id in LABEL_ID_TO_HAND
    }
    if not boxes:
        raise RuntimeError("No hands detected; model inference has not been validated")
    print("Detected boxes:", boxes, flush=True)
    network_devices = {
        str(p.device)
        for name, p in predictor.model.named_parameters()
        if not name.startswith("soma_hand.")
    }
    rig_devices = {str(p.device) for p in predictor.model.soma_hand.parameters()}
    if network_devices != {args.device + (":0" if args.device == "mps" else "")}:
        raise RuntimeError(f"Unexpected network placement: {network_devices}")
    if rig_devices and rig_devices != {"cpu"}:
        raise RuntimeError(f"Unexpected rig placement: {rig_devices}")
    started = time.perf_counter()
    results = predictor.run(
        [image],
        ["0"],
        {"0": boxes},
        ["left", "right"],
        intrinsics_per_image={"0": camera},
    )
    if args.device == "mps":
        torch.mps.synchronize()
    metadata.update(
        {
            "inference_ms": (time.perf_counter() - started) * 1000,
            "network_stages_ms": predictor.stage_timings_ms,
            "soma_forward_ms": predictor.model.soma_hand.timings_ms,
            "network_parameter_devices": sorted(network_devices),
            "rig_parameter_devices": sorted(rig_devices),
            "torch": torch.__version__,
            "source_revision": (args.source / "UPSTREAM_REVISION").read_text().strip(),
            "image": str(args.image),
            "calibration": calibration,
            "boxes": boxes,
            "outputs": {},
            "real_checkpoint_inference": True,
            "live_performance_verified": False,
        }
    )
    arrays = {}
    for side in boxes:
        result = results[side]["0"]
        shapes = {}
        for key in (
            "pred_keypoints_3d",
            "pred_vertices",
            "poses",
            "transl",
            "identity_coeffs",
            "global_scale",
            "scale_params",
        ):
            if result.get(key) is None:
                continue
            value = np.asarray(result[key])
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite output: {side}/{key}")
            arrays[f"{side}_{key}"] = value
            shapes[key] = list(value.shape)
        if result["pred_keypoints_3d"].shape != (25, 3):
            raise RuntimeError("Wrong joint shape")
        rotations = result["poses"]
        if rotations.shape != (25, 3, 3):
            raise RuntimeError("Wrong pose shape")
        rotation_error = float(
            np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3)).max()
        )
        determinant_error = float(np.abs(np.linalg.det(rotations) - 1).max())
        if max(rotation_error, determinant_error) > 1e-3:
            raise RuntimeError("Invalid rotation matrices")
        metadata["outputs"][side] = {
            "shapes": shapes,
            "all_finite": True,
            "rotation_orthogonality_max_error": rotation_error,
            "rotation_determinant_max_error": determinant_error,
            "minimum_camera_depth_m": float(
                (result["pred_keypoints_3d"] + result["transl"])[:, 2].min()
            ),
        }
        arrays[f"{side}_faces"] = predictor.model.soma_hand.faces(side)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    (args.output / "verification.json").write_text(json.dumps(metadata, indent=2))
    from egohand_live.visualize import Preview

    hands = {side: results[side]["0"] for side in boxes}
    faces = {side: predictor.model.soma_hand.faces(side) for side in boxes}
    view = Preview(show_window=False).render(
        image,
        hands,
        faces,
        camera,
        f"{args.device.upper()} real checkpoint | {metadata['inference_ms']:.0f} ms | first frame",
    )
    if not cv2.imwrite(str(args.output / "preview.png"), view):
        raise RuntimeError("Cannot save verification preview")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
