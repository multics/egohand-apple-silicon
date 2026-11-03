"""Prepare the pinned detector for Core ML with fixed batch-one shapes."""

import hashlib
import json
from pathlib import Path
import time

SOURCE_SHA256 = "85d36428cc1ecf8832ca2d48f7aa04df0659cf94dde358352fb3cf651a319810"


def select_distinct_detections(detections):
    """Keep the strongest label for an identical box, without overlap NMS.

    One RT-DETR proposal can pass the score threshold for both hand classes.
    Two distinct hands can overlap, so suppress only exactly equal boxes.
    This does not validate anatomical handedness.
    """
    accepted, rejected = [], []
    seen_boxes, seen_labels = set(), set()
    for detection in sorted(detections, key=lambda item: -item.score):
        if detection.label_id not in (1, 2):
            continue
        box = tuple(detection.xyxy)
        if box in seen_boxes or detection.label_id in seen_labels:
            rejected.append(detection)
            continue
        accepted.append(detection)
        seen_boxes.add(box)
        seen_labels.add(detection.label_id)
    return accepted, rejected


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_coreml_model(source):
    import onnx
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    source = Path(source)
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Core ML preparation requires the pinned official detector")
    target = source.with_name(source.stem + "_batch1.onnx")
    manifest = target.with_suffix(".json")
    if target.exists() and manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached.get("source_sha256") == SOURCE_SHA256 and cached.get(
            "derived_sha256"
        ) == sha256(target):
            return target
    model = onnx.load(str(source))
    for value in model.graph.input:
        value.type.tensor_type.shape.dim[0].dim_value = 1
    model = SymbolicShapeInference.infer_shapes(
        model, auto_merge=True, guess_output_rank=True
    )
    onnx.checker.check_model(model)
    temporary = target.with_suffix(".onnx.tmp")
    onnx.save(model, str(temporary))
    temporary.replace(target)
    manifest.write_text(
        json.dumps(
            {
                "source_sha256": SOURCE_SHA256,
                "derived_sha256": sha256(target),
                "transformation": "fixed input batch=1 and symbolic shape inference; weights unchanged",
                "onnx": onnx.__version__,
            },
            indent=2,
        )
    )
    return target


def load_detector(weights, device="coreml"):
    import onnxruntime as ort
    from PIL import Image
    from inference.hand_detector_2d.rtdetr.onnx_model import HandDetector

    providers = ["CPUExecutionProvider"]
    model_path = Path(weights)
    if device == "coreml":
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "Core ML is unavailable; select --detector-device cpu explicitly"
            )
        model_path = prepare_coreml_model(model_path)
        cache = model_path.parent / "coreml-cache"
        cache.mkdir(exist_ok=True)
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "ModelCacheDirectory": str(cache.resolve()),
                },
            ),
            "CPUExecutionProvider",
        ]
    started = time.perf_counter()
    detector = HandDetector(
        model_path,
        providers=providers,
        top1_per_class=True,
        pad=0.15,
        square=True,
        clip=False,
    )
    # Compile before opening the camera so startup does not age a captured frame.
    detector.detect(Image.new("RGB", (640, 640)))
    return detector, {
        "detector_device": device,
        "detector_model": str(model_path),
        "detector_providers": detector.providers,
        "detector_load_and_warmup_seconds": time.perf_counter() - started,
        "detector_compute_units": "ALL" if device == "coreml" else "CPUOnly",
        "detector_ane_execution_verified": False,
    }
