"""Report runtime readiness. This does not validate model inference."""

import importlib
import json
import platform

report = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "model_inference_verified": False,
}
for name in ["torch", "cv2", "soma", "dinov3", "onnxruntime", "omegaconf"]:
    try:
        module = importlib.import_module(name)
        report[name] = {
            "version": getattr(module, "__version__", "unknown"),
            "import": True,
        }
        if name == "torch":
            report["mps_available"] = module.backends.mps.is_available()
            if report["mps_available"]:
                tensor = module.eye(3, device="mps")
                report["mps_tensor_check"] = bool(
                    module.isfinite(tensor @ tensor).all().item()
                )
    except Exception as exc:
        report[name] = {"import": False, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(report, indent=2))
