"""Keep neural inference on MPS and isolate SOMA/MHR on CPU."""

from pathlib import Path
import os
import sys
import time


def map_tensors(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: map_tensors(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(map_tensors(v, device) for v in value)
    if isinstance(value, list):
        return [map_tensors(v, device) for v in value]
    return value


def solve_camera_on_cpu(joints_3d, *args, **kwargs):
    from egohand.utils.fixed_depth_translation import (
        solve_lateral_translation_full_perspective,
    )

    result = solve_lateral_translation_full_perspective(
        joints_3d.cpu(), *map_tensors(args, "cpu"), **map_tensors(kwargs, "cpu")
    )
    return type(result)(*(map_tensors(value, joints_3d.device) for value in result))


def load_predictor(
    source,
    checkpoint,
    assets,
    device="mps",
    batch_size=2,
    *,
    profile_stages=True,
    mhr_device="cpu",
):
    import torch
    from omegaconf import OmegaConf

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; the network will not silently use CPU")
    if (device == "mps" or mhr_device == "mps") and os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "1":
        raise RuntimeError(
            "Unset PYTORCH_ENABLE_MPS_FALLBACK to verify network execution on MPS"
        )
    source = Path(source).resolve()
    revision = "a74babb02cd8a018749f1f2374b841ea04650f1b"
    if (source / "UPSTREAM_REVISION").read_text().strip() != revision:
        raise RuntimeError("Unsupported EgoPost revision; fetch the pinned source")
    if not (source / "egohand/models/egohand.py").is_file():
        raise FileNotFoundError(f"Not an EgoPost source directory: {source}")
    sys.path.insert(0, str(source))
    import egohand.models.soma_hand_wrapper as wrapper_module
    import egohand.models.egohand as model_module
    from inference.pipeline import predictor as predictor_module

    # MPS lacks eigvalsh/solve used by the small camera translation system.
    model_module.solve_lateral_translation_full_perspective = solve_camera_on_cpu
    original_wrapper = wrapper_module.SomaHandWrapper

    class CPUSomaHand(original_wrapper):
        def __init__(
            self, data_root=None, device=None, mode=None, identity_model_type="mhr"
        ):
            super().__init__(
                data_root=data_root,
                device="cpu",
                mode="dense",
                identity_model_type=identity_model_type,
            )
            self.timings_ms = []

        def _apply(self, fn, recurse=True):
            # Parent model.to(mps) must leave the complete rig on CPU.
            return self

        def forward(self, *args, **kwargs):
            tensor = kwargs.get("poses", args[0] if args else None)
            target = tensor.device
            start = time.perf_counter()
            out = super().forward(
                *map_tensors(args, "cpu"), **map_tensors(kwargs, "cpu")
            )
            self.timings_ms.append((time.perf_counter() - start) * 1000)
            return map_tensors(out, target)

    # The constructor imported the class by value; change only this model symbol.
    model_module.SomaHandWrapper = CPUSomaHand
    mirror = predictor_module.mirror_left_prediction_to_right
    # Repeated loading must not wrap the wrapper from an earlier instance.
    mirror = getattr(mirror, "_cpu_original", mirror)

    def mirror_on_cpu(model, poses, *args, **kwargs):
        target = poses.device
        return map_tensors(
            mirror(
                model,
                poses.cpu(),
                *map_tensors(args, "cpu"),
                **map_tensors(kwargs, "cpu"),
            ),
            target,
        )

    mirror_on_cpu._cpu_original = mirror
    predictor_module.mirror_left_prediction_to_right = mirror_on_cpu
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])
    if not str(cfg.get("_target_", "")).endswith("egohand.EgoHand"):
        raise ValueError("Unsupported checkpoint target")
    cfg.backbone_weights = None
    cfg.soma_hand.data_root = str(Path(assets).resolve())
    model = model_module.EgoHand(cfg)
    model.validate_camera_depth_checkpoint(ckpt)
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    # SOMA rig data is not part of the trained network checkpoint.
    missing_network = [
        k for k in incompatible.missing_keys if not k.startswith("soma_hand.")
    ]
    if missing_network or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing_network}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del ckpt
    from egohand_live.mhr import MHRRunner, prepare_mps_mhr

    model.soma_hand.mhr_timings_ms = []
    mhr_path = None
    if mhr_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS MHR requested but unavailable")
        mhr_path = prepare_mps_mhr(Path(assets) / "MHR/mhr_model_lod1.pt")
    for side in ("left", "right"):
        wrapper = getattr(model.soma_hand, side).identity_model._mhr_model
        mhr_model = (
            torch.jit.load(str(mhr_path), map_location="mps")
            if mhr_path
            else wrapper.module
        )
        parameter_devices = {
            parameter.device.type for parameter in mhr_model.parameters()
        }
        buffer_devices = {buffer.device.type for buffer in mhr_model.buffers()}
        if parameter_devices != {mhr_device} or buffer_devices != {mhr_device}:
            raise RuntimeError("MHR parameters/buffers are not on the requested device")
        wrapper.module = MHRRunner(
            mhr_model, mhr_device, model.soma_hand.mhr_timings_ms
        )
    predictor = predictor_module.EgoHandPredictor(
        model.eval(),
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
        smooth=False,
        use_perspective_cropping=True,
    )
    predictor.stage_timings_ms = {"backbone": [], "decoder": []}

    def add_timing(name, module):
        started = []

        def sync():
            if device == "mps":
                torch.mps.synchronize()

        def before(module, inputs):
            sync()
            started.append(time.perf_counter())

        def after(module, inputs, output):
            sync()
            predictor.stage_timings_ms[name].append(
                (time.perf_counter() - started.pop()) * 1000
            )

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    if profile_stages:
        add_timing("backbone", model.backbone)
        add_timing("decoder", model.hand_head)
    return predictor, {
        "network_device": device,
        "soma_device": "cpu",
        "mhr_device": mhr_device,
        "mhr_derived_model": str(mhr_path) if mhr_path else None,
        "camera_solver_device": "cpu",
        "dtype": "float32",
        "profile_stages": profile_stages,
        "mps_options": {
            "prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL", "0"),
            "fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "0"),
            "automatic_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        },
        "missing_rig_keys": incompatible.missing_keys,
    }
