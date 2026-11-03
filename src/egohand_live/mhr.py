"""Run the pinned MHR shape model on MPS without changing learned weights."""

import json
from pathlib import Path
import time
import zipfile

import torch

from egohand_live.detector import sha256

MHR_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"


def prepare_mps_mhr(source):
    source = Path(source)
    if sha256(source) != MHR_SHA256:
        raise ValueError("MPS MHR preparation requires the pinned official asset")
    target = source.with_name(source.stem + "_mps.pt")
    manifest = target.with_suffix(".json")
    if target.exists() and manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached.get("source_sha256") == MHR_SHA256 and cached.get(
            "derived_sha256"
        ) == sha256(target):
            return target
    temporary = target.with_suffix(".source.tmp.pt")
    replacements = 0
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(temporary, "w") as derived,
    ):
        for info in original.infolist():
            data = original.read(info.filename)
            if info.filename.endswith(
                "code/__torch__/pymomentum/backend/skel_state_backend.py"
            ):
                before = b"_2 = _0(local_skel_state, prefix_mul_indices, True, True, )"
                after = b"_2 = _0(local_skel_state, prefix_mul_indices, True, False, )"
                if data.count(before) != 1:
                    raise RuntimeError("Unexpected MHR skeleton code")
                data = data.replace(before, after)
                replacements += 1
            derived.writestr(info, data)
    if replacements != 1:
        raise RuntimeError("Missing MHR float64 skeleton call")
    model = torch.jit.load(str(temporary), map_location="cpu")
    converted = 0
    for name, module in model.named_modules():
        if not hasattr(module, "sparse_shape"):
            continue
        if name != "pose_correctives_model.pose_dirs_predictor.0" or tuple(
            module.sparse_shape
        ) != (3000, 750):
            raise RuntimeError("Unexpected MHR sparse layer")
        dense = torch.sparse_coo_tensor(
            module.sparse_indices, module.sparse_weight, module.sparse_shape
        ).to_dense()
        graph = module.forward.graph
        nodes = [
            node for node in graph.nodes() if node.kind() == "aten::sparse_coo_tensor"
        ]
        if len(nodes) != 1:
            raise RuntimeError("Unexpected MHR sparse graph")
        node = nodes[0]
        value = graph.insertConstant(dense)
        value.node().moveBefore(node)
        node.output().replaceAllUsesWith(value)
        node.destroy()
        torch._C._jit_pass_dce(graph)
        graph.lint()
        converted += 1
    if converted != 1:
        raise RuntimeError("Missing MHR sparse layer")
    saved = target.with_suffix(".save.tmp.pt")
    torch.jit.save(model, str(saved))
    saved.replace(target)
    temporary.unlink()
    manifest.write_text(
        json.dumps(
            {
                "source_sha256": MHR_SHA256,
                "derived_sha256": sha256(target),
                "torch": torch.__version__,
                "changes": [
                    "skeleton accumulation float64 to float32",
                    "constant sparse linear matrix expanded to dense; values unchanged",
                ],
            },
            indent=2,
        )
    )
    return target


class MHRRunner(torch.nn.Module):
    """Explicit MHR device boundary; the surrounding rig receives CPU tensors."""

    def __init__(self, model, device, timings):
        super().__init__()
        self.model = model.eval()
        self.device = device
        self.timings = timings

    def forward(self, *inputs):
        started = time.perf_counter()
        outputs = self.model(*(value.to(self.device) for value in inputs))
        outputs = tuple(value.cpu() for value in outputs)
        self.timings.append((time.perf_counter() - started) * 1000)
        return outputs
