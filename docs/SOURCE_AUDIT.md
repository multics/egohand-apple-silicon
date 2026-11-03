# Source audit

Audit date: 2026-10-28. This is source evidence, not proof of MPS inference.

## Pinned sources
- Isaac Capture: `a74babb02cd8a018749f1f2374b841ea04650f1b`
  https://github.com/NVIDIA/IsaacCapture/tree/a74babb02cd8a018749f1f2374b841ea04650f1b/src/python/isaaccapture_egopost
- DINOv3: `5aa7e93ab42ae3526291f6b4e3fcd22637e03326`
  This is the revision in the upstream `requirements-dinov3.txt`.
- EgoHand-v1: `a153616c5192e6a8b09dbd302f682093080c5b83`
  https://huggingface.co/nvidia/EgoHand-v1/tree/a153616c5192e6a8b09dbd302f682093080c5b83
- SOMA-X assets: `f81a01796f652260ff930d5318d6c6b845a505ed`
  https://huggingface.co/nvidia/SOMA-X/tree/f81a01796f652260ff930d5318d6c6b845a505ed
- RT-DETR: `55572ddc93a5916c3e486811dc6e62bb2b7bdb9d`
  https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1/tree/55572ddc93a5916c3e486811dc6e62bb2b7bdb9d
- SOMA Python: `py-soma-x==0.3.0`, as pinned by the upstream pipeline.
  The package source was inspected locally.

## Findings

`egohand/models/egohand.py` constructs `SomaHandWrapper(device="cuda")`.
Changing only the predictor device is insufficient. The integration substitutes
a CPU wrapper at model construction. Its `_apply` keeps the rig on CPU when
the parent model moves to MPS. Tensor inputs and outputs cross this boundary
explicitly. The network has no automatic CPU fallback.

`inference/preprocessing/mirroring.py` calls `left.prepare_identity()` and
`left.pose()` directly before right-hand reconstruction. A wrapper around
`SomaHandWrapper.forward()` alone cannot cover this path. The integration
runs the full upstream right-hand mirror function on CPU and returns its
results to the network device.

SOMA 0.3.0 `soma/geometry/batched_skinning.py` supports `warp` and a dense
PyTorch path. `soma/hand/identity_model.py` loads MHR with `torch.jit.load`
and `map_location=self.device`. CPU is a source-supported option. Both CPU rigs now load and return finite geometry. Real checkpoint outputs
match the CPU reference on the calibrated sample; see `verification/`.

The predictor uses CUDA autocast. The first port uses float32, so autocast is
disabled. Its built-in performance reporter calls CUDA synchronization for
any non-CPU device; the adapter does not enable that reporter. It uses MPS
synchronization around measured inference instead.

The DINOv3 wrapper constructs ViT-H/16+ and uses eager intermediate features.
No CUDA-only call was found in this wrapper. The real checkpoint now verifies the backbone and decoder on MPS. The
camera translation solver reaches `torch.linalg.eigvalsh`, unsupported on
this MPS runtime. The integration runs this small solver explicitly on CPU
and restores its NamedTuple outputs to the input device. Automatic MPS
fallback remains disabled. A numerical regression check recovers known
translation on both CPU and MPS inputs.

The official requirements include `onnxruntime-gpu` and CUDA libraries.
The initial macOS integration used ONNX Runtime CPU for RT-DETR. The CLI
now uses Core ML MLProgram with fixed batch-one shapes; the original CPU
provider is selectable as a reference. See the acceleration finding below.

The upstream predictor accepts in-memory frames and returns per-hand results.
Its perspective crop path restores source camera axes and physical hand side.
The adapter reuses that path and disables whole-clip smoothing for live frames.

OpenCV 5 removes `undistortPointsIter`, used by the upstream rational
camera inverse projection. Pin OpenCV 4.11.0.86, matching the upstream
requirement, rather than changing camera mathematics. The calibrated
rational sample passes inference with this version.

## Output conventions

The upstream result has 25 joints per hand, wrist-relative vertices/joints,
SOMA rotation matrices, identity coefficients, global and segment scales,
and camera translation. Camera axes are x right, y down, z forward, in meters.
Camera-space joints equal wrist-relative joints plus translation. The mesh
preview uses wrist-relative geometry; export retains translation separately.

The model checkpoint is 10,657,169,753 bytes. Hugging Face metadata reports
these repositories as public and ungated. Downloads and complete asset
validation remain separate checks.

## License boundaries

EgoPost files declare `LicenseRef-OpenMDW-1.1`; the model card also names
OpenMDW 1.1. SOMA package source declares Apache-2.0. DINOv3 and model assets
have their own terms. Fetch and retain upstream notices. This repository does
not redistribute model weights, assets or the upstream source tree.

## Fixed-shape detector on Core ML

The official detector ONNX has symbolic batch N. Dynamic Core ML partitioning
triggered an Apple compiler concat-shape assertion. Requiring static shapes
without shape inference offloaded too little work. Fixing both inputs to
batch one and applying ONNX Runtime symbolic shape inference enables 673 of
688 graph nodes in six Core ML partitions. Weights remain unchanged. The
source SHA-256 is checked; the derived model has a checksum manifest.

Twenty warm iterations on a real frame measure 163 ms CPU versus 16 ms
Core ML execution, excluding preprocessing. Labels match, maximum raw box
coordinate difference is 0.00171 pixels and score difference is 0.0000021.
Forty calibrated frames retain identical hand-presence combinations and all
finite geometry; maximum mesh-coordinate difference is 0.111 mm.
Core ML ALL compute units allow CPU/GPU/ANE selection, but ANE placement has
not been traced. Do not describe the complete detector as CPU-free.

The upstream detector docstring calls the model RT-DETRv2 R18. The actual
pinned ONNX graph contains conv1/res_layers residual blocks consistent with
that implementation. The current online NVIDIA model card instead describes
HGNetv2-S; do not silently substitute a different model based on that text.

FP16 MPS autocast and stored-half neural weights were tested on the real
checkpoint. Outputs stayed finite, but warm reconstruction was slower than
the float32 control in that experiment. Keep float32. The rig and camera
solver remained float32 throughout. No MLX conversion was used.

## Round 1: targeted MHR MPS port

Direct full-rig MPS probing found four distinct blockers: forced float64
inside MHR skeleton propagation, runtime sparse COO construction in one MHR
linear layer, RBF LU solve, and Warp-based skeleton fitting. The first round
ports only MHR shape evaluation; it does not claim full SOMA Metal execution.

The source asset SHA-256 is
`352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`.
`mhr.py` makes a separate TorchScript derivative. It changes exactly the
`use_double_precision` argument in the stored skeleton function and replaces
the 3000 x 750 sparse matrix construction with a constant dense tensor with
the same entries. Graph structure, layer shape and source hash are checked.
Original weights are not overwritten. This uses pinned TorchScript graph
APIs; revalidate the derivative when changing PyTorch.

`MHRRunner` moves only MHR inputs to MPS, executes the derived model there
and returns results to the CPU rig. The complete 40-frame real checkpoint
comparison checks shapes, finite values, handedness, camera-space joints and
rotation validity. Maximum geometry displacement is 0.00065 mm. The original
CPU reference remains selectable. Unsupported LU/Warp work is still explicit
CPU work; its later port is a separate optimization round.
