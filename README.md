# EgoHand on Apple Silicon

Run NVIDIA EgoHand-v1 on Apple Silicon with PyTorch MPS and a Core ML hand
detector. This experimental integration exports both hand meshes, 25 joints
per hand, pose rotations and camera translation.

## Quick start

Requires an Apple Silicon Mac, macOS 14 or later, Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The setup script
creates a Python 3.12 environment. Only the M2 Max configuration below has
measured performance; other Macs are not covered by these measurements.
Allow tens of GB of free storage for dependencies, downloaded assets and
model loading. The EgoHand checkpoint alone is about 10.66 GB. Peak memory
and a minimum supported RAM size have not been established.

```sh
git clone https://github.com/multics/egohand-apple-silicon.git
cd egohand-apple-silicon
./setup.sh
./run_demo.sh
```

No personal path needs to be edited. Setup installs the pinned dependencies,
fetches the pinned upstream source, downloads the model assets and verifies
their hashes. It does not bundle weights. Read the
[third-party terms](docs/THIRD_PARTY.md) before downloading. If a host requires
access approval or authentication, complete that with the model provider.
`./setup.sh --skip-assets` installs software without downloading weights;
run the full setup before inference.

Grant camera access to the terminal application when macOS asks. The default
camera index is 0. Select another camera with `./run_demo.sh --camera 1`.
The program writes a new session under `runs/`. Press Q or Escape to stop;
drag the mesh preview to rotate the view.

### Camera geometry

The default demo uses **estimated intrinsics** and labels the result as
uncalibrated. It is suitable for an interaction check. Metric depth is not
verified. For a measured camera, provide a calibration file that matches
its exact resolution and the upstream camera schema:

```sh
./run_demo.sh --calibration /path/to/your-camera.json
```

That path is your input, not a developer-specific path. Calibration files
are not included. For video input, use:

```sh
./run_demo.sh --video /path/to/your-video.mp4 --no-preview --max-frames 40
```

Use `--help` for all options. Resource paths in the launcher resolve from
the checkout, even when the launcher is called from another directory.
User-provided relative input/output paths also resolve from that checkout.
Use absolute paths for inputs elsewhere.

## Runtime boundaries

- RT-DETR uses ONNX Runtime Core ML with a fixed batch-one graph. CPU is
  selectable with `--detector-device cpu`.
- DINOv3 and the main and auxiliary decoders use MPS in FP32. Automatic MPS
  fallback is disabled.
- MHR shape evaluation uses a separately generated MPS-compatible model.
  Original assets and learned values are retained.
- SOMA fitting/skinning, physical right-hand mirroring and the camera
  translation solver have explicit CPU boundaries.
- Frames are processed independently, without offline temporal smoothing.
  Live capture keeps the latest frame instead of building a stale queue.

This release uses the full-mesh baseline. It does not include later
joints-only or native Metal rig experiments. Read the
[source audit](docs/SOURCE_AUDIT.md) for the pinned revisions and boundaries.
Camera axes are x right, y down and z forward, in meters. Meshes and joints
are wrist-relative; camera translation is exported separately.

## Hardware and measured performance

Recorded on Apple M2 Max: 12 CPU cores, 38 GPU cores and 96 GB unified memory.
The reference machine has an 8 TB SSD. These are test-machine specifications,
not minimum requirements.

| Measurement | Recorded result |
| --- | --- |
| Warm detector graph, CPU → Core ML | 163.1 → 16.2 ms median |
| Matched replay, one-hand reconstruction, CPU MHR → MPS MHR | 220.2 → 182.1 ms median |
| Matched replay, two-hand reconstruction, CPU MHR → MPS MHR | 289.8 → 243.5 ms median |
| Separate full-mesh profile, two-hand reconstruction | 247.6 ms median |

These are separate experiments recorded on October 28 and 31, 2026. The
measurements do not establish sustained real-time operation or ANE placement.
Read [measurement scope and limitations](docs/BENCHMARKS.md). Sanitized
historical JSON summaries are in `docs/verification/`. New setup checks are
not new benchmark measurements.

## Validation

After setup, from the checkout:

```sh
.venv/bin/python scripts/doctor.py
.venv/bin/python -m unittest discover -s tests -v
```

`doctor.py` checks imports and a small MPS tensor operation. It does not mark
model inference as verified. To validate the released checkpoint, use a real
image and its measured calibration:

```sh
.venv/bin/python scripts/verify_inference.py \
  --source assets/egopost \
  --checkpoint weights/egohand/egohand_s060000.ckpt \
  --assets weights/soma --detector weights/detector/hand_detector.onnx \
  --image /path/to/your-frame.png \
  --calibration /path/to/your-camera.json --output runs/frame-check
```

A successful camera preview, random-weight model or unit test does not prove
released-checkpoint accuracy. Compare CPU and MPS on the same calibrated
input before changing precision, model files or device boundaries.

## Limits and troubleshooting

- Core ML can take several seconds to compile at first use. Its graph
  assignment does not prove which nodes execute on the Neural Engine.
- Keep OpenCV at 4.11.0.86. The pinned camera code uses an API removed in
  OpenCV 5. Keep the tested PyTorch version for the MHR graph conversion.
- Use `--profile-stages` for synchronized stage timing. Synchronization can
  reduce throughput; record whether it was enabled when comparing results.
- Camera index and permission vary between machines. A file or camera error
  does not validate model inference.
- Numerical parity is agreement with a reference, not ground-truth pose
  accuracy. Calibrated live accuracy and sustained performance remain open.

## License

Original integration code: [MIT](LICENSE), copyright Yong (Edward) Tian.
Downloaded NVIDIA, Meta, DINOv3, SOMA/MHR code and assets keep their own
licenses. See [third-party resources](docs/THIRD_PARTY.md).
