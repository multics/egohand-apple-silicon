# Recorded performance

## Reference hardware

MacBook Pro, Apple M2 Max: 12 CPU cores (8 performance and 4 efficiency),
38 GPU cores, 96 GB unified memory and 8 TB storage. Recorded system version:
macOS 27.2. This is a measurement environment, not a minimum requirement.
Python 3.12 and FP32 PyTorch MPS inference were used.

## Separate experiments

| Experiment | Conditions | Result |
| --- | --- | --- |
| Detector graph | 20 warm iterations; excludes preprocessing | CPU 163.07 ms median / 164.50 ms p95; Core ML 16.21 / 18.74 ms |
| MHR offload | Matched 40-frame calibrated replay: 22 one-hand and 18 two-hand frames | One-hand reconstruction 220.17 → 182.14 ms median; two-hand 289.79 → 243.52 ms |
| Same MHR comparison | Whole replay, including different cold spikes | 3.29 → 3.39 FPS; maximum geometry distance 0.000651 mm |
| Full-mesh stage profile | 264 frames; 213 both-hand frames; stage synchronization enabled | Two-hand complete reconstruction 247.60 ms median / 261.36 ms p95 |
| Duplicate-box regression | Separate 40-frame replay; no duplicate detections in that clip | One/two-hand reconstruction 172.3 / 229.6 ms median; not an attributed speedup |

The full-mesh profile has these two-hand stage medians: detection including
preprocessing 25.74 ms, DINOv3 150.54 ms, main 3D decoder 12.94 ms, MHR shape
call 25.72 ms and remaining reconstruction 58.31 ms. These are independent
summaries and are not additive. Wrapper timings overlap MHR. The remaining
work includes the auxiliary 2D path.

Read-to-export age in that profile is 295.07 ms median and 316.79 ms p95.
The clock starts at camera read completion. This is not exposure-to-display
latency. Rendering was not measured separately.

## Interpretation

- Detector timings exclude image preprocessing and do not represent a
  tenfold end-to-end speedup.
- Geometry differences measure numerical agreement, not pose accuracy.
- The reciprocal of a median reconstruction time is not measured FPS.
- Cold compilation, frame waiting, preview and export affect throughput.
- These results were recorded on October 28 and 31, 2026. Public release
  validation does not replace these records with a new performance claim.

## Evidence

The JSON summaries in `verification/` retain measurement values and settings.
Private directory paths, personal camera calibration and session identifiers
are excluded. For `round1-mhr-comparison.json`, the two entries in `sessions`
are CPU MHR reference, then MPS MHR. For `duplicate-box-fix-replay.json`, they
are the MPS reference, then the duplicate-box regression. See
`full-mesh-b1e0baf-profile.json` for the separate synchronized profile.
