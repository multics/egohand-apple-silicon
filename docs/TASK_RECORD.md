# Public release task

## Request and scope

Provide a portable Apple Silicon EgoHand integration under MIT. Keep measured
hardware and performance evidence. Exclude private paths, camera identifiers,
personal calibration, development session history and machine-specific setup.
This repository starts with a clean history. It uses the full-mesh integration;
experimental joints-only and native Metal rig branches are not included.

## Acceptance

- Setup and launcher resolve resources from the checkout.
- Dependencies and upstream revisions are pinned.
- Unit tests, lint and real-checkpoint validation have separate outcomes.
- Published content and history contain no private paths or configuration.
- Model resources are fetched separately under upstream license terms.

## Status

Fresh Python 3.12 environment installed successfully with pinned dependencies.
Pinned upstream source fetch checked 122 files. Imports, MPS tensor checks,
dependency consistency, 11 unit tests and Ruff pass. The launcher works
from another directory and accepts camera/calibration/output overrides.
Real released-checkpoint inference on a calibrated public sample produces
finite left/right outputs with 25 joints and 2,859 vertices per hand and
valid rotations. Model assets were reused locally and verified against
official hashes; the multi-GB weight download was not repeated. These are
release correctness checks, not a new benchmark.

Historical performance data is kept
in `verification/`; paths and session identifiers have been removed. See
`BENCHMARKS.md` for measurement scope. No new performance result is claimed.

Release launcher validation: two frames generated from the same calibrated
public image completed Core ML detection and MPS reconstruction when launched
from outside the checkout. Both frames contain both physical hands; exported
geometry passed validation. This checks the launch path, not video accuracy
or throughput. See `verification/release-checks.json`. Original private
repository and runtime remain unchanged.

Public-clone follow-up: the second checkout exposed an editable-install path
problem. Setup now installs the integration as a regular wheel and reinstalls
it when setup is repeated. The source runtime is unchanged. A fresh environment in
the public clone now passes setup, imports, MPS tensor checks, all 11 tests,
Ruff and dependency checks. CLI help works when the launcher is invoked from
another directory. Public source contains no private configuration or history.
Release validation is complete; no new speed benchmark is claimed.
