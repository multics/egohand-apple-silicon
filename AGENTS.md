# Project guidance

- Read `docs/TASK_RECORD.md` and `docs/SOURCE_AUDIT.md` before changes.
- Use Python 3.12 in `.venv`. Keep models, upstream source, calibration,
  camera recordings, session exports, credentials and environments out of Git.
- Use project-relative paths. Do not add a contributor's home directory,
  camera identifier or machine-specific configuration to source or docs.
- Preserve pinned upstream revisions, asset hashes, physical handedness,
  camera geometry and explicit CPU/MPS boundaries. Disable automatic MPS fallback.
- Run unit tests and Ruff. For runtime changes, also use a real checkpoint
  comparison on the same calibrated input. Do not substitute unit tests for inference.
- Keep task status concise. Separate software setup, model inference and
  measured performance. Keep upstream licenses separate from the MIT integration.
