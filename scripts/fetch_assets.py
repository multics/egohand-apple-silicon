"""Download official model assets at fixed revisions into a local cache."""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS = {
    "nvidia/EgoHand-v1": (
        "a153616c5192e6a8b09dbd302f682093080c5b83",
        "egohand",
        ["egohand_s060000.ckpt", "README.md"],
    ),
    "nvidia/SOMA-X": (
        "f81a01796f652260ff930d5318d6c6b845a505ed",
        "soma",
        [
            "SOMAHand.npz",
            "SOMA_neutral.npz",
            "SOMA_template_rig.usda",
            "SOMA_procedural_transforms.json",
            "MHR/mhr_model_lod1.pt",
            "MHR/base_body_lod1.obj",
            "MHR/SOMA_wrap_lod1.obj",
            "LICENSE",
        ],
    ),
    "nvidia/RT-DETR-Hand-Detector-v1": (
        "55572ddc93a5916c3e486811dc6e62bb2b7bdb9d",
        "detector",
        ["hand_detector.onnx", "README.md"],
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    p.add_argument("--models", nargs="+", choices=["egohand", "soma", "detector"])
    args = p.parse_args()
    for repo, (revision, directory, files) in MODELS.items():
        if args.models and directory not in args.models:
            continue
        for filename in files:
            print(f"Downloading {repo}@{revision}: {filename}", flush=True)
            hf_hub_download(
                repo,
                filename,
                revision=revision,
                local_dir=args.destination / directory,
            )


if __name__ == "__main__":
    main()
