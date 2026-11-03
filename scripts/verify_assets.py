"""Check required inference files against pinned Hugging Face LFS hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from fetch_assets import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["egohand", "soma", "detector"])
    args = parser.parse_args()
    evidence = []
    for repo, (revision, directory, files) in MODELS.items():
        if args.models and directory not in args.models:
            continue
        api = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(api, timeout=60) as response:
            metadata = json.load(response)
        siblings = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        for filename in files:
            if filename.endswith("README.md") or filename == "LICENSE":
                continue
            entry = siblings[filename]
            path = args.directory / directory / filename
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise RuntimeError(f"Missing or incomplete asset: {path}")
            sha256 = hashlib.sha256()
            sha1 = hashlib.sha1(b"blob " + str(entry["size"]).encode() + b"\0")
            with path.open("rb") as handle:
                while data := handle.read(8 * 1024 * 1024):
                    sha256.update(data)
                    sha1.update(data)
            expected = entry.get("lfs", {}).get("sha256")
            if expected:
                if sha256.hexdigest() != expected:
                    raise RuntimeError(f"LFS SHA-256 mismatch: {path}")
            elif sha1.hexdigest() != entry["blobId"]:
                raise RuntimeError(f"Git blob checksum mismatch: {path}")
            evidence.append(
                {
                    "repo": repo,
                    "revision": revision,
                    "filename": filename,
                    "bytes": entry["size"],
                    "sha256": sha256.hexdigest(),
                    "official_hash_verified": True,
                }
            )
            print(f"Verified: {path}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
