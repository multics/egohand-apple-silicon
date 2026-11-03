"""Fetch only the pinned EgoPost sources, without a full Isaac Capture clone."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "a74babb02cd8a018749f1f2374b841ea04650f1b"
PREFIX = "src/python/isaaccapture_egopost/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    url = f"https://api.github.com/repos/NVIDIA/IsaacCapture/git/trees/{REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as response:
        tree = json.load(response)
    files = [
        f
        for f in tree["tree"]
        if f["type"] == "blob"
        and (f["path"].startswith(PREFIX) or f["path"].startswith("LICENSE"))
    ]

    def fetch(entry):
        path = entry["path"]
        with urllib.request.urlopen(
            f"https://raw.githubusercontent.com/NVIDIA/IsaacCapture/{REVISION}/{path}",
            timeout=60,
        ) as response:
            data = response.read()
        digest = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if digest != entry["sha"]:
            raise ValueError(f"Git blob checksum mismatch: {path}")
        relative = path[len(PREFIX) :] if path.startswith(PREFIX) else path
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, files))
    (args.destination / "UPSTREAM_REVISION").write_text(REVISION + "\n")
    print(f"Fetched and checked {len(files)} files at {REVISION}")


if __name__ == "__main__":
    main()
