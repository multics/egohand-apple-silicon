"""Resume the official checkpoint and verify its published LFS SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "nvidia/EgoHand-v1"
REVISION = "a153616c5192e6a8b09dbd302f682093080c5b83"
FILENAME = "egohand_s060000.ckpt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    api = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        info = json.load(response)
    entry = next(item for item in info["siblings"] if item["rfilename"] == FILENAME)
    expected_size, expected_hash = entry["lfs"]["size"], entry["lfs"]["sha256"]
    args.directory.mkdir(parents=True, exist_ok=True)
    final = args.directory / FILENAME
    partial = args.directory / (FILENAME + ".partial")
    if final.exists():
        if final.stat().st_size != expected_size or sha256(final) != expected_hash:
            raise RuntimeError("Existing checkpoint failed integrity validation")
        print("Checkpoint already complete and verified")
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError("Partial file is larger than the official checkpoint")
    if offset < expected_size:
        request = urllib.request.Request(
            f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}",
            headers={"Range": f"bytes={offset}-"} if offset else {},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and (
                response.status != 206
                or not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                )
            ):
                raise RuntimeError("Server did not honor the resume offset")
            with partial.open("ab") as handle:
                while data := response.read(1024 * 1024):
                    handle.write(data)
                    print(f"{handle.tell()} / {expected_size} bytes", flush=True)
    if partial.stat().st_size != expected_size or sha256(partial) != expected_hash:
        raise RuntimeError(
            "Incomplete checkpoint or SHA-256 mismatch; final file not created"
        )
    partial.replace(final)
    print("Checkpoint complete; published SHA-256 verified")


if __name__ == "__main__":
    main()
