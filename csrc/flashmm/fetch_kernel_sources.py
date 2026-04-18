"""Fetch the two large flashmm CUDA sources from the upstream m2 repo.

These two files are not vendored in-tree due to size. Running this script
downloads them into the right place so the flashmm extension can be built.

Requires the GITHUB_TOKEN env var if the m2 repository is private.
"""

import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "larochec/m2"
REF = "main"
FILES = [
    "csrc/flashmm/mm_block_fwd_cuda.cu",
    "csrc/flashmm/lut.h",
]


def fetch(path_in_m2: str) -> None:
    url = f"https://raw.githubusercontent.com/{REPO}/{REF}/{path_in_m2}"
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    target = HERE / Path(path_in_m2).name
    print(f"Downloading {url} -> {target}")
    with urllib.request.urlopen(req) as resp:
        target.write_bytes(resp.read())


def main() -> int:
    for f in FILES:
        try:
            fetch(f)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to fetch {f}: {exc}", file=sys.stderr)
            print(
                "If the m2 repo is private, export GITHUB_TOKEN with an access token.",
                file=sys.stderr,
            )
            return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
