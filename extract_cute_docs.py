#!/usr/bin/env python3
"""
Extract CuTe/CUTLASS tutorial sources from a Modal H100 sandbox.

Writes extracted docs to outputs/cute_docs/.
"""

from __future__ import annotations

from pathlib import Path

import modal

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git")
    .run_commands("git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass")
)

app = modal.App("kernelbench-cute-docs")


@app.function(gpu="H100", image=IMAGE, timeout=900)
def extract_cute_examples() -> dict:
    import os

    out: dict[str, object] = {}

    tutorial_dir = "/opt/cutlass/examples/cute/tutorial"
    readme_path = "/opt/cutlass/include/cute/README.md"
    examples_root = "/opt/cutlass/examples"

    tutorial_files: dict[str, str] = {}
    tutorial_listing: list[str] = []
    if os.path.isdir(tutorial_dir):
        tutorial_listing = sorted(os.listdir(tutorial_dir))
        for name in tutorial_listing:
            if name.endswith(".cu"):
                path = os.path.join(tutorial_dir, name)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    tutorial_files[name] = fh.read()

    include_hits: list[str] = []
    for root, _, files in os.walk(examples_root):
        for name in files:
            if not name.endswith(".cu"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    data = fh.read()
                if "cute/tensor.hpp" in data:
                    include_hits.append(path)
            except Exception:
                continue
    include_hits = sorted(include_hits)[:10]

    readme = ""
    if os.path.isfile(readme_path):
        with open(readme_path, encoding="utf-8", errors="replace") as fh:
            readme = fh.read()

    out["tutorial_dir"] = tutorial_dir
    out["tutorial_listing"] = tutorial_listing
    out["tutorial_files"] = tutorial_files
    out["include_hits"] = include_hits
    out["readme_path"] = readme_path
    out["readme"] = readme
    return out


def main() -> None:
    output_dir = Path("outputs/cute_docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with app.run():
        docs = extract_cute_examples.remote()

    (output_dir / "include_hits.txt").write_text(
        "\n".join(docs.get("include_hits", [])) + "\n",
        encoding="utf-8",
    )

    listing = docs.get("tutorial_listing", [])
    (output_dir / "tutorial_listing.txt").write_text(
        "\n".join(listing) + "\n",
        encoding="utf-8",
    )

    readme = docs.get("readme", "")
    (output_dir / "cute_README.md").write_text(readme, encoding="utf-8")

    tutorial_files = docs.get("tutorial_files", {})
    tutorial_out = output_dir / "tutorial"
    tutorial_out.mkdir(parents=True, exist_ok=True)
    for name, content in tutorial_files.items():
        (tutorial_out / name).write_text(content, encoding="utf-8")

    print(f"Wrote CuTe docs to {output_dir}")
    print(f"Tutorial files: {len(tutorial_files)}")
    print(f"Include hits: {len(docs.get('include_hits', []))}")


if __name__ == "__main__":
    main()
