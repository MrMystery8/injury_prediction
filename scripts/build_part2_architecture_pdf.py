#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

SOURCE_MD = DOCS_DIR / "PART2_ARCHITECTURE_FRAMEWORK_DESIGN.md"
RENDERED_MD = DOCS_DIR / "PART2_ARCHITECTURE_FRAMEWORK_DESIGN_RENDERED.md"

RENDERED_PDF = DOCS_DIR / "PART2_ARCHITECTURE_FRAMEWORK_DESIGN_RENDERED.pdf"
PRIMARY_PDF = DOCS_DIR / "PART2_ARCHITECTURE_FRAMEWORK_DESIGN.pdf"


def _extract_mermaid_blocks(markdown: str) -> tuple[str, list[str]]:
    blocks: list[str] = []
    out_parts: list[str] = []

    lines = markdown.splitlines(keepends=True)
    i = 0
    figure_index = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "```mermaid":
            figure_index += 1
            i += 1
            mermaid_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "```":
                mermaid_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise RuntimeError("Unclosed ```mermaid block in markdown.")
            i += 1  # consume closing ```

            mermaid_src = "".join(mermaid_lines).strip() + "\n"
            blocks.append(mermaid_src)

            out_parts.append(f"![Figure {figure_index}](assets/part2_figure_{figure_index}.png)\n\n")
            continue

        out_parts.append(line)
        i += 1

    rendered = "".join(out_parts)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered, blocks


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    source = SOURCE_MD.read_text(encoding="utf-8")
    rendered_md, mermaid_blocks = _extract_mermaid_blocks(source)

    RENDERED_MD.write_text(rendered_md, encoding="utf-8")

    # Render mermaid blocks to PNGs.
    for idx, block in enumerate(mermaid_blocks, start=1):
        mmd_path = ASSETS_DIR / f"part2_figure_{idx}.mmd"
        png_path = ASSETS_DIR / f"part2_figure_{idx}.png"
        mmd_path.write_text(block, encoding="utf-8")
        _run(
            [
                "npx",
                "-y",
                "@mermaid-js/mermaid-cli",
                "-i",
                str(mmd_path),
                "-o",
                str(png_path),
                "--backgroundColor",
                "white",
            ],
            cwd=ROOT,
        )

    # Remove any old, now-unused part2_figure_N.{png,mmd} artefacts.
    used = {f"part2_figure_{i}" for i in range(1, len(mermaid_blocks) + 1)}
    for path in ASSETS_DIR.glob("part2_figure_*.*"):
        stem = path.name.split(".", 1)[0]
        if stem not in used:
            path.unlink(missing_ok=True)

    # Build PDF from the rendered markdown (images embedded).
    _run(["npx", "-y", "md-to-pdf", str(RENDERED_MD)], cwd=ROOT)

    if not RENDERED_PDF.exists():
        raise RuntimeError(f"Expected PDF not found: {RENDERED_PDF}")

    shutil.copyfile(RENDERED_PDF, PRIMARY_PDF)


if __name__ == "__main__":
    main()

