#!/usr/bin/env python3
"""Render docs/reports/2026-05-12_status_update/README.md to HTML and PDF.

Usage:
    uv run python tools/render_status_report.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "docs/reports/2026-05-12_status_update"
MD_PATH = REPORT_DIR / "README.md"
HTML_PATH = REPORT_DIR / "status_report.html"
PDF_PATH = REPORT_DIR / "status_report.pdf"

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


STYLE = r"""
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {
        content: "Seite " counter(page) " / " counter(pages);
        font-size: 8pt;
        color: #888;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
    @bottom-left {
        content: "FACETpy v2 · Statusupdate · 2026-05-12";
        font-size: 8pt;
        color: #888;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
}

html {
    font-family: "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #222;
}

body {
    max-width: 100%;
}

h1 {
    font-size: 22pt;
    color: #1a3a5c;
    margin-top: 0;
    margin-bottom: 4pt;
    padding-bottom: 6pt;
    border-bottom: 2pt solid #1a3a5c;
}

h1 + p strong {
    font-size: 11pt;
    color: #555;
}

h2 {
    font-size: 14pt;
    color: #1a3a5c;
    margin-top: 20pt;
    margin-bottom: 8pt;
    padding-bottom: 3pt;
    border-bottom: 1pt solid #c8d4e0;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    color: #2c5282;
    margin-top: 14pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

p {
    margin: 6pt 0;
    text-align: justify;
}

ul, ol {
    margin: 4pt 0;
    padding-left: 22pt;
}

li {
    margin: 2pt 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    text-align: left;
    padding: 4pt 7pt;
    border-bottom: 0.5pt solid #d0d7de;
    vertical-align: top;
}

th {
    background-color: #eaf1f8;
    color: #1a3a5c;
    font-weight: 600;
    border-bottom: 1pt solid #1a3a5c;
}

tr:nth-child(even) td {
    background-color: #f7f9fc;
}

code {
    font-family: "SF Mono", "Consolas", "Monaco", monospace;
    font-size: 9pt;
    background-color: #f1f3f5;
    padding: 1pt 4pt;
    border-radius: 3pt;
    color: #b3204b;
}

pre {
    background-color: #f7f9fc;
    border: 0.5pt solid #d0d7de;
    border-radius: 4pt;
    padding: 8pt 10pt;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: transparent;
    color: #222;
    padding: 0;
}

blockquote {
    border-left: 3pt solid #1a3a5c;
    background-color: #f0f6fc;
    margin: 8pt 0;
    padding: 6pt 12pt;
    color: #2c5282;
    font-style: normal;
    page-break-inside: avoid;
}

blockquote p {
    margin: 4pt 0;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 8pt auto;
    page-break-inside: avoid;
}

a {
    color: #2c5282;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

hr {
    border: none;
    border-top: 0.5pt solid #c8d4e0;
    margin: 14pt 0;
}

em {
    color: #555;
}

/* Print-friendly page breaks */
h2 {
    page-break-before: auto;
}

/* Tight figure captions */
p img + em, em {
    display: block;
    text-align: center;
    color: #666;
    font-size: 9pt;
    margin-top: -4pt;
}

/* TL;DR blockquote gets special styling */
blockquote:first-of-type {
    border-left-color: #d4a017;
    background-color: #fef9e7;
    color: #6b4d00;
}
"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>FACETpy Statusupdate · 2026-05-12</title>
<style>{style}</style>
</head>
<body>
{body}
</body>
</html>
"""


def fix_image_paths(html: str, base_dir: Path) -> str:
    """Replace relative figure paths with absolute file:// URIs for weasyprint."""
    def repl(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "file://", "/", "data:")):
            return match.group(0)
        abs_path = (base_dir / src).resolve()
        if abs_path.exists():
            return match.group(0).replace(src, abs_path.as_uri())
        return match.group(0)

    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html)


def find_chrome() -> str | None:
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    return shutil.which("chromium") or shutil.which("chrome")


def html_to_pdf_chrome(html_path: Path, pdf_path: Path) -> None:
    chrome = find_chrome()
    if chrome is None:
        raise RuntimeError("No Chrome/Chromium/Edge found for PDF generation")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        "--no-margins",
        "--virtual-time-budget=10000",
        f"file://{html_path.resolve()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Chrome PDF generation failed: {result.stderr}")


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    body_with_abs_imgs = fix_image_paths(body, MD_PATH.parent)
    html = HTML_TEMPLATE.format(style=STYLE, body=body_with_abs_imgs)

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH.relative_to(REPO_ROOT)} ({HTML_PATH.stat().st_size // 1024} KB)")

    html_to_pdf_chrome(HTML_PATH, PDF_PATH)
    print(f"Wrote {PDF_PATH.relative_to(REPO_ROOT)} ({PDF_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
