from pathlib import Path

import pymupdf


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Will take PDF path in and outputs list of (page_num, text)"""
    doc = pymupdf.open(path)
    results = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        results.append((page_num, text))

    doc.close()
    return results
