"""Utilities for extracting text from clinical PDF documents."""

from pypdf import PdfReader


def extract_pdf_pages(pdf_path):
    """Extract text from a PDF while preserving page numbers."""

    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages

def combine_pdf_text(pages):
    """Combine extracted PDF pages into one document for NLP analysis."""

    return "\n\n".join(page["text"] for page in pages)