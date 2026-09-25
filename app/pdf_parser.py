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

def build_page_spans(pages):
    """
    Map each PDF page to its character range in the
    combined clinical document.
    """

    page_spans = []
    position = 0

    for index, page in enumerate(pages):

        page_text = page["text"]

        start = position
        end = start + len(page_text)

        page_spans.append(
            {
                "page": page["page"],
                "start": start,
                "end": end,
            }
        )

        position = end

        if index < len(pages) - 1:
            position += 2

    return page_spans