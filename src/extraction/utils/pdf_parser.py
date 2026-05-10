"""Extract raw text from ETSI standard PDFs using pdftotext."""
import subprocess
import re
from pathlib import Path


def extract_text(pdf_path: str | Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def clean_text(text: str) -> str:
    # Remove page headers/footers (ETSI lines and page numbers)
    text = re.sub(r"ETSI\s*\n", "\n", text)
    text = re.sub(r"ETSI EN \d+ \d+ V[\d.]+ \(\d{4}-\d{2}\)\s*\n", "\n", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_standard(pdf_path: str | Path) -> str:
    raw = extract_text(pdf_path)
    return clean_text(raw)