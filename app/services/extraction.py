from pathlib import Path


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n\n".join(pages_text)

    elif suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")