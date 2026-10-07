"""Build small, real PDFs (with extractable text) for tests."""

DEFAULT_LINES = [
    "Marie Curie was a physicist who worked at the University of Paris.",
    "Marie Curie discovered polonium and radium with Pierre Curie.",
    "Radioactivity was studied by Marie Curie and Henri Becquerel.",
    "The University of Paris awarded Marie Curie a doctorate in 1903.",
]


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_text_pdf(lines: list[str] | str | None = None) -> bytes:
    """Return the bytes of a one-page PDF whose text pypdf can extract."""
    if lines is None:
        lines = DEFAULT_LINES
    elif isinstance(lines, str):
        lines = [lines]

    body = " T* ".join(f"({_escape(line)}) Tj" for line in lines)
    stream = f"BT /F1 11 Tf 14 TL 40 750 Td {body} ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref,
    )
    return bytes(out)
