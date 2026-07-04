import re
from dataclasses import dataclass

CHUNK_TARGET_TOKENS = 700
CHUNK_OVERLAP_RATIO = 0.15
TRANSCRIPT_DETECTION_THRESHOLD = 0.6

SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .]{0,40}):\s+\S")


@dataclass
class ChunkResult:
    text: str
    chunk_type: str
    speakers: str
    position: int


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def detect_source_type(raw_text: str) -> str:
    lines = [l for l in raw_text.splitlines() if l.strip()]
    if not lines:
        return "notes"
    matches = sum(1 for l in lines if SPEAKER_LINE_RE.match(l))
    ratio = matches / len(lines)
    return "transcript" if ratio >= TRANSCRIPT_DETECTION_THRESHOLD else "notes"


def _chunk_transcript(raw_text: str) -> list[ChunkResult]:
    lines = [l for l in raw_text.splitlines() if l.strip()]

    turns = []
    for line in lines:
        m = SPEAKER_LINE_RE.match(line)
        if m:
            speaker = m.group(1).strip()
            turns.append([speaker, line])
        elif turns:
            turns[-1][1] += " " + line.strip()
        else:
            turns.append(["Unknown", line])

    chunks: list[ChunkResult] = []
    current_lines: list[str] = []
    current_speakers = set()
    current_tokens = 0
    position = 0

    def flush():
        nonlocal current_lines, current_speakers, current_tokens, position
        if not current_lines:
            return
        text = "\n".join(current_lines)
        chunks.append(ChunkResult(
            text=text,
            chunk_type="transcript",
            speakers=",".join(sorted(current_speakers)),
            position=position,
        ))
        position += 1

    for speaker, line in turns:
        current_lines.append(line)
        current_speakers.add(speaker)
        current_tokens += _approx_tokens(line)

        if current_tokens >= CHUNK_TARGET_TOKENS:
            flush()
            overlap_n = max(1, int(len(current_lines) * CHUNK_OVERLAP_RATIO))
            overlap_lines = current_lines[-overlap_n:]
            current_lines = list(overlap_lines)
            current_speakers = set()
            current_tokens = sum(_approx_tokens(l) for l in current_lines)

    flush()
    return chunks


def _chunk_notes(raw_text: str) -> list[ChunkResult]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

    chunks: list[ChunkResult] = []
    current_paras: list[str] = []
    current_tokens = 0
    position = 0

    def flush():
        nonlocal current_paras, current_tokens, position
        if not current_paras:
            return
        text = "\n\n".join(current_paras)
        chunks.append(ChunkResult(
            text=text, chunk_type="notes", speakers="", position=position
        ))
        position += 1

    for para in paragraphs:
        current_paras.append(para)
        current_tokens += _approx_tokens(para)
        if current_tokens >= CHUNK_TARGET_TOKENS:
            flush()
            overlap_n = max(1, int(len(current_paras) * CHUNK_OVERLAP_RATIO))
            overlap = current_paras[-overlap_n:]
            current_paras = list(overlap)
            current_tokens = sum(_approx_tokens(p) for p in current_paras)

    flush()
    return chunks


def chunk_document(raw_text: str, source_type: str = None) -> list[ChunkResult]:
    if source_type is None:
        source_type = detect_source_type(raw_text)
    if source_type == "transcript":
        return _chunk_transcript(raw_text)
    return _chunk_notes(raw_text)