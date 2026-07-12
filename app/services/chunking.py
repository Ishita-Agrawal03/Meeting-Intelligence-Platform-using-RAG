import re
from collections import Counter
from dataclasses import dataclass

CHUNK_TARGET_TOKENS = 700
CHUNK_OVERLAP_RATIO = 0.15
TRANSCRIPT_DETECTION_THRESHOLD = 0.6

# A line like "Word: rest of line" is only a CANDIDATE speaker turn —
# on its own this pattern also matches section headers like "Summary:",
# "Action:", "Note:". We filter those out below.
SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .]{0,40}):\s+\S")

# Common non-name words that show up as "Word:" headers in meeting
# notes but are never actual speaker names. Checked case-insensitively.
NON_SPEAKER_WORDS = {
    "summary", "note", "notes", "agenda", "action", "actions", "decision",
    "decisions", "task", "tasks", "deadline", "deadlines", "participants",
    "attendees", "duration", "project", "title", "date", "location",
    "minutes", "topic", "objective", "goal", "status", "priority", "owner",
    "meeting", "overview", "background", "context", "next", "steps",
    "conclusion", "recap", "purpose", "attendee",
}


def _looks_like_name(candidate: str) -> bool:
    """A real speaker label is a short, name-shaped phrase — not a
    common section-header word."""
    words = candidate.strip().split()
    if not (1 <= len(words) <= 3):
        return False
    if candidate.strip().lower() in NON_SPEAKER_WORDS:
        return False
    return True


def _find_real_speakers(lines: list[str]) -> set[str]:
    """A candidate is only treated as a real speaker if its label
    repeats at least twice — a genuine back-and-forth conversation has
    the same names recurring; a one-off section header does not."""
    candidates = Counter()
    for line in lines:
        m = SPEAKER_LINE_RE.match(line)
        if m:
            name = m.group(1).strip()
            if _looks_like_name(name):
                candidates[name] += 1
    return {name for name, count in candidates.items() if count >= 2}


def detect_speakers(raw_text: str) -> set[str]:
    """Public entry point: returns the set of real speaker names found
    in a transcript, using the same detection logic used for chunking.
    Returns an empty set for notes documents with no genuine speakers.
    Used to auto-populate the participants table at upload time."""
    lines = [l for l in raw_text.splitlines() if l.strip()]
    return _find_real_speakers(lines)


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
    real_speakers = _find_real_speakers(lines)
    if not real_speakers:
        return "notes"
    matches = sum(
        1 for l in lines
        if (m := SPEAKER_LINE_RE.match(l)) and m.group(1).strip() in real_speakers
    )
    ratio = matches / len(lines)
    return "transcript" if ratio >= TRANSCRIPT_DETECTION_THRESHOLD else "notes"


def _chunk_transcript(raw_text: str) -> list[ChunkResult]:
    lines = [l for l in raw_text.splitlines() if l.strip()]
    real_speakers = _find_real_speakers(lines)

    turns = []
    for line in lines:
        m = SPEAKER_LINE_RE.match(line)
        if m and m.group(1).strip() in real_speakers:
            speaker = m.group(1).strip()
            turns.append([speaker, line])
        elif turns:
            # not a real speaker label (e.g. a "Note:"-style line) —
            # treat as a continuation of whoever is currently speaking
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


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(raw_text: str) -> list[str]:
    """Splits notes text into chunkable units. Prefers blank-line
    paragraphs, but falls back to sentence-splitting for any block
    that has no paragraph breaks at all (e.g. text pasted as one
    continuous block with no blank lines) — otherwise that whole
    block becomes a single oversized chunk instead of being split."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

    units = []
    for para in paragraphs:
        if _approx_tokens(para) > CHUNK_TARGET_TOKENS * 1.5:
            sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(para) if s.strip()]
            units.extend(sentences if sentences else [para])
        else:
            units.append(para)
    return units


def _chunk_notes(raw_text: str) -> list[ChunkResult]:
    units = _split_into_units(raw_text)

    chunks: list[ChunkResult] = []
    current_units: list[str] = []
    current_tokens = 0
    position = 0

    def flush():
        nonlocal current_units, current_tokens, position
        if not current_units:
            return
        text = "\n\n".join(current_units)
        chunks.append(ChunkResult(
            text=text, chunk_type="notes", speakers="", position=position
        ))
        position += 1

    for unit in units:
        current_units.append(unit)
        current_tokens += _approx_tokens(unit)
        if current_tokens >= CHUNK_TARGET_TOKENS:
            flush()
            overlap_n = max(1, int(len(current_units) * CHUNK_OVERLAP_RATIO))
            overlap = current_units[-overlap_n:]
            current_units = list(overlap)
            current_tokens = sum(_approx_tokens(u) for u in current_units)

    flush()
    return chunks


def chunk_document(raw_text: str, source_type: str = None) -> list[ChunkResult]:
    if source_type is None:
        source_type = detect_source_type(raw_text)
    if source_type == "transcript":
        return _chunk_transcript(raw_text)
    return _chunk_notes(raw_text)