import re
from collections import Counter
from dataclasses import dataclass

CHUNK_TARGET_TOKENS = 700
CHUNK_OVERLAP_RATIO = 0.15
TRANSCRIPT_DETECTION_THRESHOLD = 0.6

# Broadened to allow digits, hyphens, apostrophes — "Speaker 1:",
# "Jean-Luc:", "O'Brien:" are all valid speaker labels that the
# original character class silently excluded from ever becoming
# candidates at all.
SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .'\-]{0,40}):\s+\S")

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
    common section-header word. Checks EVERY word in the candidate
    against the stoplist, not just the phrase as a whole — otherwise
    a multi-word header like "Action Items" or "Next Steps" slips
    through since neither exact phrase matches a single stoplist
    entry, even though both individual words clearly should."""
    words = candidate.strip().split()
    if not (1 <= len(words) <= 3):
        return False
    if any(w.lower() in NON_SPEAKER_WORDS for w in words):
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
    Used to auto-populate the participants table at upload time."""
    lines = [l for l in raw_text.splitlines() if l.strip()]
    return _find_real_speakers(lines)


def _group_into_turns(lines: list[str], real_speakers: set[str]) -> list[list]:
    """Groups raw lines into [speaker, text] turns. A line only starts
    a new turn if it matches a CONFIRMED real speaker; anything else
    (including a one-off "Note:"-style line) continues the current
    speaker's turn instead of being wrongly treated as a new one."""
    turns = []
    for line in lines:
        m = SPEAKER_LINE_RE.match(line)
        if m and m.group(1).strip() in real_speakers:
            speaker = m.group(1).strip()
            turns.append([speaker, line])
        elif turns:
            turns[-1][1] += " " + line.strip()
        else:
            turns.append(["Unknown", line])
    return turns


def detect_source_type(raw_text: str) -> str:
    lines = [l for l in raw_text.splitlines() if l.strip()]
    if not lines:
        return "notes"

    real_speakers = _find_real_speakers(lines)
    if not real_speakers:
        return "notes"

    # Ratio is computed over TURNS, not raw lines — a real transcript
    # can have multi-line utterances (one speaker's turn wrapping
    # across several lines), which would previously drag the line-based
    # ratio below the threshold even for a genuine, heavily back-and-
    # forth conversation. Turn count reflects actual speaker changes.
    turns = _group_into_turns(lines, real_speakers)
    speaker_turns = sum(1 for speaker, _ in turns if speaker in real_speakers)
    ratio = speaker_turns / len(turns) if turns else 0
    return "transcript" if ratio >= TRANSCRIPT_DETECTION_THRESHOLD else "notes"


@dataclass
class ChunkResult:
    text: str
    chunk_type: str
    speakers: str
    position: int


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _chunk_transcript(raw_text: str) -> list[ChunkResult]:
    lines = [l for l in raw_text.splitlines() if l.strip()]
    real_speakers = _find_real_speakers(lines)
    turns = _group_into_turns(lines, real_speakers)

    chunks: list[ChunkResult] = []
    current_lines: list[str] = []
    current_speakers: set[str] = set()
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

    # Track which speaker said each line in current_lines so overlap
    # carry-forward can correctly re-derive current_speakers instead
    # of losing attribution entirely (the original bug: current_speakers
    # was reset to an empty set on every overlap window, even though
    # the carried-over LINES still belonged to real speakers).
    current_line_speakers: list[str] = []

    for speaker, line in turns:
        current_lines.append(line)
        current_line_speakers.append(speaker)
        current_speakers.add(speaker)
        current_tokens += _approx_tokens(line)

        if current_tokens >= CHUNK_TARGET_TOKENS:
            flush()
            overlap_n = max(1, int(len(current_lines) * CHUNK_OVERLAP_RATIO))
            current_lines = current_lines[-overlap_n:]
            current_line_speakers = current_line_speakers[-overlap_n:]
            current_speakers = set(current_line_speakers)  # re-derived, not reset to empty
            current_tokens = sum(_approx_tokens(l) for l in current_lines)

    flush()
    return chunks


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(raw_text: str) -> list[str]:
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
            current_units = current_units[-overlap_n:]
            current_tokens = sum(_approx_tokens(u) for u in current_units)

    flush()
    return chunks


def chunk_document(raw_text: str, source_type: str = None) -> list[ChunkResult]:
    if source_type is None:
        source_type = detect_source_type(raw_text)
    if source_type == "transcript":
        return _chunk_transcript(raw_text)
    return _chunk_notes(raw_text)