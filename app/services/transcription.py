"""
Transcribes audio/video files to text using Groq's hosted Whisper API.

Reuses the same GROQ_API_KEY already configured for chat/extraction —
no separate credential or local model download needed.

Known limitation: plain Whisper transcription returns continuous text
with no speaker labels, so downstream chunking will classify this as
"notes" rather than "transcript" (see detect_source_type in
chunking.py) — true speaker diarization is a separate, more complex
feature not covered by this basic transcription call.
"""
from pathlib import Path
from app.config import settings

GROQ_WHISPER_MODEL = "whisper-large-v3"


def transcribe_audio(filepath: Path) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set — cannot transcribe audio.")

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    with open(filepath, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(filepath.name, f.read()),
            model=GROQ_WHISPER_MODEL,
            response_format="text",
        )

    # Depending on SDK version, this comes back as a plain string or
    # an object with a .text attribute — handle both.
    if isinstance(transcription, str):
        return transcription
    return getattr(transcription, "text", str(transcription))