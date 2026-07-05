"""
Structured extraction: turns a meeting's full text into a summary,
a list of decisions, and a list of action items — using Groq's JSON
mode so the output is parseable, not free text we have to regex.

Runs once per meeting, after chunking, not per-chunk — extraction
needs the whole meeting's context to avoid producing duplicate or
fragmented tasks/decisions from adjacent chunks.
"""
import json
from app.config import settings

GROQ_MODEL = "llama-3.3-70b-versatile"

EXTRACTION_SYSTEM_PROMPT = """You are analyzing a meeting transcript or notes document.
Extract the following, using ONLY information explicitly present in the text —
never invent names, dates, or facts not stated.

Respond ONLY with JSON matching this exact schema, nothing else:
{
  "summary": "2-4 sentence summary of the meeting",
  "decisions": ["decision 1", "decision 2", ...],
  "tasks": [
    {"owner": "person name or null if unclear", "task": "what needs doing", "deadline": "deadline text or null if none mentioned"}
  ]
}

If there are no clear decisions or tasks, return empty lists rather than guessing.
"""


def extract_structured_info(meeting_text: str) -> dict:
    """Returns {"summary": str, "decisions": [str], "tasks": [{"owner","task","deadline"}]}.
    Falls back to empty results (not a crash) if no Groq key is configured
    or the call fails — extraction is best-effort, not required for the
    core RAG loop to keep working."""
    empty_result = {"summary": "", "decisions": [], "tasks": []}

    if not settings.groq_api_key:
        return empty_result

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": meeting_text[:12000]},  # cap input size
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        parsed = json.loads(resp.choices[0].message.content)
        return {
            "summary": parsed.get("summary", ""),
            "decisions": parsed.get("decisions", []) or [],
            "tasks": parsed.get("tasks", []) or [],
        }
    except Exception as e:
        print(f"Structured extraction failed: {e}")
        return empty_result