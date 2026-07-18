"""
Runs the benchmark questions against a LIVE /chat endpoint and logs
results to evaluation.db — not just stdout — so you can track trend
over time across runs (e.g. after tuning chunk size or prompts).

Reports TWO separate numbers on purpose:
  - retrieval_hit_rate: did the correct PROJECT show up among cited
    sources? (checks project_name, not meeting_title — meeting titles
    are now auto-set to the uploaded filename, not a stable label)
  - answer_accuracy: did the final answer contain the expected text?
Conflating these hides whether failures come from retrieval or from
the LLM's answer generation.

IMPORTANT: this benchmark expects two real projects to already exist,
created via the Streamlit UI (or POST /projects/{id}/upload):
  - "Smart Customer Support AI"  <- upload the kickoff meeting transcript
  - "PostgreSQL Architecture Decision"  <- upload the Rahul/Ishita transcript
Project names must match exactly (case-sensitive) or retrieval checks
will fail even if the system is working correctly.

Usage:
    (make sure `uvicorn app.main:app --reload` is already running)
    python eval/run_eval.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
import time
from pathlib import Path
import requests

from eval_db import init_eval_db, EvalSessionLocal, EvaluationRun, EvaluationResult

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"

# Free-tier Groq accounts have a per-minute request/token rate limit.
# Each benchmark question can trigger up to 2 Groq calls (query rewrite
# + answer generation), so firing all 12 back-to-back with no pause
# can trip a 429. This delay keeps the run comfortably under the limit.
DELAY_BETWEEN_QUESTIONS_SECONDS = 10

ABSTENTION_PHRASES = [
    "don't know", "do not know", "no information", "not mentioned",
    "doesn't mention", "does not mention", "couldn't find", "could not find",
    "not discuss",  # catches "not discussed", "did not discuss", "does not discuss"
    "no mention", "not addressed", "unable to find",
    "doesn't contain", "does not contain", "don't contain", "do not contain",
    "not covered",
]


def _is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def _call_chat_with_retry(base_url: str, payload: dict, max_retries: int = 3):
    """Retries once or twice with backoff if Groq rate-limits us mid-run,
    instead of just recording a failure and moving on."""
    for attempt in range(max_retries):
        resp = requests.post(f"{base_url}/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "429" in data.get("answer", "") or "Rate limit" in data.get("answer", ""):
            wait = 15 * (attempt + 1)
            print(f"  Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
            continue
        return data
    return data  # give up after max_retries, return whatever the last attempt gave


def run(base_url: str):
    init_eval_db()
    questions = json.loads(BENCHMARK_PATH.read_text())

    projects = requests.get(f"{base_url}/projects/").json()
    project_name_to_id = {p["name"]: p["id"] for p in projects}

    per_question_results = []

    for i, q in enumerate(questions):
        is_adversarial = q.get("is_adversarial", False)
        payload = {"query": q["question"], "chat_history": []}

        expected_project = q.get("expected_project_name")
        if expected_project and expected_project in project_name_to_id:
            payload["project_id"] = project_name_to_id[expected_project]

        try:
            data = _call_chat_with_retry(base_url, payload)
        except Exception as e:
            print(f"Request failed for '{q['question']}': {e}")
            per_question_results.append({
                "question": q["question"],
                "answer_text": f"REQUEST FAILED: {e}",
                "retrieved_chunk_ids": [],
                "answer_correct": False,
                "retrieval_correct": False,
                "expected_project_name": q.get("expected_project_name"),
                "is_adversarial": is_adversarial,
            })
            continue

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        retrieved_ids = data.get("retrieved_chunk_ids", [])

        if is_adversarial:
            answer_correct = _is_abstention(answer)
            retrieval_correct = None
        else:
            answer_correct = False
            if q.get("expected_answer_contains"):
                answer_correct = q["expected_answer_contains"].lower() in answer.lower()

            retrieval_correct = False
            if q.get("expected_project_name"):
                cited_projects = {c["project_name"] for c in citations}
                retrieval_correct = q["expected_project_name"] in cited_projects
            else:
                retrieval_correct = len(retrieved_ids) > 0

        per_question_results.append({
            "question": q["question"],
            "answer_text": answer,
            "retrieved_chunk_ids": retrieved_ids,
            "answer_correct": answer_correct,
            "retrieval_correct": retrieval_correct,
            "expected_project_name": q.get("expected_project_name"),
            "is_adversarial": is_adversarial,
        })

        if i < len(questions) - 1:
            time.sleep(DELAY_BETWEEN_QUESTIONS_SECONDS)

    n = len(per_question_results)
    retrieval_scored = [r for r in per_question_results if r["retrieval_correct"] is not None]
    retrieval_rate = (
        sum(r["retrieval_correct"] for r in retrieval_scored) / len(retrieval_scored)
        if retrieval_scored else 0
    )
    answer_rate = sum(r["answer_correct"] for r in per_question_results) / n if n else 0

    db = EvalSessionLocal()
    eval_run = EvaluationRun(
        retrieval_hit_rate=retrieval_rate,
        answer_accuracy=answer_rate,
        notes=f"{n} questions, base_url={base_url}",
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    run_id = eval_run.id

    for r in per_question_results:
        db.add(EvaluationResult(
            run_id=run_id,
            question=r["question"],
            expected_meeting_title=r.get("expected_project_name"),
            retrieved_chunk_ids=",".join(str(cid) for cid in r["retrieved_chunk_ids"]),
            retrieval_correct=r["retrieval_correct"],
            answer_correct=r["answer_correct"],
            answer_text=r["answer_text"],
        ))
    db.commit()
    db.close()

    print(f"\nRan {n} benchmark questions (run_id={run_id})")
    print(f"Retrieval hit rate:  {retrieval_rate:.0%}  (over {len(retrieval_scored)} scorable questions)")
    print(f"Answer accuracy:     {answer_rate:.0%}\n")

    for r in per_question_results:
        status = "OK  " if r["answer_correct"] else "MISS"
        tag = " [adversarial]" if r["is_adversarial"] else ""
        print(f"[{status}]{tag} {r['question']}")
        print(f"        retrieved_chunks={r['retrieved_chunk_ids']}")
        print(f"        answer='{r['answer_text'][:100]}'")

    return per_question_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    run(args.base_url)