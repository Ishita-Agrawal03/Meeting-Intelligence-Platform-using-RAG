"""
Runs the benchmark questions against a LIVE /chat endpoint and logs
results to evaluation.db — not just stdout — so you can track trend
over time across runs (e.g. after tuning chunk size or prompts).

Reports TWO separate numbers on purpose:
  - retrieval_hit_rate: did the correct meeting show up among cited sources?
  - answer_accuracy: did the final answer contain the expected text?
Conflating these hides whether failures come from retrieval or from
the LLM's answer generation.

Usage:
    (make sure `uvicorn app.main:app --reload` is already running)
    python eval/run_eval.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
from pathlib import Path
import requests

from eval_db import init_eval_db, EvalSessionLocal, EvaluationRun, EvaluationResult

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"


ABSTENTION_PHRASES = [
    "don't know", "do not know", "no information", "not mentioned",
    "doesn't mention", "does not mention", "couldn't find", "could not find",
    "not discussed", "no mention", "not addressed", "unable to find",
    "doesn't contain","don't contain" , "do not contain" ,"does not contain", "not covered",
]


def _is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def run(base_url: str):
    init_eval_db()
    questions = json.loads(BENCHMARK_PATH.read_text())

    per_question_results = []

    for q in questions:
        is_adversarial = q.get("is_adversarial", False)
        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"query": q["question"], "chat_history": []},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Request failed for '{q['question']}': {e}")
            per_question_results.append({
                "question": q["question"],
                "answer_text": f"REQUEST FAILED: {e}",
                "retrieved_chunk_ids": [],
                "answer_correct": False,
                "retrieval_correct": False,
                "expected_meeting_title": q.get("expected_meeting_title"),
                "is_adversarial": is_adversarial,
            })
            continue

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        retrieved_ids = data.get("retrieved_chunk_ids", [])

        if is_adversarial:
            # "correct" here means the system admitted it doesn't know,
            # rather than confidently fabricating an answer.
            answer_correct = _is_abstention(answer)
            retrieval_correct = None  # not meaningful for adversarial questions
        else:
            answer_correct = False
            if q.get("expected_answer_contains"):
                answer_correct = q["expected_answer_contains"].lower() in answer.lower()

            retrieval_correct = False
            if q.get("expected_meeting_title"):
                cited_titles = {c["meeting_title"] for c in citations}
                retrieval_correct = q["expected_meeting_title"] in cited_titles
            else:
                retrieval_correct = len(retrieved_ids) > 0

        per_question_results.append({
            "question": q["question"],
            "answer_text": answer,
            "retrieved_chunk_ids": retrieved_ids,
            "answer_correct": answer_correct,
            "retrieval_correct": retrieval_correct,
            "expected_meeting_title": q.get("expected_meeting_title"),
            "is_adversarial": is_adversarial,
        })

    n = len(per_question_results)
    # retrieval rate only counted over non-adversarial questions
    retrieval_scored = [r for r in per_question_results if r["retrieval_correct"] is not None]
    retrieval_rate = (
        sum(r["retrieval_correct"] for r in retrieval_scored) / len(retrieval_scored)
        if retrieval_scored else 0
    )
    answer_rate = sum(r["answer_correct"] for r in per_question_results) / n if n else 0

    # --- persist to evaluation.db ---
    db = EvalSessionLocal()
    eval_run = EvaluationRun(
        retrieval_hit_rate=retrieval_rate,
        answer_accuracy=answer_rate,
        notes=f"{n} questions, base_url={base_url}",
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    run_id = eval_run.id  # capture now, before the session closes

    for r in per_question_results:
        db.add(EvaluationResult(
            run_id=run_id,
            question=r["question"],
            expected_meeting_title=r.get("expected_meeting_title"),
            retrieved_chunk_ids=",".join(str(cid) for cid in r["retrieved_chunk_ids"]),
            retrieval_correct=r["retrieval_correct"],  # may be None for adversarial rows
            answer_correct=r["answer_correct"],
            answer_text=r["answer_text"],
        ))
    db.commit()
    db.close()

    # --- print summary ---
    print(f"\nRan {n} benchmark questions (run_id={run_id})")
    print(f"Retrieval hit rate:  {retrieval_rate:.0%}  (over {len(retrieval_scored)} scorable questions)")
    print(f"Answer accuracy:     {answer_rate:.0%}\n")

    for r in per_question_results:
        status = "OK  " if r["answer_correct"] else "MISS"
        tag = " [adversarial]" if r["is_adversarial"] else ""
        print(f"[{status}]{tag} {r['question']}")
        print(f"        retrieved_chunks={r['retrieved_chunk_ids']}")
        print(f"        answer='{r['answer_text'][:100]}'")

    print(f"\nLogged to evaluation.db (run_id={run_id}) for trend tracking.")
    return per_question_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    run(args.base_url)