"""CLI entry point for the M5 deterministic evaluation suite.

Usage (from the backend/ directory, with the project's venv active):

    python -m eval.run_eval
    python -m eval.run_eval --top-k 5
    python -m eval.run_eval --json

Runs the fixed-corpus retrieval evaluation (Recall@K / Precision@K / MRR)
and the mocked-LLM grounded-answer/citation check, and prints a report to
stdout. Both are fully offline and deterministic - no network access, no
API key required, no real embedding model loaded. See
docs/EVALUATION.md for what these numbers do and do not mean.

Exit code is 0 if every metric meets its fixed floor (full recall, valid
citations, full citation presence in the scripted grounded-answer check),
1 otherwise - suitable for a CI gate on retrieval regressions, distinct
from (and much cheaper than) a live-model evaluation.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import harness


def _print_retrieval_report(report):
    print(f"\n=== Retrieval evaluation (top_k={report['top_k']}, {report['case_count']} cases) ===")
    for case in report["cases"]:
        status = "OK" if case["recall_at_k"] == 1.0 else "MISS"
        print(f"  [{status:4s}] {case['id']:28s} recall@k={case['recall_at_k']:.2f} "
              f"precision@k={case['precision_at_k']:.2f} rr={case['reciprocal_rank']:.2f}")
    print(f"  mean Recall@K:    {report['mean_recall_at_k']:.3f}")
    print(f"  mean Precision@K: {report['mean_precision_at_k']:.3f}")
    print(f"  MRR:              {report['mrr']:.3f}")
    print(f"  all citations valid against corpus: {report['all_citations_valid']}")


def _print_grounded_answer_report(report):
    print(f"\n=== Grounded-answer / citation evaluation ({report['case_count']} cases, scripted LLM) ===")
    for case in report["cases"]:
        print(f"  [{case['id']:28s}] cited_correct_page={case['cited_correct_page']!s:5s} "
              f"retrieval_grounded={case['retrieval_found_relevant_chunk']!s:5s} "
              f"stop_reason={case['stop_reason']}")
    print(f"  citation presence rate:   {report['citation_presence_rate']:.3f}")
    print(f"  retrieval grounding rate: {report['retrieval_grounding_rate']:.3f}")
    print("  NOTE: the LLM here is scripted, not real - this checks the retrieval/"
          "citation pipeline end to end, not whether a real model cites well. "
          "See docs/EVALUATION.md.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="AEGIS M5 deterministic evaluation suite")
    parser.add_argument("--top-k", type=int, default=harness.DEFAULT_TOP_K,
                         help="top-k cutoff for Recall@K/Precision@K (default: %(default)s)")
    parser.add_argument("--json", action="store_true",
                         help="print machine-readable JSON instead of the table")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        retrieval_report = harness.run_retrieval_eval(tmp_path, top_k=args.top_k)
        grounded_report = harness.run_grounded_answer_eval(tmp_path)

    if args.json:
        print(json.dumps({"retrieval": retrieval_report, "grounded_answer": grounded_report}, indent=2))
    else:
        _print_retrieval_report(retrieval_report)
        _print_grounded_answer_report(grounded_report)

    ok = (retrieval_report["mean_recall_at_k"] == 1.0
          and retrieval_report["all_citations_valid"]
          and grounded_report["citation_presence_rate"] == 1.0)
    if not args.json:
        print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
