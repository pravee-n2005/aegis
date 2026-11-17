"""Fixed question dataset for the M5 evaluation suite, paired with the
corpus in corpus.py. Each case's `relevant` set is the hand-labeled ground
truth of which (filename, page) pairs genuinely answer the query - the
ground truth retrieval and citation metrics in metrics.py/harness.py are
judged against.

To add a case: add a CHUNKS entry to corpus.py if needed, then add an
entry here with a `query_vector` built from corpus.topic_vector() that
reflects which topic(s) the query is actually about, and a `relevant`
list of the (filename, page) pairs a correct retrieval should surface.
Keep the query text here and the query text the harness embeds in sync -
harness._fake_embed_texts looks vectors up by exact query string.
"""
from .corpus import topic_vector

CASES = [
    {
        "id": "benefits",
        "query": "What benefits are employees eligible for?",
        "query_vector": topic_vector(("benefits", 1.0)),
        "relevant": [("handbook.pdf", 1)],
    },
    {
        "id": "leave",
        "query": "How does annual leave accrual work?",
        "query_vector": topic_vector(("leave", 1.0)),
        "relevant": [("handbook.pdf", 2)],
    },
    {
        "id": "security",
        "query": "What security requirements apply to company laptops?",
        "query_vector": topic_vector(("security", 1.0)),
        "relevant": [("handbook.pdf", 3)],
    },
    {
        "id": "pm_internship_role",
        "query": "What are the responsibilities and requirements for the PM internship?",
        "query_vector": topic_vector(("responsibilities", 1.0), ("requirements", 1.0)),
        "relevant": [("jd.pdf", 2), ("jd.pdf", 3)],
    },
    {
        "id": "eligibility",
        "query": "What is the eligibility requirement for the internship?",
        "query_vector": topic_vector(("eligibility", 1.0)),
        "relevant": [("jd.pdf", 4)],
    },
    {
        "id": "expense_and_security_review",
        "query": "What is the expense reimbursement policy and how often are security policies reviewed?",
        "query_vector": topic_vector(("expense", 1.0), ("security", 1.0)),
        "relevant": [("handbook.pdf", 4), ("handbook.pdf", 5)],
    },
]
