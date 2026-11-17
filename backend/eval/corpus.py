"""Fixed, hand-authored document corpus for the M5 deterministic
evaluation suite.

This is NOT real PDF content run through the real extraction/chunking/
embedding pipeline - it is a small, fully-labeled fixture loaded directly
into the chunks table (see harness.py), the same pattern
tests/test_agent.py's `_insert_chunk_row` helper already uses. Each chunk
carries a hand-crafted "topic vector" instead of a sentence-transformers
embedding, so retrieval ranking is exact and reproducible without a
model download or any network access.

Two documents, 9 chunks (one page each) total:
- handbook.pdf: 5 pages (benefits, leave, security, expense, a page that
  blends security+expense)
- jd.pdf: 4 pages (overview, responsibilities, requirements, eligibility)
"""

TOPICS = [
    "benefits", "leave", "security", "expense",
    "overview", "responsibilities", "requirements", "eligibility",
]


def topic_vector(*weighted_topics):
    """weighted_topics: (topic_name, weight) pairs. Topics not listed get
    weight 0. Orthogonal one-hot-style vectors give exact, tie-free cosine
    rankings on this fixture; a page that blends two topics (handbook.pdf
    page 5) gets nonzero weight on both axes, same as a real document page
    that genuinely covers two subjects."""
    vec = [0.0] * len(TOPICS)
    for topic, weight in weighted_topics:
        vec[TOPICS.index(topic)] = weight
    return vec


# (filename, page, text, embedding_vector)
CHUNKS = [
    ("handbook.pdf", 1,
     "Employees are eligible for health and dental benefits after 90 days of employment.",
     topic_vector(("benefits", 1.0))),
    ("handbook.pdf", 2,
     "Annual leave accrues at 1.5 days per month and must be approved by a manager.",
     topic_vector(("leave", 1.0))),
    ("handbook.pdf", 3,
     "All laptops must be encrypted and VPN must be used when accessing internal systems.",
     topic_vector(("security", 1.0))),
    ("handbook.pdf", 4,
     "Expense reports must be submitted within 30 days with original receipts.",
     topic_vector(("expense", 1.0))),
    ("handbook.pdf", 5,
     "Security and expense policies are reviewed annually by the compliance team.",
     topic_vector(("security", 0.7), ("expense", 0.7))),
    ("jd.pdf", 1,
     "This role is a Product Management internship on the PM1 team.",
     topic_vector(("overview", 1.0))),
    ("jd.pdf", 2,
     "Responsibilities include conducting user research, writing specs, and coordinating with engineering.",
     topic_vector(("responsibilities", 1.0))),
    ("jd.pdf", 3,
     "Requirements include strong communication skills and familiarity with agile methodologies.",
     topic_vector(("requirements", 1.0))),
    ("jd.pdf", 4,
     "Eligibility requires enrollment in a degree program graduating within 12 months.",
     topic_vector(("eligibility", 1.0))),
]
