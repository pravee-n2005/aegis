# Writing a skill for AEGIS

A skill is one `SKILL.md` file under its own directory: `backend/skills/<skill-name>/SKILL.md`.

## Format

```markdown
---
name: my-skill
description: One or two sentences. This is what the model reads to decide whether
  the skill applies — be specific about when to use it, not just what it does.
allowed-tools:
  - calculator
---

# My Skill

Markdown instructions. This body is only shown to the model after it calls
`load_skill("my-skill")` — never injected up front.
```

## Fields

| Field | Required? | Notes |
|---|---|---|
| `name` | Required | Must match the directory name by convention (not enforced, but keeps `/api/skills` output sane). |
| `description` | Required | **This is the only thing the model sees before deciding to load the skill.** There is no separate "trigger keywords" field — selection is the model reading this text in context, exactly like real Claude Skills. Write it as you would explain to a colleague when to reach for this skill. |
| `allowed-tools` | Optional | A YAML list of tool names from AEGIS's base tool registry (currently `calculator`, `search_workspace`). **Omitting this field entirely** means the skill inherits full access to all base tools. **Setting it to an empty list (`allowed-tools: []`)** means deny-all — the skill gets none of the base tools. These two states are not interchangeable; pick the one you mean. |

## What this format does NOT support yet

- **No `triggers` field.** An earlier draft of this runtime tried external keyword matching against a `triggers` field; that's not how real Claude Skills select, and it's not what this parser reads. Selection is entirely the model's judgment from `description`.
- **No bundled reference files (Level 3).** A real Claude Skill can link to extra files (e.g. `references/foo.md`) that get loaded on demand. This runtime only implements Levels 1-2 (metadata always visible, full body loaded on `load_skill`) — if your skill's body links to another file, that link will go nowhere once injected as plain text. Don't rely on it yet.
- **No multi-skill composition.** At most one skill loads per chat request. A second `load_skill` call in the same request is rejected.

## How to check your skill parses

Before trusting a new skill, parse it the same way the backend does:

```python
import yaml
from pathlib import Path

text = Path("backend/skills/my-skill/SKILL.md").read_text(encoding="utf-8")
_, frontmatter, body = text.split("---", 2)
meta = yaml.safe_load(frontmatter)
print(meta["name"], meta["description"], meta.get("allowed-tools"))
```

If this raises or `meta` doesn't have `name`/`description`, the real backend will exclude your skill and log a `PARSE_ERROR` (or `UNKNOWN_TOOL_DECLARED` if `allowed-tools` names something outside the base registry) to `backend/.traces/traces.jsonl` — it won't crash the chat request, but your skill also won't be offered to the model.
