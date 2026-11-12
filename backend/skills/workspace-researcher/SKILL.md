---
name: workspace-researcher
description: Searches the local AEGIS workspace for notes, goals, and saved research to answer questions grounded in the user's own files, always citing which file and line the answer came from. Use when the user asks about their project notes, prior research, or anything that might already be written down in the workspace.
allowed-tools:
  - search_workspace
---

# Workspace Researcher

Answer from the user's own saved notes, not from general knowledge, whenever the workspace has something relevant.

## How to use this skill

1. Turn the user's question into one or more short search queries and call `search_workspace` with each.
2. Read the matched lines carefully before answering — a filename and line match isn't the same as the match actually answering the question.
3. When you answer, cite the exact file and line(s) the evidence came from (e.g. "per `notes.md:12`").
4. If `search_workspace` returns nothing relevant, say so explicitly rather than falling back to a general-knowledge guess framed as if it came from the workspace.

## Guardrails

- Only use the `search_workspace` tool while this skill is active. Do not attempt arithmetic or any other tool call in this mode.
- Workspace search is read-only — never claim to have changed, created, or saved anything in the workspace.
