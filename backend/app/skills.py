"""Claude-compatible skill discovery, model-driven selection, and tool scoping.

Selection is model-driven, not an external keyword matcher: every valid
discovered skill's name+description is shown to the model on every request
(Level 1 metadata); the model decides relevance itself and fetches a skill's
full body via the `load_skill` tool (Level 2). This mirrors real Claude
Skills' progressive disclosure instead of inventing an incompatible
trigger-keyword mechanism — see docs/designs/m2-skills-runtime.md.

Discovery and parsing happen fresh on every request (no startup caching),
consistent with the rest of M1's stateless, per-request design.
"""
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .tracing import redact, write_trace

EVENT_DISCOVERED = "skill_discovered"
EVENT_LOADED = "skill_loaded"
EVENT_TOOL_SCOPING = "skill_tool_scoping"

LOAD_SKILL_TOOL = {
    "name": "load_skill",
    "description": "Load a skill's full instructions by name. Call this once you've "
                    "decided, from the Available Skills list, that a specific skill "
                    "applies to the user's request.",
    "input_schema": {"type": "object", "properties": {
        "name": {"type": "string"}}, "required": ["name"]},
}


@dataclass
class DiscoveredSkill:
    name: str
    description: str
    allowed_tools: Optional[list]  # None = unrestricted, [] = deny-all, else explicit subset
    body: str


_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?\n)---[ \t]*\r?\n(.*)\Z", re.DOTALL)


def _read_frontmatter(text: str):
    """Split a SKILL.md's leading '---' frontmatter from its Markdown body.
    Delimiters must each be alone on their own line — a naive substring split
    would misparse a description value that happens to contain the literal
    text "---". Raises ValueError if the file isn't delimited as expected."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("SKILL.md must start with '---' frontmatter delimiters, "
                          "each alone on its own line")
    return match.group(1), match.group(2).strip()


def discover_skills(skills_dir: str, known_tools: set, request_id: str) -> list:
    """Discover and parse every SKILL.md under skills_dir. Returns only valid
    skills; a malformed or unvalidatable skill is excluded and traced, never
    raised to the caller — a broken skill must never break the chat request."""
    base = Path(skills_dir)
    try:
        if not base.is_dir():
            return []
        paths = sorted(base.glob("*/SKILL.md"))
    except OSError:
        return []

    valid = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            frontmatter_text, body = _read_frontmatter(text)
            meta = yaml.safe_load(frontmatter_text)
            if not isinstance(meta, dict):
                raise ValueError("frontmatter must be a YAML mapping")
            name = meta.get("name")
            description = meta.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("'name' is required")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("'description' is required")
            allowed_tools = meta.get("allowed-tools")
            if allowed_tools is not None:
                if not isinstance(allowed_tools, list) or not all(isinstance(t, str) for t in allowed_tools):
                    raise ValueError("'allowed-tools' must be a list of tool names")
                unknown = [t for t in allowed_tools if t not in known_tools]
                if unknown:
                    write_trace(EVENT_DISCOVERED, request_id, reason="UNKNOWN_TOOL_DECLARED",
                                skill=name, path=str(path), unknown_tools=unknown)
                    continue
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
            write_trace(EVENT_DISCOVERED, request_id, reason="PARSE_ERROR",
                        path=str(path), error=redact(str(exc)))
            continue

        write_trace(EVENT_DISCOVERED, request_id, reason="OK", skill=name, path=str(path))
        valid.append(DiscoveredSkill(name=name, description=description,
                                      allowed_tools=allowed_tools, body=body))
    return valid


def format_skills_prompt(skills: list) -> str:
    """Render the Level-1 metadata block shown to the model on every request."""
    if not skills:
        return ""
    lines = ["Available skills (call load_skill(name) if one applies to this request):"]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)


def load_skill(name: str, skills: list, already_loaded) -> dict:
    """Handle a load_skill(name) tool call. Returns a dict with either
    'skill' (the matched DiscoveredSkill) or 'error', plus a 'reason' code."""
    if already_loaded is not None:
        return {"error": "A skill is already loaded for this request.", "reason": "ALREADY_LOADED"}
    for skill in skills:
        if skill.name == name:
            return {"skill": skill, "reason": "LOADED"}
    return {"error": f"No skill named {name!r} was discovered.", "reason": "LOAD_ERROR"}


def scope_tools(base_tools: list, skill: DiscoveredSkill):
    """Intersect base_tools (Anthropic tool-schema dicts) with a loaded skill's
    allowed-tools. Returns (scoped_tools, denied_names). None/omitted allowed-tools
    means unrestricted access; an empty list means deny-all."""
    if skill.allowed_tools is None:
        return base_tools, []
    allowed = set(skill.allowed_tools)
    scoped = [t for t in base_tools if t["name"] in allowed]
    denied = [t["name"] for t in base_tools if t["name"] not in allowed]
    return scoped, denied
