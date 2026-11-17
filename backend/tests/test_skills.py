import json

from app import skills, tracing

# TRACE_PATH isolation comes from conftest.py's autouse isolate_trace_file
# fixture.


def _write_skill(base, dirname, frontmatter, body="Body text.", encoding="utf-8"):
    skill_dir = base / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_bytes(
        f"---\n{frontmatter}\n---\n{body}".encode(encoding))


def _events(request_id=None):
    if not tracing.TRACE_PATH.exists():
        return []
    records = [json.loads(line) for line in tracing.TRACE_PATH.read_text().splitlines()]
    return [r for r in records if request_id is None or r["request_id"] == request_id]


KNOWN_TOOLS = {"calculator", "search_workspace"}


def test_discover_skills_valid_skill(tmp_path):
    _write_skill(tmp_path, "demo", "name: demo\ndescription: A demo skill.\nallowed-tools:\n  - calculator")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-1")
    assert len(found) == 1
    assert found[0].name == "demo"
    assert found[0].description == "A demo skill."
    assert found[0].allowed_tools == ["calculator"]
    assert found[0].body == "Body text."
    events = _events("req-1")
    assert len(events) == 1
    assert events[0]["event"] == skills.EVENT_DISCOVERED
    assert events[0]["reason"] == "OK"
    assert events[0]["skill"] == "demo"


def test_discover_skills_malformed_frontmatter(tmp_path):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("no frontmatter delimiters here", encoding="utf-8")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-2")
    assert found == []
    events = _events("req-2")
    assert len(events) == 1
    assert events[0]["event"] == skills.EVENT_DISCOVERED
    assert events[0]["reason"] == "PARSE_ERROR"


def test_discover_skills_description_containing_triple_dash(tmp_path):
    """Regression: a description value that legitimately contains the literal
    text "---" must not be mistaken for the closing frontmatter delimiter."""
    _write_skill(tmp_path, "dashy", "name: dashy\ndescription: Use for A---B workflows",
                 body="actual body")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-dashy")
    assert len(found) == 1
    assert found[0].description == "Use for A---B workflows"
    assert found[0].body == "actual body"


def test_discover_skills_content_before_opening_delimiter(tmp_path):
    skill_dir = tmp_path / "preamble"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "some preamble text\n---\nname: x\ndescription: y\n---\nbody", encoding="utf-8")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-2b")
    assert found == []
    assert _events("req-2b")[0]["reason"] == "PARSE_ERROR"


def test_discover_skills_invalid_yaml(tmp_path):
    _write_skill(tmp_path, "badyaml", "name: [unclosed")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-3")
    assert found == []
    assert _events("req-3")[0]["reason"] == "PARSE_ERROR"


def test_discover_skills_missing_required_field(tmp_path):
    _write_skill(tmp_path, "nodesc", "name: nodesc")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-4")
    assert found == []
    assert _events("req-4")[0]["reason"] == "PARSE_ERROR"


def test_discover_skills_unknown_allowed_tool(tmp_path):
    _write_skill(tmp_path, "badtools",
                 "name: badtools\ndescription: Uses a tool that doesn't exist.\n"
                 "allowed-tools:\n  - nonexistent_tool")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-5")
    assert found == []
    event = _events("req-5")[0]
    assert event["reason"] == "UNKNOWN_TOOL_DECLARED"
    assert event["unknown_tools"] == ["nonexistent_tool"]


def test_discover_skills_non_utf8_file(tmp_path):
    skill_dir = tmp_path / "nonutf8"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_bytes(b"---\nname: x\n---\n\xff\xfe invalid utf8")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-6")
    assert found == []
    assert _events("req-6")[0]["reason"] == "PARSE_ERROR"


def test_discover_skills_missing_directory(tmp_path):
    found = skills.discover_skills(str(tmp_path / "does-not-exist"), KNOWN_TOOLS, "req-7")
    assert found == []
    assert _events("req-7") == []


def test_discover_skills_empty_directory(tmp_path):
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-8")
    assert found == []
    assert _events("req-8") == []


def test_discover_skills_two_skills_discovery_order(tmp_path):
    _write_skill(tmp_path, "b-skill", "name: b-skill\ndescription: Second alphabetically... no, first.")
    _write_skill(tmp_path, "a-skill", "name: a-skill\ndescription: Comes first alphabetically.")
    found = skills.discover_skills(str(tmp_path), KNOWN_TOOLS, "req-9")
    assert [s.name for s in found] == ["a-skill", "b-skill"]


def test_shipped_skills_parse_successfully():
    """The real production fixtures under backend/skills/ must actually parse —
    unlike the synthetic tmp_path fixtures above, nothing else regression-tests
    these specific files, so a typo here would silently drop a skill in
    production (excluded + traced, not raised) with no test failure."""
    from pathlib import Path
    real_skills_dir = Path(__file__).resolve().parent.parent / "skills"
    found = skills.discover_skills(str(real_skills_dir), KNOWN_TOOLS, "req-shipped")
    names = {s.name for s in found}
    assert names == {"math-tutor", "workspace-researcher"}
    by_name = {s.name: s for s in found}
    assert by_name["math-tutor"].allowed_tools == ["calculator"]
    assert by_name["workspace-researcher"].allowed_tools == ["search_workspace"]


def test_format_skills_prompt_empty():
    assert skills.format_skills_prompt([]) == ""


def test_format_skills_prompt_lists_name_and_description_not_body():
    skill = skills.DiscoveredSkill(name="demo", description="Does a thing.",
                                    allowed_tools=None, body="SECRET INSTRUCTIONS")
    prompt = skills.format_skills_prompt([skill])
    assert "demo" in prompt
    assert "Does a thing." in prompt
    assert "SECRET INSTRUCTIONS" not in prompt


def test_load_skill_happy_path():
    skill = skills.DiscoveredSkill(name="demo", description="d", allowed_tools=None, body="instructions")
    outcome = skills.load_skill("demo", [skill], already_loaded=None)
    assert outcome["reason"] == "LOADED"
    assert outcome["skill"] is skill


def test_load_skill_name_mismatch():
    skill = skills.DiscoveredSkill(name="demo", description="d", allowed_tools=None, body="x")
    outcome = skills.load_skill("nonexistent", [skill], already_loaded=None)
    assert outcome["reason"] == "LOAD_ERROR"
    assert "skill" not in outcome


def test_load_skill_already_loaded_rejects_second_call():
    skill = skills.DiscoveredSkill(name="demo", description="d", allowed_tools=None, body="x")
    outcome = skills.load_skill("demo", [skill], already_loaded=skill)
    assert outcome["reason"] == "ALREADY_LOADED"
    assert "skill" not in outcome


BASE_TOOLS = [{"name": "calculator"}, {"name": "search_workspace"}]


def test_scope_tools_omitted_allowed_tools_grants_full_access():
    skill = skills.DiscoveredSkill(name="demo", description="d", allowed_tools=None, body="x")
    scoped, denied = skills.scope_tools(BASE_TOOLS, skill)
    assert scoped == BASE_TOOLS
    assert denied == []


def test_scope_tools_empty_list_denies_all():
    skill = skills.DiscoveredSkill(name="demo", description="d", allowed_tools=[], body="x")
    scoped, denied = skills.scope_tools(BASE_TOOLS, skill)
    assert scoped == []
    assert sorted(denied) == ["calculator", "search_workspace"]


def test_scope_tools_partial_allowlist():
    skill = skills.DiscoveredSkill(name="demo", description="d", allowed_tools=["calculator"], body="x")
    scoped, denied = skills.scope_tools(BASE_TOOLS, skill)
    assert scoped == [{"name": "calculator"}]
    assert denied == ["search_workspace"]
