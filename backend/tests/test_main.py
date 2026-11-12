import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app import tracing  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_trace_and_skills(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr(settings, "skills_dir", str(tmp_path / "skills"))
    return tmp_path / "skills"


def _write_skill(skills_dir, dirname, name, description, allowed_tools=None):
    skill_dir = skills_dir / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = f"name: {name}\ndescription: {description}"
    if allowed_tools is not None:
        tools_yaml = "\n".join(f"  - {t}" for t in allowed_tools)
        frontmatter += f"\nallowed-tools:\n{tools_yaml}"
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nSecret body text.", encoding="utf-8")


def test_list_skills_empty_when_no_skills_directory():
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json() == {"skills": []}


def test_list_skills_happy_path(isolate_trace_and_skills):
    _write_skill(isolate_trace_and_skills, "demo", "demo-skill", "A demo skill.",
                 allowed_tools=["calculator"])
    response = client.get("/api/skills")
    assert response.status_code == 200
    body = response.json()
    assert body == {"skills": [
        {"name": "demo-skill", "description": "A demo skill.", "allowed_tools": ["calculator"]}
    ]}
    assert "Secret body text." not in response.text


def test_list_skills_degrades_to_empty_list_on_discovery_error(isolate_trace_and_skills, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated discovery failure")

    import app.main as main_module
    monkeypatch.setattr(main_module, "discover_skills", _raise)
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json() == {"skills": []}
