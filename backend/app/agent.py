import json
from anthropic import Anthropic
from .config import settings
from .tools import safe_calculate, search_workspace
from .tracing import content_length, new_request_id, write_trace
from .skills import (
    LOAD_SKILL_TOOL, discover_skills, format_skills_prompt, load_skill,
    scope_tools, EVENT_LOADED, EVENT_TOOL_SCOPING,
)

SYSTEM_PROMPT = (
    "You are AEGIS, an engineering learning and research assistant. "
    "Be accurate, explain clearly, and distinguish evidence from assumptions. "
    "Use tools when helpful. Workspace search is read-only. Never claim to have "
    "changed files or verified code unless that actually happened. If evidence "
    "is insufficient, say so."
)

TOOLS = [
    {"name": "calculator", "description": "Safely evaluate basic arithmetic.",
     "input_schema": {"type": "object", "properties": {
         "expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "search_workspace",
     "description": "Search local Markdown/text workspace files.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}}, "required": ["query"]}}
]

def _run_tool(name, args, request_id, skill_ctx):
    if name == "calculator":
        return {"result": safe_calculate(args["expression"])}
    if name == "search_workspace":
        return {"results": search_workspace(args["query"], settings.workspace_dir)}
    if name == "load_skill":
        outcome = load_skill(args["name"], skill_ctx["skills"], skill_ctx["loaded"])
        write_trace(EVENT_LOADED, request_id, reason=outcome["reason"], name=args["name"])
        if "skill" in outcome:
            skill_ctx["loaded"] = outcome["skill"]
            return {"instructions": outcome["skill"].body}
        return {"error": outcome["error"]}
    return {"error": "Unknown tool"}

def ask_aegis(message, history=None):
    request_id = new_request_id()
    client = Anthropic(api_key=settings.anthropic_api_key)
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    trace = []

    known_tools = {tool["name"] for tool in TOOLS}
    skills = discover_skills(settings.skills_dir, known_tools, request_id)
    skill_ctx = {"skills": skills, "loaded": None}
    system_prompt = SYSTEM_PROMPT
    skills_block = format_skills_prompt(skills)
    if skills_block:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{skills_block}"
    current_tools = TOOLS + [LOAD_SKILL_TOOL] if skills else TOOLS

    for _ in range(settings.max_tool_rounds):
        create_kwargs = {"model": settings.anthropic_model, "max_tokens": 1200,
                          "system": system_prompt, "messages": messages}
        if current_tools:
            # Omit `tools` entirely rather than pass [] — some SDKs/APIs treat
            # an explicit empty list differently from "no tools offered."
            create_kwargs["tools"] = current_tools
        response = client.messages.create(**create_kwargs)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            answer = "\n".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text")
            write_trace("final", request_id, stop_reason=response.stop_reason,
                         tool_calls=len(trace), answer_length=len(answer))
            return {"answer": answer, "trace": trace,
                    "stop_reason": response.stop_reason}
        loaded_before = skill_ctx["loaded"]
        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            try:
                result = _run_tool(block.name, block.input, request_id, skill_ctx)
            except Exception as exc:
                result = {"error": str(exc)}
            trace.append({"tool": block.name, "input": block.input, "result": result})
            write_trace("tool_call", request_id, tool=block.name,
                        ok="error" not in result,
                        input_length=content_length(block.input),
                        result_length=content_length(result))
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result, default=str)})
        messages.append({"role": "user", "content": tool_results})
        if loaded_before is None and skill_ctx["loaded"] is not None:
            current_tools, denied = scope_tools(TOOLS, skill_ctx["loaded"])
            write_trace(EVENT_TOOL_SCOPING, request_id,
                        reason="TOOL_DENIED" if denied else "TOOL_ALLOWED",
                        skill=skill_ctx["loaded"].name, denied_tools=denied)
    write_trace("round_limit", request_id, tool_calls=len(trace),
                max_tool_rounds=settings.max_tool_rounds)
    return {"answer": "Tool-use limit reached. Narrow the task or ask me to continue.",
            "trace": trace, "stop_reason": "tool_round_limit"}
