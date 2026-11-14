import json
from anthropic import Anthropic
from . import context
from . import memory
from .config import settings
from .tools import safe_calculate, search_workspace
from .tracing import content_length, new_request_id, write_trace
from .skills import (
    LOAD_SKILL_TOOL, discover_skills, format_skills_prompt, load_skill,
    scope_tools, scope_tools_by_names, EVENT_LOADED, EVENT_TOOL_SCOPING,
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

def _seed_history_if_empty(session_id, history, request_id):
    """Legacy backward-compat path: a caller that never adopts session_id
    keeps resending the full history on every call, each time against a
    brand-new session (this is the only time this runs - a session with
    messages already persisted is never reseeded)."""
    if not history or context.get_messages(session_id):
        return
    for item in history:
        try:
            role, content = item["role"], item["content"]
        except (KeyError, TypeError):
            continue
        context.append_message(session_id, role, content, request_id=request_id)

def ask_aegis(session_id, message, history=None):
    request_id = new_request_id()
    client = Anthropic(api_key=settings.anthropic_api_key)

    _seed_history_if_empty(session_id, history, request_id)
    context.append_message(session_id, "user", message, request_id=request_id)
    system_prompt_addition, messages = context.build_context(session_id, request_id=request_id)
    trace = []

    known_tools = {tool["name"] for tool in TOOLS}
    skills = discover_skills(settings.skills_dir, known_tools, request_id)
    skill_ctx = {"skills": skills, "loaded": None}
    system_prompt = SYSTEM_PROMPT
    skills_block = format_skills_prompt(skills)
    if skills_block:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{skills_block}"
    if system_prompt_addition:
        system_prompt = f"{system_prompt}\n\n{system_prompt_addition}"

    persisted_allowed_tools = context.get_skill_state(session_id)
    if persisted_allowed_tools is context.UNSET:
        current_tools = TOOLS + [LOAD_SKILL_TOOL] if skills else TOOLS
    else:
        scoped, denied = scope_tools_by_names(TOOLS, persisted_allowed_tools)
        # A persisted deny-all (or any) scope must never also deny the model
        # the ability to call load_skill and switch away - Outside Voice
        # finding 1. ALREADY_LOADED still fires correctly within this
        # request only, since skill_ctx["loaded"] always starts None below.
        current_tools = scoped + [LOAD_SKILL_TOOL] if skills else scoped

    def _finalize(answer_text, stop_reason):
        maybe_summarize_ok = context.maybe_summarize(session_id, client, request_id=request_id)
        memory.extract_memories(session_id, message, request_id=request_id)
        write_trace("final", request_id, stop_reason=stop_reason,
                     tool_calls=len(trace), answer_length=len(answer_text),
                     summarized=maybe_summarize_ok)
        return {"answer": answer_text, "trace": trace, "stop_reason": stop_reason}

    for _ in range(settings.max_tool_rounds):
        create_kwargs = {"model": settings.anthropic_model, "max_tokens": 1200,
                          "system": system_prompt, "messages": messages}
        if current_tools:
            # Omit `tools` entirely rather than pass [] — some SDKs/APIs treat
            # an explicit empty list differently from "no tools offered."
            create_kwargs["tools"] = current_tools
        response = client.messages.create(**create_kwargs)
        messages.append({"role": "assistant", "content": response.content})
        context.append_message(session_id, "assistant", response.content, request_id=request_id)
        if response.stop_reason != "tool_use":
            answer = "\n".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text")
            return _finalize(answer, response.stop_reason)
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
        context.append_message(session_id, "user", tool_results, request_id=request_id)
        if loaded_before is None and skill_ctx["loaded"] is not None:
            current_tools, denied = scope_tools(TOOLS, skill_ctx["loaded"])
            context.set_skill_state(session_id, skill_ctx["loaded"].allowed_tools, request_id=request_id)
            write_trace(EVENT_TOOL_SCOPING, request_id,
                        reason="TOOL_DENIED" if denied else "TOOL_ALLOWED",
                        skill=skill_ctx["loaded"].name, denied_tools=denied)
    write_trace("round_limit", request_id, tool_calls=len(trace),
                max_tool_rounds=settings.max_tool_rounds)
    return _finalize("Tool-use limit reached. Narrow the task or ask me to continue.",
                      "tool_round_limit")
