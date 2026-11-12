import json
from anthropic import Anthropic
from .config import settings
from .tools import safe_calculate, search_workspace

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

def _run_tool(name, args):
    if name == "calculator":
        return {"result": safe_calculate(args["expression"])}
    if name == "search_workspace":
        return {"results": search_workspace(args["query"], settings.workspace_dir)}
    return {"error": "Unknown tool"}

def ask_aegis(message, history=None):
    client = Anthropic(api_key=settings.anthropic_api_key)
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    trace = []
    for _ in range(settings.max_tool_rounds):
        response = client.messages.create(
            model=settings.anthropic_model, max_tokens=1200,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            answer = "\n".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text")
            return {"answer": answer, "trace": trace,
                    "stop_reason": response.stop_reason}
        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            try:
                result = _run_tool(block.name, block.input)
            except Exception as exc:
                result = {"error": str(exc)}
            trace.append({"tool": block.name, "input": block.input, "result": result})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result, default=str)})
        messages.append({"role": "user", "content": tool_results})
    return {"answer": "Tool-use limit reached. Narrow the task or ask me to continue.",
            "trace": trace, "stop_reason": "tool_round_limit"}
