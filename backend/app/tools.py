import ast
import operator
from pathlib import Path

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def _calculate_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _calculate_node(node.left), _calculate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent too large")
        result = _BINOPS[type(node.op)](left, right)
        if abs(result) > 10**12:
            raise ValueError("Result too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_calculate_node(node.operand))
    raise ValueError("Only basic arithmetic expressions are allowed")

def safe_calculate(expression: str):
    if len(expression) > 100:
        raise ValueError("Expression is too long")
    return _calculate_node(ast.parse(expression, mode="eval").body)

def search_workspace(query: str, workspace_dir: str, max_results: int = 5):
    base = Path(workspace_dir).resolve()
    terms = [term.lower() for term in query.split() if term.strip()]
    if not terms:
        return []
    results = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        resolved = path.resolve()
        if base not in resolved.parents:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        matches = [{"line": i + 1, "text": line.strip()}
                   for i, line in enumerate(lines)
                   if any(term in line.lower() for term in terms)]
        if matches:
            results.append({"file": str(path.relative_to(base)), "matches": matches[:3]})
        if len(results) >= max_results:
            break
    return results
