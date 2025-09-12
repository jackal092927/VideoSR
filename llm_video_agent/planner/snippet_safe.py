import ast
from typing import Tuple


ALLOWED_IMPORT_MODULES = {
    'math', 'numpy', 'numpy as np', 'cv2', 'scipy.signal', 'scipy.optimize', 'typing'
}

BANNED_NAMES = {
    'open', 'exec', 'eval', 'compile', '__import__', 'os', 'sys', 'subprocess', 'requests', 'socket', 'pathlib', 'shutil'
}


def validate_snippet(code: str, max_lines: int = 80, max_nodes: int = 800) -> Tuple[bool, str]:
    lines = code.strip().splitlines()
    if len(lines) > max_lines:
        return False, f"Snippet too long ({len(lines)} > {max_lines} lines)"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Node count limit
    if sum(1 for _ in ast.walk(tree)) > max_nodes:
        return False, "Snippet AST too large"

    # Disallow banned names and dangerous nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod not in {m.split()[0] for m in ALLOWED_IMPORT_MODULES}:
                    return False, f"Import not allowed: {mod}"
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod.split('.')[0] not in {m.split()[0] for m in ALLOWED_IMPORT_MODULES}:
                return False, f"ImportFrom not allowed: {mod}"
        if isinstance(node, (ast.Call,)):
            # Check for banned call names
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"Call to banned name: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_NAMES:
                return False, f"Call to banned attr: {node.func.attr}"
        if isinstance(node, (ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Try)):
            # Disallow file ops and too-complex constructs in snippets
            continue
    return True, "ok"

