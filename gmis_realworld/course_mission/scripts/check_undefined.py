"""Undefined-name check with PER-FUNCTION scope.

The first version collected every assignment in a file into one set, so a name bound in
one method looked "defined" everywhere. That let a real bug through: goto_arc read
`stall_at`, which is only ever assigned in goto_bin, and the run died with
UnboundLocalError mid-mission — invisibly, because Isaac keeps spinning after the main
thread raises (CPU busy, GPU idle, log frozen), so it looked like a hang for over an hour.
"""

import ast
import builtins
import sys
from pathlib import Path

BUILT = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def bound_names(node):
    """Names bound anywhere inside this function body (params, assignments, imports)."""
    out = set()
    a = getattr(node, "args", None)
    if a is not None:
        out.update(x.arg for x in a.args + a.kwonlyargs + a.posonlyargs)
        if a.vararg:
            out.add(a.vararg.arg)
        if a.kwarg:
            out.add(a.kwarg.arg)
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((al.asname or al.name).split(".")[0] for al in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if n is not node:
                out |= bound_names(n)
        elif isinstance(n, ast.Lambda):
            b = n.args
            out.update(x.arg for x in b.args + b.kwonlyargs + b.posonlyargs)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def module_names(body):
    """Module- and class-level bindings ONLY.

    Must not descend into function bodies: walking the whole tree collected every
    method's locals, so `stall_at = 0` in one method made `stall_at` look defined in
    every other method — exactly the bug this script exists to catch.
    """
    out = set()
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(n.name)
        elif isinstance(n, ast.ClassDef):
            out.add(n.name)
            out |= module_names(n.body)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((al.asname or al.name).split(".")[0] for al in n.names)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For,
                            ast.With, ast.If, ast.Try, ast.While)):
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    out.add(x.id)
    return out


def check(node, enclosing, path, report):
    """Recurse into functions, carrying the names visible from enclosing scopes.

    A nested function legitimately reads its parent's locals (closures), so the parent's
    bindings have to travel down — otherwise every helper defined inside main() is a
    false positive.
    """
    for fn in [n for n in node.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + \
              [c for n in node.body if isinstance(n, ast.ClassDef)
               for c in n.body if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        mine = bound_names(fn)
        visible = mine | enclosing
        used = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.add(n.id)
        missing = sorted(used - visible)
        if missing:
            report.append(f"{path}:{fn.lineno} {fn.name}(): {missing}")
        check(fn, visible, path, report)


bad = 0
for f in sorted(Path("course_mission").rglob("*.py")):
    tree = ast.parse(f.read_text())
    report = []
    check(tree, module_names(tree.body) | BUILT, str(f), report)
    for line in report:
        print(line)
    bad += len(report)
print("clean" if not bad else f"{bad} function(s) with undefined names")
sys.exit(1 if bad else 0)
