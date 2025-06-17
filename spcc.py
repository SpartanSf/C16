from lark import Lark, Tree, Token

with open("c16.lark") as f:
    c16_grammar = f.read()

code = """
int arr[5] = {1, 2, 3, 4, 5};

int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

void main() {
    int total = 0;
    int i = 0;

    while (i < 5) {
        total = total + arr[i];
        i = i + 1;
    }

    return total;
}
"""

parser = Lark(c16_grammar, start="program", parser="lalr", propagate_positions=True)
tree = parser.parse(code)
lines = code.splitlines()

diagnostics = []

def r_error(node, msg, code):
    if isinstance(node, Token):
        ln, col = node.line, node.column
        length = len(node.value)
    else:
        ln, col = node.meta.line, node.meta.column
        length = 1
    diagnostics.append((ln, col, length, 'error', f"{msg} [{code}]"))

def r_warn(node, msg, code):
    if isinstance(node, Token):
        ln, col = node.line, node.column
        length = len(node.value)
    else:
        ln, col = node.meta.line, node.meta.column
        length = 1
    diagnostics.append((ln, col, length, 'warning', f"{msg} [{code}]"))

def r_note(node, msg):
    if isinstance(node, Token):
        ln, col = node.line, node.column
        length = len(node.value)
    else:
        ln, col = node.meta.line, node.meta.column
        length = 1
    diagnostics.append((ln, col, length, 'note', f"{msg}"))


class VarSym:
    def __init__(self, name, defining_tok, is_array=False, size=None):
        self.name = name
        self.defining_tok = defining_tok
        self.is_array = is_array
        self.size = size
        self.used = False


class FunSym:
    def __init__(self, name, ret_type, params, defining_tok):
        self.name = name
        self.ret_type = ret_type
        self.params = params
        self.defining_tok = defining_tok


globals_sym = {}
functions = {}

for node in tree.children:
    if node.data == "declaration":
        name_tok = next(
            c for c in node.children if isinstance(c, Token) and c.type == "NAME"
        )
        nums = [c for c in node.children if isinstance(c, Token) and c.type == "NUMBER"]
        if nums:
            globals_sym[name_tok.value] = VarSym(
                name_tok.value, name_tok, True, int(nums[0].value)
            )
        else:
            globals_sym[name_tok.value] = VarSym(name_tok.value, name_tok)

    elif node.data == "function":
        ret_tok = next(
            c
            for c in node.children
            if isinstance(c, Token) and c.type in ("INT_TYPE", "VOID_TYPE")
        )
        name_tok = next(
            c for c in node.children if isinstance(c, Token) and c.type == "NAME"
        )
        params = []
        for p in node.children:
            if isinstance(p, Tree) and p.data == "parameter":
                t_tok, n_tok = p.children
                params.append((n_tok.value, t_tok.value))
        if name_tok.value in functions:
            r_error(name_tok, f"redeclaration of `{name_tok.value}`", "E001")
            orig_fun = functions[name_tok.value]
            r_note(orig_fun.defining_tok, f"previous declaration of `{name_tok.value}` here")
        else:
            functions[name_tok.value] = FunSym(name_tok.value, ret_tok.value, params, name_tok)

    else:
        r_error(node, f"unexpected top-level `{node.data}`", "E002")

if "main" not in functions:
    dummy_tok = tree.children[0]
    r_error(dummy_tok, "missing `main` function", "E003")


def check_program(tree):
    for node in tree.children:
        if node.data == "function":
            check_function(node)


def check_function(fn_node):
    ret_tok, name_tok = fn_node.children[0], fn_node.children[1]
    fun_sym = functions.get(name_tok.value)
    scopes = (
        [{pname: VarSym(pname, None) for pname, _ in fun_sym.params}]
        if fun_sym
        else [{}]
    )
    local_vars = []

    block = next(
        (c for c in fn_node.children if isinstance(c, Tree) and c.data == "block"), None
    )
    if block:
        check_block(block, scopes, name_tok.value, local_vars)

    for sym in local_vars:
        if not sym.used:
            r_warn(sym.defining_tok, f"unused local variable `{sym.name}`", "W001")


def check_block(block_node, scopes, current_fun, local_vars):
    scopes = scopes + [{}]
    for child in block_node.children:
        stmt = child.children[0] if child.data == "statement" else child

        if stmt.data == "var_declaration":
            name_tok = stmt.children[1]
            is_array = any(
                isinstance(c, Token) and c.type == "NUMBER" for c in stmt.children
            )
            size = next(
                (
                    int(c.value)
                    for c in stmt.children
                    if isinstance(c, Token) and c.type == "NUMBER"
                ),
                None,
            )
            sym = VarSym(name_tok.value, name_tok, is_array=is_array, size=size)
            scopes[-1][name_tok.value] = sym
            local_vars.append(sym)
            if len(stmt.children) == 3:
                check_expr(stmt.children[2], scopes, current_fun)

        elif stmt.data == "assignment":
            lhs = stmt.children[0]
            rhs = stmt.children[1]

            if isinstance(lhs, Token) and lhs.type == "NAME":
                sym = find_var(lhs, scopes)
                if sym:
                    sym.used = True

                if sym and sym.is_array:
                    r_error(lhs, f"`{lhs.value}` must be indexed", "E006")
            elif isinstance(lhs, Tree) and lhs.data == "array_access":

                name_tok = lhs.children[0]
                sym = find_var(name_tok, scopes)
                if not sym or not sym.is_array:
                    r_error(name_tok, f"`{name_tok.value}` is not an array", "E005")
                else:
                    sym.used = True

                check_expr(lhs.children[1], scopes, current_fun)
            else:
                r_error(lhs, "invalid assignment target", "E008")

            check_expr(rhs, scopes, current_fun)

        elif stmt.data == "if_stmt":
            check_expr(stmt.children[0], scopes, current_fun)
            check_block(stmt.children[1], scopes, current_fun, local_vars)
            if len(stmt.children) == 3:
                check_block(stmt.children[2], scopes, current_fun, local_vars)

        elif stmt.data == "while_stmt":
            check_expr(stmt.children[0], scopes, current_fun)
            check_block(stmt.children[1], scopes, current_fun, local_vars)

        elif stmt.data == "return_stmt":
            check_expr(stmt.children[0], scopes, current_fun)

        elif stmt.data == "expr_stmt":
            check_expr(stmt.children[0], scopes, current_fun)

        elif stmt.data == "func_stmt":
            check_expr(stmt.children[0], scopes, current_fun)

        elif stmt.data == "block":
            check_block(stmt, scopes, current_fun, local_vars)

        else:
            r_error(stmt, f"unexpected statement `{stmt.data}`", "E008")


def find_var(name_tok, scopes):
    name = name_tok.value
    for sc in reversed(scopes):
        if name in sc:
            return sc[name]
    if name in globals_sym:
        return globals_sym[name]

    r_error(name_tok, f"undeclared variable `{name}`", "E007")
    return None


def check_expr(expr, scopes, current_fun):
    if isinstance(expr, Token):
        if expr.type == "NAME":
            sym = find_var(expr, scopes)
            if sym:
                sym.used = True
    else:
        if expr.data == "array_access":
            name_tok, idx = expr.children
            sym = find_var(name_tok, scopes)
            if sym and not sym.is_array:
                r_error(name_tok, f"`{name_tok.value}` is not an array", "E005")
            if sym:
                sym.used = True
            check_expr(idx, scopes, current_fun)
        elif expr.data == "func_call":
            name_tok = expr.children[0]
            if name_tok.value not in functions:
                r_error(
                    name_tok, f"call to undeclared function `{name_tok.value}`", "E009"
                )
            else:
                fn = functions[name_tok.value]
                args = expr.children[1:]
                if len(args) != len(fn.params):
                    r_error(
                        name_tok,
                        f"`{name_tok.value}` expects {len(fn.params)} args, got {len(args)}",
                        "E010",
                    )
                for a in args:
                    check_expr(a, scopes, current_fun)
        else:
            for c in expr.children:
                check_expr(c, scopes, current_fun)


check_program(tree)


kind_order = {'error': 0, 'warning': 1, 'note': 2}
diagnostics.sort(key=lambda x: (x[0], x[1], kind_order[x[3]]))

for ln, col, length, kind, msg in diagnostics:
    src = lines[ln - 1]
    ptr = " " * (col - 1) + "^" * max(length, 1)
    if kind == 'error':
        print(f"{ln:>4} | {src}\n     | {ptr} error: {msg}\n")
    elif kind == 'warning':
        print(f"{ln:>4} | {src}\n     | {ptr} warning: {msg}\n")
    else:
        print(f"{ln:>4} | {src}\n     | {ptr} note: {msg}\n")

if any(kind == 'error' for _, _, _, kind, _ in diagnostics):
    print(f"\nFound {sum(1 for d in diagnostics if d[3] == 'error')} error(s). See errors.md for more information on error codes.")
else:
    print("\nNo errors found.")
