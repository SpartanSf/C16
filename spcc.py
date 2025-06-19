#!/usr/bin/env python3

from lark import Lark, Tree, Token

with open("c16.lark") as f:
    c16_grammar = f.read()

code = """
typedef int* intptr;

struct Point {
    int x;
    int y;
};

int add(int a, int b) {
    return a + b;
}

int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    int total = 0;
    int val = 2;

    for (int i = 0; i < 5; i = i + 1) {
        total = total + arr[i];
    }

    int *ptr = &total;
    *ptr = *ptr + 10;

    struct Point p1;
    p1.x = 3;
    p1.y = 4;

    struct Point* pptr = &p1;
    pptr->x = pptr->x + 1;

    int result = factorial(5);

    switch (val) {
        case 1:
            result = result + 100;
            break;
        case 2:
            result = result + 200;
            break;
        default:
            result = result + 300;
    }

    while (result > 1000) {
        result = result - 100;
        if (result == 900) {
            break;
        }
    }

    return result;
}
"""

parser = Lark(
    c16_grammar,
    start="program",
    parser="earley",
    lexer="dynamic_complete",
    propagate_positions=True,
)
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
    diagnostics.append((ln, col, length, "error", f"{msg} [{code}]"))


def r_warn(node, msg, code):
    if isinstance(node, Token):
        ln, col = node.line, node.column
        length = len(node.value)
    else:
        ln, col = node.meta.line, node.meta.column
        length = 1
    diagnostics.append((ln, col, length, "warning", f"{msg} [{code}]"))


def r_note(node, msg):
    if isinstance(node, Token):
        ln, col = node.line, node.column
        length = len(node.value)
    else:
        ln, col = node.meta.line, node.meta.column
        length = 1
    diagnostics.append((ln, col, length, "note", f"{msg}"))


class VarSym:

    def __init__(
        self,
        name,
        defining_tok,
        is_array=False,
        size=None,
        is_pointer=False,
        base_type="int",
    ):
        self.name = name
        self.defining_tok = defining_tok
        self.is_array = is_array
        self.size = size
        self.is_pointer = is_pointer
        self.base_type = base_type
        self.used = False


class FunSym:

    def __init__(self, name, ret_type, params, defining_tok):
        self.name = name
        self.ret_type = ret_type
        self.params = params
        self.defining_tok = defining_tok


class StructSym:

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields


class TypedefSym:

    def __init__(self, alias, actual_type):
        self.alias = alias
        self.actual_type = actual_type


loop_depth = 0
switch_depth = 0


def enter_loop():
    global loop_depth
    loop_depth += 1


def exit_loop():
    global loop_depth
    loop_depth -= 1


def in_loop():
    return loop_depth > 0


def enter_switch():
    global switch_depth
    switch_depth += 1


def exit_switch():
    global switch_depth
    switch_depth -= 1


def in_loop_or_switch():
    return loop_depth > 0 or switch_depth > 0


globals_sym = {}
functions = {}
typedefs = {}
structs = {}


def get_param_name(param_name_node):
    if isinstance(param_name_node, Token):
        return param_name_node
    elif isinstance(param_name_node, Tree):
        for c in param_name_node.children:
            if isinstance(c, Token) and c.type == "NAME":
                return c
            elif isinstance(c, Tree):
                token = get_param_name(c)
                if token is not None:
                    return token
    return None


def get_name_token(node):
    """
    Given a Token or Tree, descend until you find
    an inner Token of type NAME (or return None).
    """
    if isinstance(node, Token) and node.type == "NAME":
        return node
    if isinstance(node, Tree):
        for child in node.children:
            tok = get_name_token(child)
            if tok is not None:
                return tok
    return None


def flatten_type(t):
    ptrs = sum(1 for c in getattr(t, "children", [])
               if isinstance(c, Token) and c.type == "STAR")

    if isinstance(t, Token):
        base = t.value.lower()
    else:
        first = t.children[0]
        if isinstance(first, Token) and first.type == "STRUCT":
            name_tok = t.children[1]
            base = f"struct {name_tok.value}"
        elif isinstance(first, Tree) and first.data == "base_type":
            btok = first.children[0]
            if btok.type == "STRUCT":
                name_tok = first.children[1]
                base = f"struct {name_tok.value}"
            else:
                base = btok.value.lower()
        elif isinstance(first, Token):
            base = first.value.lower()
        else:
            base = first.type.lower()

    star_str = "*" * ptrs
    type_str = base + star_str
    if type_str in typedefs:
        type_str = typedefs[type_str].actual_type

    return base, star_str


def handle_inline_decl(decl_node, scopes, local_vars, current_fun):
    type_tree = decl_node.children[0]
    name_node = decl_node.children[1]
    name_tok = get_name_token(name_node)
    if not name_tok:
        r_error(decl_node, "Could not find variable name token in inline declaration", "E016")
        return

    nums = [c for c in decl_node.children if isinstance(c, Token) and c.type == "NUMBER"]
    is_array = len(nums) > 0
    size = int(nums[0].value) if is_array else None

    btype, ptrs = flatten_type(type_tree)
    sym = VarSym(
        name_tok.value,
        name_tok,
        is_array=is_array,
        size=size,
        is_pointer=(len(ptrs) > 0),
        base_type=btype,
    )

    if name_tok.value in scopes[-1]:
        r_error(name_tok, f"redeclaration of variable `{name_tok.value}`", "E004")
    else:
        scopes[-1][name_tok.value] = sym
    local_vars.append(sym)

    if len(decl_node.children) > 2:
        init_expr = decl_node.children[-1]
        check_expr(init_expr, scopes, current_fun)


def is_scalar_type(typ):
    """Return True if `typ` is a scalar type (e.g., int, char, pointer)."""
    return typ in ("int", "char") or (isinstance(typ, str) and typ.endswith("*"))


def infer_expr_type(expr, scopes):
    """
    Return a string representation of expr’s type, e.g.
      - "int"
      - "int*"
      - "int[5]"
      - "struct Point"
      - "struct Point*"
    """
    if isinstance(expr, Tree) and len(expr.children) == 1:
        return infer_expr_type(expr.children[0], scopes)

    if isinstance(expr, Tree) and expr.data == "var":
        return infer_expr_type(expr.children[0], scopes)

    if isinstance(expr, Token) and expr.type == "NAME":
        sym = find_var(expr, scopes)
        if not sym:
            return "int"
        if isinstance(sym, VarSym):
            if sym.is_array:
                return f"{sym.base_type}[{sym.size}]"
            if sym.is_pointer:
                return sym.base_type + "*"
            return sym.base_type

    if isinstance(expr, Tree) and expr.data == "array_access":
        sym = find_var(get_base_var_token(expr), scopes)
        if isinstance(sym, VarSym) and sym.is_array:
            return sym.base_type
        t = infer_expr_type(expr.children[0], scopes)
        return t[:-1] if t.endswith("*") else t

    if isinstance(expr, Tree) and expr.data == "deref":
        inner_t = infer_expr_type(expr.children[0], scopes)
        return inner_t[:-1] if inner_t.endswith("*") else inner_t

    if isinstance(expr, Tree) and expr.data == "addr_of":
        inner_t = infer_expr_type(expr.children[0], scopes)
        return inner_t + "*"

    if isinstance(expr, Tree) and expr.data == "member_access":
        base = expr.children[0]
        base_t = infer_expr_type(base, scopes)
        fld_tok = get_name_token(expr)
        if base_t.startswith("struct ") and not base_t.endswith("*") and fld_tok:
            struct_name = base_t.split()[1]
            struct_sym = structs.get(struct_name)
            if struct_sym and fld_tok.value in struct_sym.fields:
                return struct_sym.fields[fld_tok.value]
        return "int"

    if isinstance(expr, Tree) and expr.data == "ptr_access":
        base = expr.children[0]
        base_t = infer_expr_type(base, scopes)
        fld_tok = get_name_token(expr)
        if base_t.startswith("struct ") and base_t.endswith("*") and fld_tok:
            struct_name = base_t.split()[1][:-1]
            struct_sym = structs.get(struct_name)
            if struct_sym and fld_tok.value in struct_sym.fields:
                return struct_sym.fields[fld_tok.value]
        return "int"

    if isinstance(expr, Tree) and expr.data == "func_call":
        name_tok = get_name_token(expr.children[0])
        fn = functions.get(name_tok.value) if name_tok else None
        return fn.ret_type if fn else "int"

    if isinstance(expr, Tree) and expr.data in {
        "add", "sub", "mul", "div",
        "eq", "ne", "lt", "gt", "le", "ge",
        "and_", "or_", "not_"
    }:
        return "int"

    if isinstance(expr, Tree) and expr.data in ("number", "string"):
        return "int"

    return "int"


for node in tree.children:
    if node.data == "declaration":
        if node.data == "declaration":
            type_tree = node.children[0]
            name_tok = next(c for c in node.children if isinstance(c, Token) and c.type == "NAME")
            nums = [c for c in node.children if isinstance(c, Token) and c.type == "NUMBER"]
            is_array = bool(nums)
            size = int(nums[0].value) if nums else None
            btype, ptrs = flatten_type(type_tree)
            globals_sym[name_tok.value] = VarSym(
                name_tok.value,
                name_tok,
                is_array=is_array,
                size=size,
                is_pointer=bool(ptrs),
                base_type=btype
            )

    elif node.data == "function":
        ret_type_tree = node.children[0]
        name_tok = node.children[1]

        base_type, stars = flatten_type(ret_type_tree)
        ret_type_str = base_type + stars

        params = []

        for p in node.children:
            if isinstance(p, Tree) and p.data == "parameter":
                param_type, param_name_node = p.children
                btype, ptrs = flatten_type(param_type)
                param_name_tok = get_param_name(param_name_node)
                if param_name_tok is None:
                    r_error(p, "Could not find parameter name token", "E017") 
                    continue
                params.append((param_name_tok.value, btype + ptrs))

        if name_tok.value in functions:
            r_error(name_tok, f"redeclaration of `{name_tok.value}`", "E001")
            r_note(
                functions[name_tok.value].defining_tok,
                f"previous declaration of `{name_tok.value}` here",
            )
        else:
            functions[name_tok.value] = FunSym(name_tok.value, ret_type_str, params, name_tok)

    elif node.data == "typedef":
        type_tree = node.children[1]
        alias_node = node.children[2]
        alias_tok = get_name_token(alias_node)
        if alias_tok is None:
            r_error(alias_node, "Could not find typedef alias name token", "E500")
            continue
        base_type, stars = flatten_type(type_tree)
        typedefs[alias_tok.value] = TypedefSym(alias_tok.value, base_type + stars)

    elif node.data == "struct_def":
        fields = {}
        name_tok = node.children[1]
        field_block = node.children[2]

        for field_decl in field_block.children:
            if not (isinstance(field_decl, Tree) and field_decl.data == "declaration"):
                continue
            ftok = get_name_token(field_decl.children[1])
            btype, ptrs = flatten_type(field_decl.children[0])
            fields[ftok.value] = btype + ptrs

        structs[name_tok.value] = StructSym(name_tok.value, fields)

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
    _, name_tok = fn_node.children[0], fn_node.children[1]
    fun_sym = functions.get(name_tok.value)
    scopes = [{
        pname: VarSym(pname, None, is_pointer=ptype.endswith("*"), base_type=ptype.rstrip("*"))
        for pname, ptype in fun_sym.params
    }]
    local_vars = []

    block = next((c for c in fn_node.children if isinstance(c, Tree) and c.data == "block"), None)
    if block:
        check_block(block, scopes, name_tok.value, local_vars)

    for sym in local_vars:
        if not sym.used:
            r_warn(sym.defining_tok, f"unused local variable `{sym.name}`", "W001")


def check_block(block_node, scopes, current_fun, local_vars):

    if block_node is None:
        return

    if not (isinstance(block_node, Tree) and block_node.data == "block"):
        block_node = Tree("block", [block_node])

    scopes = scopes + [{}]

    for child in block_node.children:

        stmt = child
        if isinstance(child, Tree) and child.data == "statement":
            stmt = child.children[0]

        if stmt.data == "var_declaration":
            type_tree = stmt.children[0]
            name_node = stmt.children[1]
            name_tok = get_name_token(name_node)
            if not name_tok:
                r_error(stmt, "Could not find variable name token in declaration", "E018")
                continue

            nums = [c for c in stmt.children if isinstance(c, Token) and c.type == "NUMBER"]
            is_array = len(nums) > 0
            size = int(nums[0].value) if is_array else None

            btype, ptrs = flatten_type(type_tree)
            sym = VarSym(
                name_tok.value,
                name_tok,
                is_array=is_array,
                size=size,
                is_pointer=(len(ptrs) > 0),
                base_type=btype,
            )
            if name_tok.value in scopes[-1]:
                r_error(name_tok, f"redeclaration of variable `{name_tok.value}`", "E004")
            else:
                scopes[-1][name_tok.value] = sym
            local_vars.append(sym)

            if len(stmt.children) > 2:

                init_expr = stmt.children[-1]
                check_expr(init_expr, scopes, current_fun)

        elif stmt.data == "if_stmt":

            check_expr(stmt.children[0], scopes, current_fun)
            check_block(stmt.children[1], scopes, current_fun, local_vars)
            if len(stmt.children) == 3:
                check_block(stmt.children[2], scopes, current_fun, local_vars)

        elif stmt.data == "while_stmt":
            check_expr(stmt.children[0], scopes, current_fun)
            enter_loop()
            check_block(stmt.children[1], scopes, current_fun, local_vars)
            exit_loop()

        elif stmt.data == "return_stmt":

            if len(stmt.children) > 0:
                check_expr(stmt.children[0], scopes, current_fun)

        elif stmt.data == "expr_stmt":
            check_expr(stmt.children[0], scopes, current_fun)

        elif stmt.data == "func_stmt":

            check_expr(stmt.children[0], scopes, current_fun)

        elif stmt.data == "break_stmt":
            if not in_loop_or_switch():
                r_error(stmt, "`break` not within loop or switch", "E011")

        elif stmt.data == "continue_stmt":
            if not in_loop():
                r_error(stmt, "`continue` not within loop", "E012")

        elif stmt.data == "switch_stmt":

            discrim = stmt.children[0]
            check_expr(discrim, scopes, current_fun)
            enter_switch()

            for c in stmt.children[1:]:
                if c.data == "case_block":

                    for s in c.children[1:]:
                        check_block(Tree("block", [s]), scopes, current_fun, local_vars)
                elif c.data == "default_block":
                    for s in c.children[1:]:
                        check_block(Tree("block", [s]), scopes, current_fun, local_vars)

            exit_switch()

        elif stmt.data == "for_stmt":
            control = None
            body = None
            for c in stmt.children:
                if isinstance(c, Tree):
                    if c.data == "for_control":
                        control = c
                    else:
                        body = c
            if control is None or body is None:
                r_error(stmt, "Malformed `for` statement", "E019")
                continue

            init = cond = post = None
            for c in control.children:
                if isinstance(c, Tree):
                    if c.data == "inline_declaration":
                        init = c
                    elif c.data == "expression":
                        if cond is None:
                            cond = c
                        else:
                            post = Tree("expression_list", [c])
                    elif c.data == "expression_list":
                        post = c

            scopes.append({})
            enter_loop()
            if init:
                handle_inline_decl(init, scopes, local_vars, current_fun)
            if cond:
                check_expr(cond, scopes, current_fun)
                cond_type = infer_expr_type(cond, scopes)
                if not is_scalar_type(cond_type):
                    r_error(cond, f"for loop condition must be scalar, got {cond_type}", "E015")
            if post:
                for expr in post.children:
                    check_expr(expr, scopes, current_fun)
            check_block(body, scopes, current_fun, local_vars)
            exit_loop()
            scopes.pop()

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


def get_base_var_token(expr):
    if isinstance(expr, Token) and expr.type == "NAME":
        return expr
    if isinstance(expr, Tree):
        if expr.data == "var":
            return expr.children[0]
        if expr.data == "array_access":
            return get_base_var_token(expr.children[0])
    return None


def check_expr(expr, scopes, current_fun):
    if isinstance(expr, Token):
        if expr.type in {"BREAK", "CONTINUE"} or expr.value in {"break", "continue"}:
            return
        if expr.type == "NAME":
            sym = find_var(expr, scopes)
            if sym:
                sym.used = True
                parent = getattr(expr, "parent", None)
                if sym.is_array and not isinstance(parent, Tree) or (parent and parent.data not in {"array_access", "addr_of"}):
                    r_error(expr, f"array variable `{sym.name}` used without indexing", "E006")

    if isinstance(expr, Tree) and expr.data == "array_access":
        check_expr(expr.children[0], scopes, current_fun)
        check_expr(expr.children[1], scopes, current_fun)

    elif isinstance(expr, Tree) and expr.data == "member_access":
        base = expr.children[0]
        fld_tok = expr.children[2]
        check_expr(base, scopes, current_fun)
        base_type = infer_expr_type(base, scopes)
        if not base_type.startswith("struct "):
            r_error(base, f"Cannot access field of non-struct type `{base_type}`", "E013")
        else:
            struct_name = base_type.split()[1]
            struct_sym = structs.get(struct_name)
            if not struct_sym or fld_tok.value not in struct_sym.fields:
                r_error(fld_tok, f"`{struct_name}` has no field `{fld_tok.value}`", "E014")
                
    elif isinstance(expr, Tree) and expr.data == "ptr_access":
        base = expr.children[0]
        fld_tok = expr.children[2]
        check_expr(base, scopes, current_fun)
        base_type = infer_expr_type(base, scopes)
        if not (base_type.startswith("struct ") and base_type.endswith("*")):
            r_error(base, f"Cannot deref field from non-pointer-to-struct `{base_type}`", "E013")
        else:
            struct_name = base_type.split()[1][:-1]
            struct_sym = structs.get(struct_name)
            if not struct_sym or fld_tok.value not in struct_sym.fields:
                r_error(fld_tok, f"`{struct_name}` has no field `{fld_tok.value}`", "E014")

    elif isinstance(expr, Tree) and expr.data == "func_call":
        name_tok = get_name_token(expr.children[0])
        if name_tok is None:
            r_error(expr, "Could not find function name in call", "E021")
            return
        if name_tok.value not in functions:
            r_error(name_tok, f"call to undeclared function `{name_tok.value}`", "E009")
        else:
            fn = functions[name_tok.value]
            args = expr.children[1:]
            if len(args) != len(fn.params):
                r_error(
                    name_tok,
                    f"`{name_tok.value}` expects {len(fn.params)} args, got {len(args)}`",
                    "E010",
                )
            for a in args:
                check_expr(a, scopes, current_fun)

    else:
        for c in getattr(expr, "children", []):
            check_expr(c, scopes, current_fun)


check_program(tree)

kind_order = {"error": 0, "warning": 1, "note": 2}
diagnostics.sort(key=lambda x: (x[0], x[1], kind_order[x[3]]))

for ln, col, length, kind, msg in diagnostics:
    src = lines[ln - 1]
    ptr = " " * (col - 1) + "^" * max(length, 1)
    if kind == "error":
        print(f"{ln:>4} | {src}\n     | {ptr} error: {msg}\n")
    elif kind == "warning":
        print(f"{ln:>4} | {src}\n     | {ptr} warning: {msg}\n")
    else:
        print(f"{ln:>4} | {src}\n     | {ptr} note: {msg}\n")

if any(kind == "error" for _, _, _, kind, _ in diagnostics):
    print(f"\nFound {sum(1 for d in diagnostics if d[3] == 'error')} error(s). See errors.md for more information on error codes.")
else:
    print("\nNo errors found.")

print("Compiling...")




class CodeGenerator:
    def __init__(self):
        self.output = []
        self.label_count = 0
        self.data_section = []
        self.current_function = None
        self.local_offsets = {}
        self.local_size = 0
        self.param_count = 0
        self.loop_exit_stack = []
        self.loop_start_stack = []
        self.switch_stack = []
        self.frame_pointer_reg = 'r11'
        self.stack_pointer_reg = 'sp'
        self.return_value_reg = 'r1'

    def new_label(self, prefix):
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    def emit(self, instruction, comment=None):
        if comment:
            self.output.append(f"{instruction:20} ; {comment}")
        else:
            self.output.append(instruction)

    def emit_data(self, name, value):
        self.data_section.append(f"{name}:")
        if isinstance(value, list):
            for val in value:
                self.data_section.append(f"    #d {val}")
        else:
            self.data_section.append(f"    #d {value}")

    def function_prologue(self, name, param_count):
        self.emit(f"{name}:")
        self.current_function = name
        self.param_count = param_count
        self.emit(f"push {self.frame_pointer_reg}", "Save frame pointer")
        self.emit(f"mov {self.frame_pointer_reg}, {self.stack_pointer_reg}", "Set new frame pointer")
        if self.local_size > 0:
            self.emit(f"sub {self.stack_pointer_reg}, {self.local_size}", "Allocate space for locals")

    def function_epilogue(self):
        if self.local_size > 0:
            self.emit(f"add {self.stack_pointer_reg}, {self.local_size}", "Deallocate locals")
        self.emit(f"mov {self.stack_pointer_reg}, {self.frame_pointer_reg}", "Restore stack pointer")
        self.emit(f"pop {self.frame_pointer_reg}", "Restore frame pointer")
        self.emit("rts", "Return from function")

    def generate_return(self, stmt, scopes, current_fun):
        if stmt.children:
            self.generate_expr(stmt.children[0], scopes, current_fun)
            if self.return_value_reg != 'r1':
                self.emit(f"mov {self.return_value_reg}, r1", "Set return value")
        self.emit(f"jmp _{self.current_function}_exit", "Jump to function exit")

    def generate_expr(self, expr, scopes, current_fun):
        if isinstance(expr, Token) and expr.type == "NUMBER":
            self.emit(f"mov r1, {expr.value}", "Load constant")
        elif isinstance(expr, Token) and expr.type == "NAME":
            var_name = expr.value
            if var_name in self.local_offsets:
                offset = self.local_offsets[var_name]
                self.emit(f"lod r1, [r11+{offset}]", f"Load {var_name}")
            else:
                self.emit(f"lod r1, [{var_name}]", f"Load global {var_name}")
        elif isinstance(expr, Tree) and expr.data == "add":
            self.generate_binop(expr, "add", scopes, current_fun)
        else:
            for child in expr.children:
                if isinstance(child, (Tree, Token)):
                    self.generate_expr(child, scopes, current_fun)

    def generate_binop(self, expr, op, scopes, current_fun):
        self.generate_expr(expr.children[0], scopes, current_fun)
        self.emit("push r1", "Save left operand")
        self.generate_expr(expr.children[1], scopes, current_fun)
        self.emit("pop r2", "Restore left operand")
        self.emit(f"{op} r2, r1", f"{op} operation")
        self.emit("mov r1, r2", "Save result")

    def generate_var_decl(self, stmt, scopes, current_fun):
        name_tok = get_name_token(stmt.children[1])
        name = name_tok.value
        self.local_size += 2
        offset = -self.local_size
        self.local_offsets[name] = offset

        if len(stmt.children) > 2:
            self.generate_expr(stmt.children[2], scopes, current_fun)
            self.emit(f"sto [r11+{offset}], r1", f"Initialize {name}")

    def generate_block(self, block_node, scopes, current_fun, local_vars):
        for child in block_node.children:
            stmt = child
            if isinstance(child, Tree) and child.data == "statement":
                stmt = child.children[0]
            
            if stmt.data == "var_declaration":
                self.generate_var_decl(stmt, scopes, current_fun)
            elif stmt.data == "return_stmt":
                self.generate_return(stmt, scopes, current_fun)

    def generate_function(self, node):
        ret_type_tree = node.children[0]
        name_tok = node.children[1]
        name = name_tok.value

        parameters = None
        block = None
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "parameters":
                    parameters = child
                elif child.data == "block":
                    block = child

        param_count = len(parameters.children) if parameters else 0

        local_vars = []
        self.local_offsets = {}
        self.local_size = 0
        if block:
            for child in block.children:
                if isinstance(child, Tree) and child.data == "statement":
                    stmt = child.children[0]
                    if stmt.data == "var_declaration":
                        name_tok = get_name_token(stmt.children[1])
                        local_vars.append(name_tok.value)

        self.function_prologue(name, param_count)
        
        if block:
            current_offset = -2
            for var in local_vars:
                self.local_offsets[var] = current_offset
                current_offset -= 2
            
            scopes = [{}]
            self.generate_block(block, scopes, name, local_vars)
        
        self.emit(f"_{name}_exit:")
        self.function_epilogue()
        self.current_function = None

def compile_to_assembly(tree):
    cg = CodeGenerator()
    
    cg.emit("#bankdef mini33bank {")
    cg.emit("    #bits 16")
    cg.emit("    #addr 0xd000")
    cg.emit("    #size 0x1000")
    cg.emit("    #outp 0")
    cg.emit("}")
    cg.emit("#bank mini33bank")
    cg.emit("jmp _start", "Skip data section")
    
    for node in tree.children:
        if node.data == "function":
            cg.generate_function(node)
        elif node.data == "declaration":
            cg.generate_global(node)
    
    cg.emit("_start:")
    cg.emit("jsr main", "Call main function")
    cg.emit("jmp $", "Halt after execution")
    
    if cg.data_section:
        cg.output.extend([""] + cg.data_section)
    
    return cg.output

if not any(kind == "error" for _, _, _, kind, _ in diagnostics):
    assembly = compile_to_assembly(tree)
    print("\nCompilation successful. Generated assembly:")
    print("\n".join(assembly))
else:
    print("\nCompilation aborted due to errors")