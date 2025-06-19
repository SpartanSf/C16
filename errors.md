
# C16.256 Error codes and information

## Quick lookup

### Warnings

| Code     | Description           |
| -------- | --------------------- |
| **W001** | Unused local variable |

### Errors

| Code     | Description                                                               |
| -------- | ------------------------------------------------------------------------- |
| **E001** | Redeclaration of function                                                 |
| **E002** | Unexpected top-level construct                                            |
| **E003** | Missing `main` entry point function                                       |
| **E004** | Redeclaration of variable                                                 |
| **E005** | Variable is not an array                                                  |
| **E006** | Array variable used without indexing                                      |
| **E007** | Undeclared variable                                                       |
| **E008** | Unexpected statement                                                      |
| **E009** | Call to undeclared function                                               |
| **E010** | Function called with wrong argument count                                 |
| **E011** | `break` not within loop or switch                                         |
| **E012** | `continue` not within loop                                                |
| **E013** | Invalid struct/field access on non-struct or non-pointer-to-struct type   |
| **E014** | Struct has no such field                                                  |
| **E015** | `for` loop condition must be scalar                                       |
| **E016** | Could not find variable name token in inline declaration                  |
| **E017** | Could not find parameter name token                                       |
| **E018** | Could not find variable name token in declaration                         |
| **E019** | Malformed `for` statement                                                 |
| **E020** | *(Reserved; not currently used — could be used for future type mismatch)* |
| **E021** | Could not find function name in call                                      |
| **E500** | Could not find typedef alias name token                                   |

## Examples

## Examples

### `W001`

Example:

```none
14 | int hi_variable = 100;
   |     ^^^^^^^^^^^ warning: unused local variable `hi_variable` [W001]
```

This warning is triggered when a variable is declared but never used within its scope.

---

### `E001`

Example:

```none
   4 | int factorial(int n) {
     |     ^^^^^^^^^ note: previous declaration of `factorial` here

  12 | int factorial(int n) {
     |     ^^^^^^^^^ error: redeclaration of `factorial` [E001]
```

This error is triggered when a function is declared that shares a name with another already-declared function.

---

### `E002`

Example:

```none
6 | x = 1;
  | ^ error: unexpected top-level `statement` [E002]
```

This error is triggered when a statement (such as an assignment or control flow) appears outside any function or declaration. Only declarations and function definitions are allowed at the top level.

---

### `E003`

Example:

```none
2 | int arr[5] = {1, 2, 3, 4, 5};
  | ^ error: missing `main` function [E003]
```

This error is triggered when no `main` function is defined. Every program must have an entry point named `main`.

---

### `E004`

Example:

```c
int total = 0;
int total = 3;
```

Produces:

```none
13 | int total = 0;
   |     ^^^^^ warning: unused local variable `total` [W001]

14 | int total = 3;
   |     ^^^^^ error: redeclaration of `total` [E004]
```

This error is triggered when a local variable is declared multiple times in the same scope.

---

### `E005`

Example:

```c
int total = 0;
total[0] = 3;
```

Produces:

```none
14 | total[0] = 3;
   | ^^^^^ error: `total` is not an array [E005]
```

This error is triggered when a non-array variable is used with array-style indexing.

---

### `E006`

Example:

```c
int anArray[5] = {1, 2, 3, 4, 5};
anArray = 3;
```

Produces:

```none
14 | anArray = 3;
   | ^^^^^^^ error: `anArray` must be indexed [E006]
```

This error is triggered when an array is used as a scalar value (e.g., in assignments) instead of being indexed or passed by reference.

---

### `E007`

Example:

```none
20 | return total;
   |        ^^^^^ error: undeclared variable `total` [E007]
```

This error is triggered when a variable is referenced without being declared in the current scope or any outer scope.

---

### `E008`

This error is triggered when a statement construct is encountered in an invalid or unexpected location. This usually happens during compiler development. If you encounter this in valid code, please report it.

---

### `E009`

Example:

```none
16 | undefinedFunction();
   | ^^^^^^^^^^^^^^^^^ error: call to undeclared function `undefinedFunction` [E009]
```

This error is triggered when a function call references a function that has not been declared.

---

### `E010`

Example:

```none
8 | return n * factorial(n - 1, 5, 3);
  |            ^^^^^^^^^ error: `factorial` expects 1 args, got 3 [E010]
```

This error is triggered when a function is called with the wrong number of arguments compared to its declaration.

---

### `E011`

Example:

```none
17 | break;
   | ^^^^^ error: `break` not within loop or switch [E011]
```

This error is triggered when a `break` statement is used outside of a loop or switch context.

---

### `E012`

Example:

```none
18 | continue;
   | ^^^^^^^^ error: `continue` not within loop [E012]
```

This error is triggered when a `continue` statement is used outside of a loop.

---

### `E013`

Example:

```none
p1 = 3;
p1.x = 2;
```

Produces:

```none
15 | p1.x = 2;
   | ^^ error: Cannot access field of non-struct type `int` [E013]
```

This error is triggered when field access is attempted on a variable that is not a struct.

---

### `E014`

Example:

```none
p1.unknown = 5;
```

Produces:

```none
15 | p1.unknown = 5;
   |    ^^^^^^^ error: `Point` has no field `unknown` [E014]
```

This error is triggered when accessing a non-existent field from a struct.

---

### `E015`

Example:

```none
for (int i = 0; some_struct; i = i + 1) {
}
```

Produces:

```none
15 | for (int i = 0; some_struct; i = i + 1) {
   |                  ^^^^^^^^^^ error: for loop condition must be scalar, got struct Point [E015]
```

This error is triggered when the condition in a `for` loop does not evaluate to a scalar type (like `int` or `char`).

---

### `E016`

This error is triggered when a variable declaration fails to identify a valid name token. This likely indicates a malformed or incomplete declaration.

---

### `E017`

This error is triggered when a function parameter declaration is missing a valid name token.

---

### `E018`

This error is triggered when a standalone variable declaration lacks a valid name token. This may happen in malformed declarations.

---

### `E019`

This error is triggered when a `for` statement is malformed, missing one or more required components (e.g. initializer, condition, or post-expression).

---

### `E020`

(Reserved for future use)

---

### `E021`

This error is triggered when a function call expression is missing a name token that can identify the target function.

---

### `E500`

This error is triggered when a `typedef` declaration is missing a valid alias name token. This usually results from invalid typedef syntax.
