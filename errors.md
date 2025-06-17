
# C16.256 Error codes and information

## Quick lookup

### Warnings

- `W001`

Unused local variable

### Errors

- `E001`

Redeclaration of function

- `E002`

Unexpected top-level construct

- `E003`

Missing `main` entry point function

- `E004`

Redeclaration of variable

- `E005`

Variable is not an array

- `E006`

Array variable used without indexing

- `E007`

Undeclared variable

- `E008`

Unexpected statement

- `E009`

Call to undeclared function

- `E010`

Function called with wrong arg count

## Examples

### `W001`

Example:

```none
14 | int hi_variable = 100;

| ^ warning: unused local variable `hi_variable` [W001]
```

This warning is triggered when a variable is declared but never used.

### `E001`

Example:

```none
12 | int factorial(int n) {

| ^ error: redeclaration of `factorial` [E001]
```

This error is triggered when a function is declared which shares a name with another known function.

### `E002`

Example:

```none
6 | x = 1;

| ^ error: unexpected top‑level `statement` [E002]
```

This error is triggered when a statement (such as an assignment, if-statement, or return) appears at the top level outside any function or declaration.

### `E003`

Example:

```none
2 | int arr[5] = {1, 2, 3, 4, 5};

| ^ error: missing `main` function [E003]
```

This error is triggered when the `main` entry point function is missing. The `main` function can be placed anywhere in the program.

### `E004`

Example:

```c
int total =  0;

int total =  3;
```

produces:

```none
13 | int total = 0;

| ^ warning: unused local variable `total` [W001]

  
  

14 | int total = 3;

| ^ error: redeclaration of `total` [E004]
```

This error is triggered when a variable is redefined instead of reused.

### `E005`

Example:

```c
int total =  0;

total[0] =  3;
```

produces:

```none
14 | total[0] = 3;

| ^ error: `total` is not an array [E005]
```

This error is triggered when a variable is indexed as if it were an array.

### `E006`

Example:

```c
int anArray[5] = {1, 2, 3, 4, 5};

anArray =  3;
```

produces:

```none
14 | anArray = 3;

| ^ error: `anArray` must be indexed [E006]
```

This error is triggered when an array is assigned a value as if it were a scalar variable.

### `E007`

Example:

```none
20 | return total;

| ^ error: undeclared variable `total` [E007]
```

This error is triggered when a variable is used when it has not been assigned a value.

### `E008`

This is an error that occurs mostly when doing work on the semantics parser, if you see it, please report it.

### `E009`

Example:

```none
16 | undefinedFunction();

| ^ error: call to undeclared function `undefinedFunction` [E009]
```

This error is triggered when a function which has not been defined is called.

### `E010`

Example:

```none
8 | return n * factorial(n - 1, 5, 3);

| ^ error: `factorial` expects 1 args, got 3 [E010]
```

This error is triggered when the wrong amount of arguments have been supplied to a function.
