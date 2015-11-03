# Exercise `01-logic_syntax_bugs.c`

## Solution

Compile with the warnings

```
$ gcc -Wall -g 01-logic_syntax_bugs-solution.c -o 01-logic_syntax_bugs
01-logic_syntax_bugs-solution.c: In function 'main':
01-logic_syntax_bugs-solution.c:24:3: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
   else if ( nb_params = 1 )
   ^
```

The compiler gives a warning about an assignment at line 24.
The code contains a mistake here as comparison should be done here instead of an assignment.
We should replace `=` by `==`.
