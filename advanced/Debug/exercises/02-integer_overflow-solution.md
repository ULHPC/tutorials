# Exercise `02-integer_overflow.c`

## Solution

This programs calculate the factorial of the parameter.
Execution with parameter 13 or above returns a wrong value because of an integer overflow.
A `int` variable cannot hold value bigger than 2^31-1.

```
$ ./02-integer_overflow 12
 fact(12) = 479001600
$ ./02-integer_overflow 13
 fact(13) = 1932053504
```

One solution is to use an integer type that can use larger integer value, for example `long long int`. This allows to calculate factorial correctly up to 20.

```
$ ./02-integer_overflow-solution 20                          
 fact(20) = 2432902008176640000
```

We can also add an error message if the user asks for factorial above 20.

```
$ ./02-integer_overflow-solution 21                                                                    
Error: cannot calculate factorial above 20!
```
