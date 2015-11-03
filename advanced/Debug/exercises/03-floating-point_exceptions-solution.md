# Exercise `03-floating-point_exceptions.c`

## Solution

We compile and execute the different type of floating-point exceptions:

```
$ gcc -Wall -g -lm 03-floating-point_exceptions.c -Ddivbyzero -o 03-floating-point_exceptions-divbyzero 
$ ./03-floating-point_exceptions-divbyzero 
Division by zero:   1.0 / 0.0 = inf
```

```
$ gcc -Wall -g -lm 03-floating-point_exceptions.c -Dinvalidop -o 03-floating-point_exceptions-invalidop 
$ ./03-floating-point_exceptions-invalidop 
Invalid operation:  sqrt(-1.0) = -nan
```

```
$ gcc -Wall -g -lm 03-floating-point_exceptions.c -Doverflow -o 03-floating-point_exceptions-overflow 
$ ./03-floating-point_exceptions-overflow  
Overflow:           exp( 1e30 ) = inf
```

In all the above cases, the program continues its execution without error but shows incorrect value.

In order to identify the origin of the error, we enable the floating-point exception using
```
feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
```

Now, the programs crashes when the error occurs and it is possible to locate where the error occurs with GDB.

