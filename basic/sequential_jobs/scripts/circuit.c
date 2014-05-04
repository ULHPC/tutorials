#include <stdio.h>      /* printf */
#include <stdlib.h>     /* atoi */
#include <unistd.h>     /* usleep */

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

/* Solution 39413 */
int check_circuit(int input) {
    int v[16], i;
    for (i = 0; i < 16; i++) v[i] = EXTRACT_BIT(input,i);
    usleep(10000);
    return ((v[0] || v[1])    && (!v[1] || !v[3]) && (v[2] || v[3])  &&
            (!v[3] || !v[4])  && (v[4] || !v[5])  && (v[5] || !v[6]) &&
            (v[5] || v[6])    && (v[6] || !v[15]) && (v[7] || !v[8]) &&
            (!v[7] || !v[13]) && (v[8] || v[9])   && (v[8] || !v[9]) &&
            (!v[9] || !v[10]) && (v[9] || v[11])  && (v[10] || v[11])&&
            (v[12] || v[13])  && (v[13] || !v[14])&& (v[14] || v[15]));
}

int main(int argc, char *argv[]) {
        int param;

        if (argc != 2) {
            printf("No parameter?\n");
            return 1;
        }

        param = atoi(argv[1]);

        if ( check_circuit(param) ) {
            printf("Found solution: %d\n", param);
            return 0;
        }
        else {
            return 2;
        }
}

