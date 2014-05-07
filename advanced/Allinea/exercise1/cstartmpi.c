/* Some comments.... */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    int myInt;
    char* charStar;
} typeOne;

typedef struct {
    int myInt;
    int yourInt;
    typeOne subList;
} typeTwo;

typedef struct {
    double myDoube;
    unsigned int unsignedInt;
    typeTwo anotherList;
    typeOne hippo;
    char c;
} typeThree;

int func2()
{
    typeTwo peace;

    peace.myInt = 1;
    peace.yourInt = 3;

    return 17;
}

void func1()
{
    int test;
    test = func2();
    if(test>1)
        test=0;
    else
        test=-1;
}

void func3()
{
    void* i = 0;
    while(++i || !i)
        free((void*)i);
}

void print_arg(const char* arg)
{
    printf("%s\n", arg);
}

int main(int argc, char** argv, char** environ)
{
    typeThree test;
    typeThree* t2;
    int i;
    int p=0;    		/* Number of processors */
    int source;			/* Rank of sender */
    int dest;			/* Rank of receiver */
    int tag = 50;		/* Tag for messages */
    char message[100];	/* Storage for the message */
    int my_rank;		/* Rank of process */
    float tables[12][12];
    const char* troopa;
    int bigArray[10000];
    int x,y;
    int beingWatched;
    int* sdim;
    int* dynamicArray;

    void (*s)(int);

    MPI_Status status;	/* Return status for receive */

    t2 = malloc(sizeof(typeThree));

    for(p=0;p<100;p++)
        bigArray[p]=80000+p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    dynamicArray = malloc(sizeof(int)*100000);
    sdim = malloc(sizeof(int) * p);

    for(x=0;x<10000;x++)
    {
        dynamicArray[x] = x%10;
    }

    printf("my rank is %d\n", my_rank);

    for(x=0;x<12;x++)
    {
        y = 0;
        while(y != 12)
        {
            tables[x][y] = (x+1)*(y+1);
            y += my_rank + 1;
        }
    }

    if(argc > 1 && my_rank == 0) 
    {
        printf("Rank %d has %d arguments.\n", my_rank, argc);
        printf("They are:\n");
        for(x=0; x<argc; y++)
            print_arg(argv[y]);
    }

    func1();
    func2();

    fprintf(stderr, "I can write to stderr too\n");

    beingWatched = 1;

    test.anotherList.subList.charStar = "hello";
    test.c = 'p';
    beingWatched = 0;
    /* while (-1) { } */

    for(x=0; x<p; x++) sdim[x] = my_rank * x;

    if (my_rank != 0)
    {
        sprintf(message, "Greetings from process %d!", my_rank);
        printf("sending message from (%d)\n",my_rank);
        dest = 0;
        /* Use strlen(message)+1 to include '\0' */
        MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        beingWatched--;
    }
    else 
    {
        /* my_rank == 0 */
        for (source = 1; source < p; source++)
        {
            printf("waiting for message from (%d)\n", source);
            MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
            beingWatched++;
        }
    }

    for(i=1;i<argc;i++) 
        if (argv[i] && !strcmp(argv[i], "memcrash")) 
            func3();

    for(i=1;i<argc;i++)
        if (argv[i] && !strcmp(argv[i], "sleepy"))
            sleep(500000);


    beingWatched = 12;
    MPI_Finalize();

    beingWatched = 0;

    printf("all done...(%d)\n",my_rank);

    return 0;
} /* main */
