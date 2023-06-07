// Authour: Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

#include <stdio.h>              
// INCLUDE OPENACC LIBRARY ????

void Print_Hello_World()    
{
#pragma acc ..............?????
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
} 

int main()
{ 
  Print_Hello_World();     
  return 0;     
}
