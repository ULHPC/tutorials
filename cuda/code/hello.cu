
/*
 * FIXME
 */

void helloCPU()
{
  std::cout<<"Hello from Cpu.\n";
}

void helloGPU()
{
  printf("Hello also from Gpu.\n");
}

int main()
{

  helloCPU();
  helloGPU();

  return EXIT_SUCCESS;
}
