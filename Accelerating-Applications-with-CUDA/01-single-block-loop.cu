#include <stdio.h>


/*
 * Notice the absence of the previously expected argument `N`.
 */

__global__ void loop()
{

  printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{

  loop<<<1, 10>>>();
  cudaDeviceSynchronize();
}
