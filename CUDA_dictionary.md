# CUDA dictionary

### Kernel declaration
``` cpp
__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}
```

`__global__ void GPUFunction()`
  - The `__global__` keyword indicates that the following function will run on the GPU, and can be invoked **globally**, which in this context means either by the CPU, or, by the GPU.
  - Often, code executed on the CPU is referred to as **host** code, and code running on the GPU is referred to as **device** code.
  - Notice the return type `void`. It is required that functions defined with the `__global__` keyword return type `void`.
### Calling kernel
```cpp
int main()
{
  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```

`GPUFunction<<<1, 1>>>();`
  - Typically, when calling a function to run on the GPU, we call this function a **kernel**, which is **launched**.
  - When launching a kernel, we must provide an **execution configuration**, which is done by using the `<<< ... >>>` syntax just prior to passing the kernel any expected arguments.
  - At a high level, execution configuration allows programmers to specify the **thread hierarchy** for a kernel launch, which defines the number of thread groupings (called **blocks**), as well as how many **threads** to execute in each block. Notice the kernel is launching with `1` block of threads (the first execution configuration argument) which contains `1` thread (the second configuration argument).

For example, `GPUFunction<<<2, 4>>>();` is:

```
┌─────────────────────┐
│        Block 0      │
│  ┌───┬───┬───┬───┐  │
│  │ T0│ T1│ T2│ T3│  │
│  └───┴───┴───┴───┘  │
├─────────────────────┤
│        Block 1      │
│  ┌───┬───┬───┬───┐  │
│  │ T4│ T5│ T6│ T7│  │
│  └───┴───┴───┴───┘  │
└─────────────────────┘

<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>
```

`cudaDeviceSynchronize();`
  - Unlike much C/C++ code, launching kernels is **asynchronous**: the CPU code will continue to execute *without waiting for the kernel launch to complete*.
  - A call to `cudaDeviceSynchronize`, a function provided by the CUDA runtime, will cause the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU.


### Mapping cuda indexes into index of data
  - `gridim.x` is the number of blocks in the grid, in the above case 2
  - `blockIdx.x` is the index of the currect block within the grid, in the case 0 when executing the code in 0th block, 1 for when executing the code in 1st block, so on
  - `blockDim.x` is the number of threads in a block, in the above case 4 (all blocks has same number of threads)
  - `threadIdx.x` is the index of the thread within an executing block
Using these variables, the formula
``` cpp
int data_index = threadIdx.x + blockIdx.x * blockDim.x;
```
will map each thread to one element in a vector. For example
```
┌─────────────────────┐
│        Block 0      │
│  ┌───┬───┬───┬───┐  │
│  │ T0│ T1│ T2│ T3│  │
│  └───┴───┴───┴───┘  │
├─────────────────────┤
│        Block 1      │
│  ┌───┬───┬───┬───┐  │
│  │ T4│ T5│ T6│ T7│  │
│  └───┴───┴───┴───┘  │
└─────────────────────┘
```
would give 0,1,2,3,4,5,6,7.
- There is a limit to the number of threads that can exist in a thread block: 1024 to be precise

### Computing number of needed blocks for data
``` cpp
int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
```
(note here in C/C++: `a/b` is an integer division)

### Manging shared memory
``` cpp
int N = 100;
int *a;
size_t size = N * sizeof(int);
cudaMallocManaged(&a, size);
cudaFree(a);
```

### Grid stride loop
``` cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

for (int i = idx; i < N; i += stride)
{
	a[i] = a[i]; //do work
}
```

### Shared memory management error handling
``` cpp
cudaError_t err;
err = cudaMallocManaged(&a, N);                    	// Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           	// `cudaSuccess` is provided by CUDA.
{
	printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}

```

### Kernel running error handling
``` cpp
someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
	printf("Error: %s\n", cudaGetErrorString(err));
}
```

### General CUDA error handling function
``` cpp
inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}
```
