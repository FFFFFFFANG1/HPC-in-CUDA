// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void add(float* input1, float *prev_sum, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0 && i < len) {
    input1[i] += prev_sum[blockIdx.x - 1];
  }

}


__global__ void scan(float *input, float *output, float* prev_sum, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T0[BLOCK_SIZE];
  __shared__ float T1[BLOCK_SIZE];

  float* src = T0;
  float* dst = T1;
  float* tmp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    T0[threadIdx.x] = input[i];
    T1[threadIdx.x] = T0[threadIdx.x];
  
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
      __syncthreads();
      if (threadIdx.x >= stride) {
        dst[threadIdx.x] = src[threadIdx.x - stride] + src[threadIdx.x];
      } else {
        dst[threadIdx.x] = src[threadIdx.x];
      }
      tmp = src;
      src = dst;
      dst = tmp;
    }
    output[i] = src[threadIdx.x];
  }
  if (prev_sum != NULL && threadIdx.x == 0) {
    prev_sum[blockIdx.x] = src[blockDim.x - 1];
  }
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *devicePrev_sum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  int numBlocks = ceil(numElements/(float)BLOCK_SIZE);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicePrev_sum, numBlocks * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(devicePrev_sum, 0, numBlocks * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numBlocks, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, devicePrev_sum, numElements);
  if (numBlocks > 1) {
    scan<<<1, numBlocks>>>(devicePrev_sum, devicePrev_sum, NULL, numBlocks);
    add<<<DimGrid, DimBlock>>>(deviceOutput, devicePrev_sum, numElements);
  }
  
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

