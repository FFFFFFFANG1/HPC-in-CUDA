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
  __shared__ float prev;
  int tx = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;
  if (blockIdx.x > 0) {
    if (tx == 0) {
      prev = prev_sum[blockIdx.x - 1];
    }
    __syncthreads();
    if (start + tx < len) {
      input1[start + tx] += prev;
    }
    if (start + blockDim.x + tx < len) {
      input1[start + blockDim.x + tx] += prev;
    }
  }
}


__global__ void scan(float *input, float *output, float* prev_sum, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  __shared__ float T[2 * BLOCK_SIZE];
  int start = 2 * blockIdx.x * blockDim.x;
  int tx = threadIdx.x;
  int stride;

  if (start + tx < len) {
    T[tx] = input[start + tx];
  } else {
    T[tx] = 0;
  }
  if (start + blockDim.x + tx < len) {
    T[blockDim.x + tx] = input[start + blockDim.x + tx];
  } else {
    T[blockDim.x + tx] = 0;
  }

  stride = 1;
  while (stride < 2 * blockDim.x) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < 2 * blockDim.x) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }

  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2 * blockDim.x) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();
  if (start + tx < len) {
    output[start + tx] = T[tx];
  }
  if (start + blockDim.x + tx < len) {
    output[start + blockDim.x + tx] = T[blockDim.x + tx];
  }
  if (prev_sum != NULL && tx == 0) {
    prev_sum[blockIdx.x] = T[2 * blockDim.x - 1];
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
  int numBlocks = ceil(numElements/(float)(BLOCK_SIZE*2));
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
  dim3 DimGrid_Add(1, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, devicePrev_sum, numElements);
  if (numBlocks > 1) {
    scan<<<DimGrid_Add, DimBlock>>>(devicePrev_sum, devicePrev_sum, NULL, numBlocks);
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

