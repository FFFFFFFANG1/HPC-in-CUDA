#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8
//@@ Define constant memory for device kernel here
__constant__ float mask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

  int radius = MASK_WIDTH / 2;
  int x_o = threadIdx.x + blockIdx.x * TILE_WIDTH; // col
  int y_o = threadIdx.y + blockIdx.y * TILE_WIDTH; // row
  int z_o = threadIdx.z + blockIdx.z * TILE_WIDTH; 

  int x_i = x_o - radius;
  int y_i = y_o - radius;
  int z_i = z_o - radius;

  if (x_i >= 0 && x_i < x_size && y_i >= 0 && y_i < y_size && z_i >= 0 && z_i < z_size) {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[z_i * y_size * x_size + y_i * x_size + x_i];
  } else {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();

  float Pvalue = 0.0;
  if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH && threadIdx.z < TILE_WIDTH) {
    for (int k = 0; k < MASK_WIDTH; k++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        for (int i = 0; i < MASK_WIDTH; i++) {
          Pvalue += mask[k][j][i] * tile[threadIdx.z + k][threadIdx.y + j][threadIdx.x + i];
        }
      }
    }
    if (x_o < x_size && y_o < y_size && z_o < z_size) {
      output[z_o * y_size * x_size + y_o * x_size + x_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength-3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength-3) * sizeof(float));
  

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, hostKernel, kernelLength * sizeof(float));

  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size / (1.0*TILE_WIDTH)), ceil(y_size / (1.0*TILE_WIDTH)), ceil(z_size / (1.0*TILE_WIDTH)));
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  //@@ Launch the GPU kernel here

  cudaDeviceSynchronize();
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);


  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);


  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

