// Histogram Equalization
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ insert code here
#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32


__global__ void float_to_char(float *input, unsigned char *output, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;
  if (x < width && y < height) {
    output[(y * width + x) * channels + z] = (unsigned char)(255 * input[(y * width + x) * channels + z]);
  }
}

__global__ void RGB_to_Gray(unsigned char *input, unsigned char *output, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned char r = input[(y * width + x) * channels];
    unsigned char g = input[(y * width + x) * channels + 1];
    unsigned char b = input[(y * width + x) * channels + 2];
    output[y * width + x] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void histogramize(unsigned char *input, unsigned int *histogram, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    atomicAdd(&(histogram[input[y * width + x]]), 1);
  }
}


__global__ void correct_n_float(unsigned char *input, float* output, float *cdf, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;
  if (x < width && y < height) {
    float cdf_min = cdf[0];
    float correct_color = min(255.0, max(255.0 * (cdf[input[(y * width + x) * channels + z]] - cdf_min) / (1.0 - cdf_min), 0.0));
    output[(y * width + x) * channels + z] = correct_color / 255.0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char* deviceInputImageData_char_gray;
  unsigned char* deviceInputImageData_char_3c;
  unsigned int * histogram;
  float * cdf;
  float host_cdf[HISTOGRAM_LENGTH];
  unsigned int host_histogram[HISTOGRAM_LENGTH];

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ insert code here
  //Allocate
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceInputImageData_char_gray, imageWidth * imageHeight * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceInputImageData_char_3c, imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float)));

  //copy and set
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(cdf, 0, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));

  dim3 dimGrid_3c((imageWidth - 1) / BLOCK_SIZE + 1, (imageHeight - 1) / BLOCK_SIZE + 1, imageChannels);
  dim3 dimGrid_1c((imageWidth - 1) / BLOCK_SIZE + 1, (imageHeight - 1) / BLOCK_SIZE + 1, 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  float_to_char<<<dimGrid_3c, dimBlock>>>(deviceInputImageData, deviceInputImageData_char_3c, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  RGB_to_Gray<<<dimGrid_1c, dimBlock>>>(deviceInputImageData_char_3c, deviceInputImageData_char_gray, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  histogramize<<<dimGrid_1c, dimBlock>>>(deviceInputImageData_char_gray, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //only takes O(1)
  wbCheck(cudaMemcpy(host_histogram, histogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  host_cdf[0] = host_histogram[0] / float(imageWidth * imageHeight);
  for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
    host_cdf[i] = host_cdf[i - 1] + host_histogram[i] / float(imageWidth * imageHeight);
  }
  wbCheck(cudaMemcpy(cdf, host_cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

  correct_n_float<<<dimGrid_3c, dimBlock>>>(deviceInputImageData_char_3c, deviceOutputImageData, cdf, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost));

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceInputImageData_char_gray);
  cudaFree(deviceInputImageData_char_3c);
  cudaFree(histogram);
  cudaFree(cdf);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}