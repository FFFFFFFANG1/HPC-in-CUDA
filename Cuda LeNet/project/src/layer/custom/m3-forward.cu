#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;

#define TILE_WIDTH 16
#define BLOCK_SIZE 256


#define wbcheck(stmt) do {                                                    \
        if (stmt != cudaSuccess) {                                             \
            std::cout<<"Failed to run stmt "<<#stmt<<std::endl;                       \
            std::cout<<"Got CUDA error ...  "<<cudaGetErrorString(stmt)<<std::endl;    \
            exit(-1);                                                        \
        }                                                                     \
    } while(0)


__global__ void conv_forward_kernel (const float * __restrict__ input, const float * __restrict__ mask, float * __restrict__ output, const int Batch, 
                                        const int Map_out, const int Channel, const int Height, const int Width, const int K){
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;                                        
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * Width_out + i0]

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half input_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ half mask_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float output_tile[TILE_WIDTH][TILE_WIDTH];
    
    //unroll
    int batch_idx = blockIdx.z;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // pixel
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // map

    //matmul
    int numAColumns = Channel * K * K;
    int numBColumns = Height_out * Width_out;
    int h = col / Width_out;
    int w = col % Width_out;
    int mask_size = K * K;
    int ty = threadIdx.y;

    #pragma unroll
    for (int tileId = 0; tileId < ceil(numAColumns / (TILE_WIDTH * 1.0)); tileId++) {
        int colA = tileId * TILE_WIDTH + threadIdx.x;
        int rowB = tileId * TILE_WIDTH + threadIdx.y;
        int c = rowB / mask_size;
        int p = (rowB % mask_size) / K;
        int q = (rowB % mask_size) % K;
        if (row < Map_out && colA < numAColumns) {
            mask_tile[threadIdx.y][threadIdx.x] = __float2half(mask[row * numAColumns + tileId * TILE_WIDTH + threadIdx.x]);
        } else {
            mask_tile[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        if (col < numBColumns && rowB < numAColumns) {
            input_tile[threadIdx.y][threadIdx.x] = __float2half(in_4d(batch_idx, c, h + p, w + q));
        } else {
            input_tile[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        __syncthreads();
        if (ty == 0 || ty == 1){  // use one warp
            wmma::load_matrix_sync(a_frag, (half*)mask_tile, TILE_WIDTH);
            wmma::load_matrix_sync(b_frag, (half*)input_tile, TILE_WIDTH);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    if (ty == 0 || ty == 1){
        wmma::store_matrix_sync((float*)output_tile, c_frag, TILE_WIDTH, wmma::mem_row_major);
    }
    __syncthreads();
    
    if (row < Map_out && col < numBColumns) {
        out_4d(batch_idx, row, h, w) = output_tile[threadIdx.y][threadIdx.x];
    }
    #undef in_4d
    #undef out_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    cudaMalloc(device_output_ptr, (size_t) Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc(device_input_ptr, (size_t) Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc(device_mask_ptr, (size_t) Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (size_t) Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, (size_t) Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}




__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;    

    // image size * map out * batch
    dim3 grid_dim(ceil((Height_out * Width_out) / (1.0 * TILE_WIDTH)), ceil(Map_out / (TILE_WIDTH * 1.0)), Batch);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<grid_dim, block_dim>>>(device_input, device_mask, device_output, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, (size_t) Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}