# All works for Parallel Programming Course

The repo includes all my implementation of parallel programming in CUDA. Labs are weekly assignments aiming for practice. The CUDA LeNet is the course final project.

## CUDA LeNet Implementation & Optimization
Course final project for ECE 408 FA23 @ UIUC. For original project description, see `Cuda LeNet/README.md`.

### Baseline
I first implemented the baseline using matrix unrolling and tiled matrix multiplication for the convolution layer.

### Optimization Techniques
For a more comprehensive discussion, see `CUDA_LeNet/report.pdf`. Below is a table of optimization techniques:
| Name                | Techiques                    |
| --------------------------- | ---------------------- |
| Req 0 | Stream |
| Req 1 | Tensor Core |
| Req 2 | Kernel Fusion |
| Op 1 | restrict |
| Op 2 | loop unrolling |
| Op 5 | FP16 Operation |

### Performance
After optimization, the sum of operation time reduced from over **200ms** to **73ms**.

In the project, there are three batch sizes: 100, 1000, 10000, and the network consists of two feed-forward layer. The operation time for each layer is recorded as Op Time 1 and Op Time 2. The table below shows the overall performance of each optimization and the final stacked one under 10000 batches of images. One may notice that not all optimizations really make improvements. In `CUDA_LeNet/report.pdf`, I analyzed why or why not did each optimization work using profiling. The result of batches of 100, 1000 can also be found there.

| Implementation | Op Time 1 (ms) | Op Time 2 (ms) |
| ------------- | --------- | --------- |
| **Baseline**      | **77.78**    | **124.33**   |
| Req 0         | 0.00408   | 0.003   |
| Req 1          | 76.69   | 114.88   |
| Req 2          | 55.01   | 30.11   |
| Op 1         | 77.53   | 124.01    |
| Op 2          | 77.63    | 124.30    |
| Op 3          | 74.65    | 90.28    |
| **Final**         | **49.42**    | **23.57**   |


Please be reminded that due to the implementation of timing micro, the op time for streaming is not correct. However, I found streaming did not improve the performance by looking at the profile result.

## Labs

Here is a list of techiques included in labs:

| Lab                 | Techiques                    |
| --------------------------- | ---------------------- |
| lab 1 | Vector Addition |
| lab 2 | Basic Matrix Multiplication |
| lab 3 | Tiled Matrix Multiplication |
| lab 4 | 3D Convolution |
| lab 5 | List Reduction Sum |
| lab 6 | Parallel Scan |
| lab 7 | Histogram and Color Correction |
| lab 8 | Sparse Matrix Multiplication |
