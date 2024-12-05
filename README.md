# All works for ECE408

The repo includes all my implementation of parallel programming in CUDA. Labs are weekly assignments aiming for practice. The CUDA LeNet is the course final project.

## CUDA LeNet Implementation & Optimization
Course final project for ECE 408 FA23 @ UIUC. For original project description, see [Cuda LeNet/README.md](_README.md).

### Optimization Techniques
For a more comprehensive discussion, see [Cuda LeNet/report.pdf](report.pdf). Below is a table of optimization techniques:
| Name                | Techiques                    |
| --------------------------- | ---------------------- |
| Req 0 | Stream |
| Req 1 | Tensor Core |
| Req 2 | Kernel Fusion |
| Op 1 | __restrict__ |
| Op 2 | loop unrolling |
| Op 5 | FP16 Operation |


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
