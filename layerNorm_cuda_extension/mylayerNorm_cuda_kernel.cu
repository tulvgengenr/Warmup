#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void mylayerNorm_kernel(
    const scalar_t* A,
    scalar_t* B,
    const int M, 
    const int N,
    const float eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M)
    {
        scalar_t mean = 0.0;
        scalar_t var = 0.0;

        // Compute mean along the last dimension
        for (int i = 0; i < N; i++) {
            mean += A[row * N + i];
        }
        mean /= N;

        // Compute variance along the last dimension
        for (int i = 0; i < N; i++) {
            scalar_t diff = A[row * N + i] - mean;
            var += diff * diff;
        }
        scalar_t std = sqrt(var / N + eps);

        // Normalize input vector using mean and variance
        for (int i = 0; i < N; ++i) {
            B[row * N + i] = (A[row * N + i] - mean) / std;
        }
    }   
}


std::vector<torch::Tensor> mylayerNorm_cuda_forward(
    torch::Tensor input,
    float eps)
{
    const int M = input.size(0);
    const int N = input.size(1);
    auto output = torch::zeros_like(input, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(256);
    const dim3 grid((M - 1) / 256 + 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylayerNorm_cuda_forward", ([&] {
        mylayerNorm_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            M,
            N,
            eps);
    }));

    return {output};
}
