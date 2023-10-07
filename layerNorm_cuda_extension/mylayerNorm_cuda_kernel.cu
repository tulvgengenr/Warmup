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
    torch::Tensor input)
{
    const int M = input.size(0);
    const int N = input.size(1);
    auto output = torch::zeros_like(input, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(256);
    const dim3 grid((M - 1) / 256 + 1);

    const float eps = 1e-5;

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

template <typename scalar_t>
__global__ void mylayerNorm_backward_kernel(
    const scalar_t* grad_output,
    const scalar_t* input,
    scalar_t* grad_input,
    const int M, 
    const int N,
    const float eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        scalar_t mean = 0.0;
        scalar_t var = 0.0;

        // Compute mean
        for (int i = 0; i < N; i++) {
            mean += input[row * N + i];
        }
        mean /= N;

        // Compute variance
        for (int i = 0; i < N; i++) {
            var += (input[row * N + i] - mean) * (input[row * N + i] - mean);
        }`
        var /= N;

        scalar_t dvar = 0.0;
        scalar_t dmean = 0.0;
        scalar_t std_inv = 1.0 / sqrt(var + eps);
        scalar_t sum_diff = 0.0;

        for (int i = 0; i < N; i++) {
            dvar += (input[row * N + i] - mean) * grad_output[row * N + i] * (-0.5) * powf(var + eps, -1.5);
            dmean += grad_output[row * N + i] * (-std_inv);
            sum_diff += (input[row * N + i] - mean);
        }

        dmean += dvar * (-2.0/N) * sum_diff;

        for (int i = 0; i < N; i++) {
            grad_input[row * N + i] = grad_output[row * N + i] * std_inv + dvar * 2.0/N * (input[row * N + i] - mean) + dmean / N;
        }
    }
}

std::vector<torch::Tensor> mylayerNorm_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input)
{
    const int M = input.size(0);
    const int N = input.size(1);
    auto grad_input = torch::zeros({M, N}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(256);
    const dim3 grid((M - 1) / 256 + 1);

    const float eps = 1e-5;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylayerNorm_cuda_backward", ([&] {
        mylayerNorm_backward_kernel<scalar_t><<<grid, block>>>(
            grad_output.data<scalar_t>(),
            input.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            M,
            N,
            eps);
    }));

    return {grad_input};
}
