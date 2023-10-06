#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA funciton declearition
std::vector<torch::Tensor> mylayerNorm_cuda_forward(
    torch::Tensor input,
    float eps);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> mylayerNorm_forward(
    torch::Tensor input,
    float eps) 
{
    CHECK_INPUT(input);

    return mylayerNorm_cuda_forward(input, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mylayerNorm_forward, "myLayerNorm forward (CUDA)");
}