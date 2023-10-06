from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mylayerNorm_cuda',
    ext_modules=[
        CUDAExtension('mylayerNorm_cuda', [
            'mylayerNorm_cuda.cpp',
            'mylayerNorm_cuda_kernel.cu',
        ],
        extra_compile_args=['-std=c++17'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)