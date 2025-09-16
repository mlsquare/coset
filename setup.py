import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup, find_packages

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Define extensions
extensions = []

if cuda_available:
    # CUDA extensions
    extensions.append(
        CUDAExtension(
            name='coset_cuda',
            sources=[
                'src/cuda/kernels/quantization.cu',
                'src/cuda/kernels/product_quantization.cu',
                'src/cuda/kernels/matmul.cu',
                'src/cuda/kernels/gradients.cu',
                'src/cuda/utils/memory.cu',
                'src/cuda/utils/math.cu',
                'src/cpp/operators.cpp',
                'src/cpp/autograd.cpp',
                'src/cpp/bindings.cpp',
            ],
            include_dirs=[
                'src/cuda/include',
                'src/cpp/include',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--gpu-architecture=sm_80',  # Ampere
                    '--gpu-architecture=sm_90',  # Hopper
                    '-Xptxas=-O3',
                    '--std=c++17'
                ]
            }
        )
    )
else:
    # CPU-only extensions
    extensions.append(
        CppExtension(
            name='coset_cpu',
            sources=[
                'src/cpp/operators.cpp',
                'src/cpp/autograd.cpp',
                'src/cpp/bindings.cpp',
            ],
            include_dirs=[
                'src/cpp/include',
            ],
            extra_compile_args=['-O3', '-std=c++17']
        )
    )

setup(
    name="coset",
    version="0.1.0",
    description="Hierarchical Nested Lattice Quantization for Matrix Operations",
    author="mlsquare",
    author_email="",
    url="https://github.com/mlsquare/coset",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
