#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension # Changed from CUDAExtension
import os
import platform

# NPU specific paths - these might need to be configured by the user or environment variables
CANN_TOOLKIT_PATH = os.environ.get("ASCEND_TOOLKIT_INSTALL_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
NPU_INCLUDE_DIRS = [
    os.path.join(CANN_TOOLKIT_PATH, "include"),
    # Add other NPU specific include paths if necessary
    # e.g., for acllib, tbe, etc. if directly included, though typically covered by CANN_TOOLKIT_PATH/include
]
NPU_LIBRARY_DIRS = [
    os.path.join(CANN_TOOLKIT_PATH, "lib64"), # For aclrt, etc.
    os.path.join(CANN_TOOLKIT_PATH, "acllib", "lib64"), # For acllib related libraries
    # Add other NPU specific library paths
]

# NPU libraries to link against
NPU_LIBRARIES = [
    "ascendcl",
    # Add other libraries like 'acl_cblas' if used, etc.
]


# Common source files for the extension
source_files = [
    "npu_rasterizer/npu_rasterizer.cpp",
    "npu_rasterizer/npu_auxiliary.cpp",
    # "rasterize_points.cu", # This was CUDA, needs NPU equivalent or removal if functionality moved
    "ext.cpp"
]

# Note: TIK kernel sources (npu_rasterizer_kernels.cpp) are typically compiled by `tbe` (Tensor Boost Engine)
# and not directly included in CppExtension sources. The resulting .o files are then packaged.
# This setup.py assumes the TIK kernels are pre-compiled and registered with the system
# or that the op definitions in npu_rasterizer_tiling.cpp handle their loading.

# Extra compile args for C++ (g++)
# Might need to specify C++ standard, an NPU target architecture if applicable for host code
cpp_extra_compile_args = ['-std=c++17', '-O2']
# if platform.system() == "Linux": # Specific flags for Linux if any
    # cpp_extra_compile_args.append("-D_GLIBCXX_USE_CXX11_ABI=0") # Or 1, depending on PyTorch ABI

setup(
    name="diff_gaussian_rasterization_npu", # Renamed to avoid conflict
    packages=['diff_gaussian_rasterization'], # Python package name
    ext_modules=[
        CppExtension( # Changed from CUDAExtension
            name="diff_gaussian_rasterization._C_npu", # Python extension module name, changed
            sources=source_files,
            include_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"), # If GLM is still used by ext.cpp or npu_rasterizer.cpp (likely not for core logic)
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "npu_rasterizer/"), # Include path for npu_rasterizer.h etc.
            ] + NPU_INCLUDE_DIRS,
            library_dirs=NPU_LIBRARY_DIRS,
            libraries=NPU_LIBRARIES,
            extra_compile_args=cpp_extra_compile_args,
            # extra_link_args=[] # Add any NPU specific linker args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
