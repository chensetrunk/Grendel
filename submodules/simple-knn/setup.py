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
    # Add other NPU specific include paths if necessary for acllib, tbe, etc.
    # However, kernel_operator.h and register/tilingdata_base.h are typically part of
    # a PyTorch NPU plugin's include path or custom op development headers.
    # For this example, assuming they are locatable by the compiler via framework setup.
]
NPU_LIBRARY_DIRS = [
    os.path.join(CANN_TOOLKIT_PATH, "lib64"),
    os.path.join(CANN_TOOLKIT_PATH, "acllib", "lib64"),
]

NPU_LIBRARIES = ["ascendcl"]

source_files_npu = [
    "npu_simple_knn/npu_simple_knn.cpp",
    "npu_simple_knn/npu_simple_knn_auxiliary.cpp",
    # TIK kernels in npu_simple_knn_kernels.cpp are compiled by TBE, not here.
    # The op definition in npu_simple_knn_tiling.cpp is also part of TBE build process
    # for custom ops, or loaded if this is a plugin.
    # This setup.py primarily compiles the C++ interface (ext.cpp, npu_simple_knn.cpp).
    "ext.cpp"
]

cpp_extra_compile_args_npu = ['-std=c++17', '-O2']
# if platform.system() == "Linux":
    # cpp_extra_compile_args_npu.append("-D_GLIBCXX_USE_CXX11_ABI=0") # Or 1

setup(
    name="simple_knn_npu", # Renamed package
    ext_modules=[
        CppExtension( # Changed from CUDAExtension
            name="simple_knn_npu._C", # Renamed module to avoid conflict
            sources=source_files_npu,
            include_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "npu_simple_knn/")
            ] + NPU_INCLUDE_DIRS,
            library_dirs=NPU_LIBRARY_DIRS,
            libraries=NPU_LIBRARIES,
            extra_compile_args=cpp_extra_compile_args_npu
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
