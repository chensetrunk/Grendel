/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "npu_simple_knn/npu_simple_knn.h" // Changed include
#include "npu_simple_knn/npu_simple_knn_config.h" // For NPU_KNN_BOX_SIZE
#include <c10/npu/NPUStream.h> // For getting current NPU stream from PyTorch (example path)
#include <iostream>


// Wrapper for NPU Simple k-NN
at::Tensor distNPU2_py(
    const at::Tensor& points, // Px3 float tensor of points
    uint32_t num_cores_to_use // Number of NPU cores to utilize for kernel launches
) {
    TORCH_CHECK(points.device().type() == c10::DeviceType::PrivateUse1 || points.device().type() == c10::DeviceType::NPU, "Input points tensor must be on NPU device");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "Input points tensor must have shape (P, 3)");
    TORCH_CHECK(points.scalar_type() == at::ScalarType::Float, "Input points tensor must be of type float");

    int P = points.size(0);
    if (P == 0) {
        return torch::empty({0}, points.options());
    }

    auto float_options_npu = points.options();
    at::Tensor mean_dists_out = torch::empty({P}, float_options_npu);

    // Workspace calculation and allocation
    // num_boxes is needed for getWorkspaceSizeBytes, calculate it here.
    uint32_t num_boxes = (P + NPU_KNN_BOX_SIZE - 1) / NPU_KNN_BOX_SIZE;
    size_t workspace_size_bytes = NpuSimpleKNN::KNN::getWorkspaceSizeBytes(P, num_boxes);

    at::Tensor workspace_tensor;
    if (workspace_size_bytes > 0) {
        workspace_tensor = torch::empty(workspace_size_bytes, points.options().dtype(torch::kByte));
    } else {
        // Create a dummy tensor if workspace_size is 0, as data_ptr() on undefined tensor is an error.
        // Or, pass nullptr if the C++ interface handles it.
        // For safety, pass a valid pointer to a tiny allocation if functions expect non-null,
        // but ideally, functions should handle nullptr for zero-size workspace.
        // For now, let's assume if workspace_size_bytes is 0, nullptr is fine.
    }

    void* workspace_gm_ptr = workspace_size_bytes > 0 ? workspace_tensor.data_ptr() : nullptr;

    aclrtStream stream = 0; // Default stream
    // c10::optional<c10_npu::NPUStream> torch_npu_stream = c10_npu::getCurrentNPUStream();
    // if (torch_npu_stream.has_value()) {
    //     stream = torch_npu_stream.value().stream();
    // } else {
    //     std::cerr << "Warning: Could not get current PyTorch NPU stream, using default NPU stream 0." << std::endl;
    // }
    // The above stream retrieval is an example, actual API depends on PyTorch NPU version.
    // If using PyTorch NPU, it usually manages the stream context. Explicitly creating one
    // with aclrtCreateStream might be needed if not within PyTorch's NPU context.
    // For custom ops, often the stream from current context is implicitly used by ACL calls,
    // or explicitly passed if the PyTorch NPU dispatcher provides it.

    NpuSimpleKNN::KNN::knn_npu(
        P,
        points.contiguous().data_ptr<float>(),
        mean_dists_out.data_ptr<float>(),
        workspace_gm_ptr,
        workspace_size_bytes,
        stream,
        num_cores_to_use
    );

    // Synchronization might be needed if subsequent operations depend on this result immediately
    // aclrtSynchronizeStream(stream); // Or PyTorch NPU equivalent torch.npu.synchronize()

    return mean_dists_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { // TORCH_EXTENSION_NAME from setup.py, e.g., simple_knn_npu._C
  m.def("dist_npu", &distNPU2_py, "Compute mean k-NN distances on NPU");
}
