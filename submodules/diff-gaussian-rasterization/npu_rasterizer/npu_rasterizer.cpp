// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

#include "npu_rasterizer.h"
#include "npu_rasterizer_tiling.h" // For optiling::NpuRasterizerTilingData
#include "kernel_operator.h"     // For CALL_KERNEL general macro (if used like add_custom)
                                 // Or directly use aclrtLaunchKernel for more control.
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <iostream> // For error messages


// Define a helper macro for checking ACL/TIK runtime errors
#define NPU_CHECK_ERROR(status, msg) \
    if ((status) != ACL_SUCCESS) { \
        std::cerr << "NPU Error: " << (msg) << " | ACL code: " << (status) << std::endl; \
        /* Potentially throw an exception or handle error appropriately */ \
        throw std::runtime_error(std::string("NPU Error: ") + msg); \
    }


namespace NpuRasterizer
{

// Helper to copy tiling data to a GM_ADDR (workspace or dedicated buffer)
// Returns the size of the tiling data copied.
template<typename TilingDataType>
static size_t copyTilingDataToNpu(
    TilingDataType& tiling_data_host,
    void* tiling_gm_workspace, // Pointer to GM where tiling data will be copied
    size_t workspace_capacity,
    aclrtStream stream)
{
    size_t tiling_data_size = tiling_data_host.GetDataSize();
    if (tiling_data_size > workspace_capacity) {
        NPU_CHECK_ERROR(ACL_ERROR_MEMORY_ADDRESS_UNALIGNED, "Tiling data size exceeds workspace capacity."); // Using a somewhat relevant error code
    }
    NPU_CHECK_ERROR(aclrtMemcpy(tiling_gm_workspace, tiling_data_size, &tiling_data_host, tiling_data_size, ACL_MEMCPY_HOST_TO_DEVICE),
                    "Failed to copy tiling data to NPU.");
    return tiling_data_size;
}


void Rasterizer::preprocessForward(
    int P, int D_sh, int M_sh,
    const float* means3D_gm, const float* scales_gm, const float* rotations_gm,
    const float* opacities_gm, const float* shs_gm,
    const float* cov3D_precomp_gm, const float* colors_precomp_gm,
    const RasterizationParams& params,
    float* means2D_out_gm, float* depths_out_gm, int* radii_out_gm,
    float* cov3Ds_computed_out_gm, float4* conic_opacity_out_gm,
    float* rgb_out_gm, bool* clamped_out_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores_to_use)
{
    optiling::NpuRasterizerTilingData tiling_data;
    // Populate tiling_data from P, D_sh, M_sh, and params
    tiling_data.set_P(P);
    tiling_data.set_D(D_sh);
    tiling_data.set_M(M_sh);
    tiling_data.set_image_width(params.image_width);
    tiling_data.set_image_height(params.image_height);
    tiling_data.set_tan_fovx(params.tan_fovx);
    tiling_data.set_tan_fovy(params.tan_fovy);
    tiling_data.set_focal_x(params.focal_x);
    tiling_data.set_focal_y(params.focal_y);
    tiling_data.set_scale_modifier(params.scale_modifier);
    // ... set other fields like block_x, block_y, core distributions etc.
    // This part needs careful implementation based on the actual tiling logic
    // defined in npu_rasterizer_tiling.cpp's TilingFunc.
    // For now, assuming these are somewhat fixed or have defaults.
    tiling_data.set_block_x(NPU_BLOCK_X); // Example
    tiling_data.set_block_y(NPU_BLOCK_Y); // Example

    // Example: Simplified core distribution (actual logic in TilingFunc is more complex)
    // This is just to show how tiling data gets populated.
    // The real TilingFunc registered with the op would do this.
    // For direct C++ calls, we'd need to replicate that logic or call a helper.
    // Let's assume for now the TilingData struct can be mostly filled from direct params.
    // The complex parts (smallCoreDataNum, etc.) are usually determined by a TilingFunc
    // that has access to PlatformInfo. Here, we might need a simplified version or
    // make num_cores_to_use the primary driver for a simple split.
    uint64_t gaussians_per_core = (P + num_cores_to_use - 1) / num_cores_to_use;
    tiling_data.set_smallCoreDataNum(gaussians_per_core); // Simplified
    tiling_data.set_bigCoreDataNum(gaussians_per_core);   // Simplified
    tiling_data.set_ubPartDataNum(std::min((uint64_t)P, (uint64_t)256)); // Example UB part size
    tiling_data.set_smallCoreLoopNum((gaussians_per_core + tiling_data.ubPartDataNum() -1) / tiling_data.ubPartDataNum());
    tiling_data.set_bigCoreLoopNum(tiling_data.smallCoreLoopNum());
    tiling_data.set_smallCoreTailDataNum(gaussians_per_core % tiling_data.ubPartDataNum());
    if(tiling_data.smallCoreTailDataNum() == 0 && gaussians_per_core > 0) tiling_data.set_smallCoreTailDataNum(tiling_data.ubPartDataNum());
    tiling_data.set_bigCoreTailDataNum(tiling_data.smallCoreTailDataNum());
    tiling_data.set_tailBlockNum(0); // Simplified: no distinction between big/small cores here

    // Copy tiling data to NPU workspace
    // Assuming workspace_gm is large enough and its beginning is used for tiling data.
    copyTilingDataToNpu(tiling_data, workspace_gm, 1024, stream); // Assuming 1KB is enough for tiling data

    // Kernel arguments (GM_ADDR for all pointers)
    // Note: TIK kernels expect GM_ADDR, which are effectively void*.
    // The const_cast might be needed if the GM_ADDR type is not const-correct.
    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_preprocess_forward", // Kernel name string
        num_cores_to_use, stream,
        const_cast<float*>(means3D_gm), const_cast<float*>(scales_gm), const_cast<float*>(rotations_gm),
        const_cast<float*>(opacities_gm), const_cast<float*>(shs_gm),
        clamped_out_gm, // output
        const_cast<float*>(cov3D_precomp_gm), const_cast<float*>(colors_precomp_gm),
        const_cast<float*>(params.viewmatrix), const_cast<float*>(params.projmatrix), const_cast<float*>(params.cam_pos),
        radii_out_gm, means2D_out_gm, depths_out_gm, cov3Ds_computed_out_gm,
        rgb_out_gm, reinterpret_cast<float4*>(conic_opacity_out_gm), // conic_opacity is float4*
        workspace_gm, // Actual workspace for kernel if any, beyond tiling data
        workspace_gm  // Tiling data (copied to the start of workspace)
    ), "Failed to launch npu_preprocess_forward kernel.");
}


void Rasterizer::renderForward(
    const uint32_t* ranges_gm, const uint32_t* point_list_gm,
    const float2* points_xy_image_gm, const float* features_gm,
    const float4* conic_opacity_gm, const bool* compute_locally_gm,
    const float* bg_color_gm,
    const RasterizationParams& params,
    float* final_T_out_gm, uint32_t* n_contrib_out_gm, float* out_color_out_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores_to_use)
{
    optiling::NpuRasterizerTilingData tiling_data;
    // Populate tiling_data relevant for renderForward
    tiling_data.set_image_width(params.image_width);
    tiling_data.set_image_height(params.image_height);
    tiling_data.set_block_x(NPU_BLOCK_X); // From npu_config.h or params
    tiling_data.set_block_y(NPU_BLOCK_Y);
    tiling_data.set_tile_grid_x((params.image_width + NPU_BLOCK_X - 1) / NPU_BLOCK_X);
    tiling_data.set_tile_grid_y((params.image_height + NPU_BLOCK_Y - 1) / NPU_BLOCK_Y);
    // ubPartDataNum for render kernel (how many Gaussians to batch load for a tile's pixel processing)
    // This should be determined by a proper tiling function based on UB size and data per Gaussian.
    tiling_data.set_ubPartDataNum(64); // Example value

    copyTilingDataToNpu(tiling_data, workspace_gm, 1024, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_render_forward",
        num_cores_to_use, stream,
        const_cast<uint32_t*>(ranges_gm), const_cast<uint32_t*>(point_list_gm),
        const_cast<float2*>(points_xy_image_gm), const_cast<float*>(features_gm),
        const_cast<float4*>(conic_opacity_gm),
        final_T_out_gm, n_contrib_out_gm,
        const_cast<float*>(bg_color_gm), out_color_out_gm,
        const_cast<bool*>(compute_locally_gm),
        workspace_gm, // Actual workspace
        workspace_gm  // Tiling data
    ), "Failed to launch npu_render_forward kernel.");
}


void Rasterizer::renderBackward(
    const float* dL_dpixels_gm,
    const uint32_t* ranges_gm, const uint32_t* point_list_gm,
    const float2* points_xy_image_gm, const float4* conic_opacity_gm, const float* colors_gm,
    const float* final_Ts_gm, const uint32_t* n_contrib_gm,
    const float* bg_color_gm, const bool* compute_locally_gm,
    const RasterizationParams& params,
    float* dL_dmean2D_out_gm, float* dL_dconic_out_gm,
    float* dL_dopacity_out_gm, float* dL_dcolors_out_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores_to_use)
{
    optiling::NpuRasterizerTilingData tiling_data;
    // Populate tiling_data (similar to renderForward)
    tiling_data.set_image_width(params.image_width);
    tiling_data.set_image_height(params.image_height);
    tiling_data.set_block_x(NPU_BLOCK_X);
    tiling_data.set_block_y(NPU_BLOCK_Y);
    tiling_data.set_tile_grid_x((params.image_width + NPU_BLOCK_X - 1) / NPU_BLOCK_X);
    tiling_data.set_tile_grid_y((params.image_height + NPU_BLOCK_Y - 1) / NPU_BLOCK_Y);
    tiling_data.set_ubPartDataNum(64); // Example

    copyTilingDataToNpu(tiling_data, workspace_gm, 1024, stream);

    // IMPORTANT: Output gradient buffers (dL_dmean2D_out_gm, etc.) must be zero-initialized by the caller
    // if the kernel uses atomic adds and is launched multiple times accumulating to the same buffers,
    // or if multiple cores write to overlapping regions without proper synchronization/reduction.
    // Assuming for now that each core's output region is distinct or atomics handle accumulation correctly.

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_render_backward",
        num_cores_to_use, stream,
        const_cast<uint32_t*>(ranges_gm), const_cast<uint32_t*>(point_list_gm),
        const_cast<float2*>(points_xy_image_gm), const_cast<float4*>(conic_opacity_gm), const_cast<float*>(colors_gm),
        const_cast<float*>(final_Ts_gm), const_cast<uint32_t*>(n_contrib_gm),
        const_cast<float*>(bg_color_gm), const_cast<float*>(dL_dpixels_gm),
        dL_dmean2D_out_gm, dL_dconic_out_gm, dL_dopacity_out_gm, dL_dcolors_out_gm,
        workspace_gm, // Actual workspace
        workspace_gm  // Tiling data
    ), "Failed to launch npu_render_backward kernel.");
}


void Rasterizer::computeCov2DBackward(
    const float* dL_dconics_gm, const float* means3D_gm, const int* radii_gm,
    const float* cov3Ds_fwd_gm, const RasterizationParams& params,
    float* dL_dmean3D_cov_out_gm, float* dL_dcov3D_out_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores_to_use)
{
    optiling::NpuRasterizerTilingData tiling_data;
    // Populate for cov2d backward (P, camera params, etc.)
    // This kernel is per-Gaussian, so P and core distribution are key.
    // Assuming P is implicitly known by sizing of input arrays, or needs to be passed.
    // For now, let's assume P is derived by the caller and tiling data is set up accordingly.
    // Example: if P is an argument to this C++ function, then:
    // tiling_data.set_P(P_arg);
    // ... and other core/loop params like in preprocessForward.
    tiling_data.set_focal_x(params.focal_x);
    tiling_data.set_focal_y(params.focal_y);
    tiling_data.set_tan_fovx(params.tan_fovx);
    tiling_data.set_tan_fovy(params.tan_fovy);
    // ... set P, core counts, ubPartDataNum etc.

    copyTilingDataToNpu(tiling_data, workspace_gm, 1024, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_compute_cov2d_backward",
        num_cores_to_use, stream,
        const_cast<float*>(means3D_gm), const_cast<int*>(radii_gm), const_cast<float*>(cov3Ds_fwd_gm),
        const_cast<float*>(params.viewmatrix), const_cast<float*>(dL_dconics_gm),
        dL_dmean3D_cov_out_gm, dL_dcov3D_out_gm,
        workspace_gm, workspace_gm
    ), "Failed to launch npu_compute_cov2d_backward kernel.");
}


void Rasterizer::preprocessBackward(
    int P, int D_sh, int M_sh,
    const float* means3D_gm, const float* scales_gm, const float* rotations_gm, const float* shs_fwd_gm,
    const int* radii_fwd_gm, const bool* clamped_fwd_gm,
    const float* cam_pos_gm, const float* projmatrix_gm, float scale_modifier_val,
    const float* dL_dmean2D_render_gm, const float* dL_dmean3D_cov_gm,
    const float* dL_dcolors_render_gm, const float* dL_dcov3D_cov_gm,
    float* dL_dmeans3D_total_out_gm, float* dL_dshs_out_gm,
    float* dL_dscales_out_gm, float* dL_drots_out_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores_to_use)
{
    optiling::NpuRasterizerTilingData tiling_data;
    // Populate for preprocess backward (P, D_sh, M_sh, core dist, etc.)
    tiling_data.set_P(P);
    tiling_data.set_D(D_sh);
    tiling_data.set_M(M_sh);
    tiling_data.set_scale_modifier(scale_modifier_val);
    // ... other core/loop params like in preprocessForward ...

    copyTilingDataToNpu(tiling_data, workspace_gm, 1024, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_preprocess_backward",
        num_cores_to_use, stream,
        const_cast<float*>(means3D_gm), const_cast<float*>(scales_gm), const_cast<float*>(rotations_gm), const_cast<float*>(shs_fwd_gm),
        const_cast<int*>(radii_fwd_gm), const_cast<bool*>(clamped_fwd_gm),
        const_cast<float*>(projmatrix_gm), const_cast<float*>(cam_pos_gm),
        const_cast<float*>(dL_dmean2D_render_gm), const_cast<float*>(dL_dmean3D_cov_gm),
        const_cast<float*>(dL_dcolors_render_gm), const_cast<float*>(dL_dcov3D_cov_gm),
        dL_dmeans3D_total_out_gm, dL_dshs_out_gm, dL_dscales_out_gm, dL_drots_out_gm,
        workspace_gm, workspace_gm
    ), "Failed to launch npu_preprocess_backward kernel.");
}


// Note: markVisible and getDistributionStrategy are not implemented for NPU yet.
// markVisible is a simple frustum culling test, could be a small preprocess kernel if needed.
// getDistributionStrategy was CUDA specific for distributed rendering logic, may not map directly.
void Rasterizer::markVisible(int P, float* मीन्स3डी, float* व्यूमैट्रिक्स, float* प्रोजमैट्रिक्स, bool* प्रेजेंट) {
    // TODO: Implement if necessary for NPU, or if this logic is part of preprocessForward.
    // This was a simple frustum check in CUDA. Can be a small, separate kernel.
    std::cerr << "NpuRasterizer::markVisible not implemented yet." << std::endl;
}


} // namespace NpuRasterizer
