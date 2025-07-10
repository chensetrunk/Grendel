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
#include "npu_rasterizer/npu_rasterizer.h" // Changed include
#include "npu_rasterizer/npu_config.h"   // For NPU_NUM_CHANNELS etc.
#include <acl/acl_rt.h> // For aclrtStream and other ACL types/functions
#include <iostream>

// Assuming PyTorch NPU backend provides a way to get current stream
// This is a placeholder; actual API might differ or require specific NPU PyTorch headers
// For example, in some PyTorch NPU versions: #include <torch_npu/csrc/core/npu/NPUStream.h>
// aclrtStream getCurrentNPUStream() {
//     return c10_npu::getCurrentNPUStream().stream();
// }
// If not available, use 0 for default stream or manage stream explicitly.
// For now, using a passed-in or default stream. A common practice is to use the default stream (0).
#define CURRENT_NPU_STREAM 0 // Placeholder for default stream

// Helper to create RasterizationParams from PyTorch inputs
NpuRasterizer::Rasterizer::RasterizationParams createParamsFromArgs(
    const at::Tensor& viewmatrix, const at::Tensor& projmatrix, const at::Tensor& cam_pos,
    int W, int H, float focal_x, float focal_y, float scale_modifier, bool prefiltered) {

    NpuRasterizer::Rasterizer::RasterizationParams p;
    p.viewmatrix = viewmatrix.data_ptr<float>();
    p.projmatrix = projmatrix.data_ptr<float>();
    p.cam_pos = cam_pos.data_ptr<float>();
    p.image_width = W;
    p.image_height = H;
    p.focal_x = focal_x;
    p.focal_y = focal_y;
    p.tan_fovx = W / (2.0f * focal_x); // tan(fovx/2) = (W/2) / focal_x
    p.tan_fovy = H / (2.0f * focal_y); // tan(fovy/2) = (H/2) / focal_y
    p.scale_modifier = scale_modifier;
    p.prefiltered = prefiltered;
    return p;
}


// Wrapper for NPU Preprocess Forward
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
preprocess_gaussians_npu(
    int P, int D_sh, int M_sh,
    const at::Tensor& means3D, const at::Tensor& scales, const at::Tensor& rotations,
    const at::Tensor& opacities, const at::Tensor& shs,
    const at::Tensor& cov3D_precomp, const at::Tensor& colors_precomp,
    const at::Tensor& viewmatrix, const at::Tensor& projmatrix, const at::Tensor& cam_pos,
    int W, int H, float focal_x, float focal_y, float scale_modifier, bool prefiltered,
    void* workspace_ptr, uint32_t num_cores)
{
    // Allocate output tensors (on NPU)
    auto options_float = means3D.options().dtype(torch::kFloat32);
    auto options_int = means3D.options().dtype(torch::kInt32);
    auto options_bool = means3D.options().dtype(torch::kBool);

    at::Tensor means2D_out = torch::empty({P, 2}, options_float);
    at::Tensor depths_out = torch::empty({P}, options_float);
    at::Tensor radii_out = torch::empty({P}, options_int);
    at::Tensor cov3Ds_computed_out = torch::empty({P, 6}, options_float);
    at::Tensor conic_opacity_out = torch::empty({P, 4}, options_float);
    at::Tensor rgb_out = torch::empty({P, 3}, options_float);
    at::Tensor clamped_out = torch::empty({P, 3}, options_bool);

    NpuRasterizer::Rasterizer::RasterizationParams params = createParamsFromArgs(
        viewmatrix, projmatrix, cam_pos, W, H, focal_x, focal_y, scale_modifier, prefiltered);

    NpuRasterizer::Rasterizer::preprocessForward(
        P, D_sh, M_sh,
        means3D.data_ptr<float>(), scales.data_ptr<float>(), rotations.data_ptr<float>(),
        opacities.data_ptr<float>(), shs.defined() ? shs.data_ptr<float>() : nullptr,
        cov3D_precomp.defined() ? cov3D_precomp.data_ptr<float>() : nullptr,
        colors_precomp.defined() ? colors_precomp.data_ptr<float>() : nullptr,
        params,
        means2D_out.data_ptr<float>(), depths_out.data_ptr<float>(), radii_out.data_ptr<int>(),
        cov3Ds_computed_out.data_ptr<float>(), reinterpret_cast<NpuRasterizer::float4*>(conic_opacity_out.data_ptr<float>()),
        rgb_out.data_ptr<float>(), clamped_out.data_ptr<bool>(),
        workspace_ptr, (aclrtStream)CURRENT_NPU_STREAM, num_cores
    );
    return std::make_tuple(means2D_out, depths_out, radii_out, cov3Ds_computed_out, conic_opacity_out, rgb_out, clamped_out);
}

// Wrapper for NPU Render Forward
std::tuple<at::Tensor, at::Tensor, at::Tensor>
render_gaussians_npu(
    const at::Tensor& ranges, const at::Tensor& point_list,
    const at::Tensor& means2D, const at::Tensor& features, const at::Tensor& conic_opacity,
    const at::Tensor& compute_locally, const at::Tensor& bg_color,
    const at::Tensor& viewmatrix, const at::Tensor& projmatrix, const at::Tensor& cam_pos, // For params
    int W, int H, float focal_x, float focal_y, float scale_modifier, bool prefiltered, // For params
    void* workspace_ptr, uint32_t num_cores)
{
    auto options_float = means2D.options().dtype(torch::kFloat32);
    auto options_uint32 = means2D.options().dtype(torch::kUInt32); // n_contrib is uint32

    at::Tensor out_color = torch::empty({NPU_NUM_CHANNELS, H, W}, options_float);
    at::Tensor final_T = torch::empty({H, W}, options_float);
    at::Tensor n_contrib = torch::empty({H, W}, options_uint32);

    NpuRasterizer::Rasterizer::RasterizationParams params = createParamsFromArgs(
        viewmatrix, projmatrix, cam_pos, W, H, focal_x, focal_y, scale_modifier, prefiltered);

    NpuRasterizer::Rasterizer::renderForward(
        ranges.data_ptr<uint32_t>(), point_list.data_ptr<uint32_t>(),
        reinterpret_cast<const NpuRasterizer::float2*>(means2D.data_ptr<float>()),
        features.data_ptr<float>(),
        reinterpret_cast<const NpuRasterizer::float4*>(conic_opacity.data_ptr<float>()),
        compute_locally.defined() ? compute_locally.data_ptr<bool>() : nullptr,
        bg_color.data_ptr<float>(),
        params,
        final_T.data_ptr<float>(), n_contrib.data_ptr<uint32_t>(), out_color.data_ptr<float>(),
        workspace_ptr, (aclrtStream)CURRENT_NPU_STREAM, num_cores
    );
    return std::make_tuple(out_color, final_T, n_contrib);
}

// Wrapper for NPU Render Backward
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
render_gaussians_backward_npu(
    const at::Tensor& dL_dpixels,
    const at::Tensor& ranges, const at::Tensor& point_list,
    const at::Tensor& points_xy_image, const at::Tensor& conic_opacity, const at::Tensor& colors,
    const at::Tensor& final_Ts, const at::Tensor& n_contrib,
    const at::Tensor& bg_color, const at::Tensor& compute_locally,
    const at::Tensor& viewmatrix, const at::Tensor& projmatrix, const at::Tensor& cam_pos, // For params
    int W, int H, float focal_x, float focal_y, float scale_modifier, bool prefiltered, // For params
    int P, // Number of Gaussians for sizing output gradient tensors
    void* workspace_ptr, uint32_t num_cores)
{
    auto options_float = dL_dpixels.options().dtype(torch::kFloat32);
    at::Tensor dL_dmean2D = torch::zeros({P, 2}, options_float); // Zero-initialize for accumulation
    at::Tensor dL_dconic = torch::zeros({P, 3}, options_float);
    at::Tensor dL_dopacity = torch::zeros({P}, options_float);
    at::Tensor dL_dcolors = torch::zeros({P, NPU_NUM_CHANNELS}, options_float);

    NpuRasterizer::Rasterizer::RasterizationParams params = createParamsFromArgs(
        viewmatrix, projmatrix, cam_pos, W, H, focal_x, focal_y, scale_modifier, prefiltered);

    NpuRasterizer::Rasterizer::renderBackward(
        dL_dpixels.data_ptr<float>(),
        ranges.data_ptr<uint32_t>(), point_list.data_ptr<uint32_t>(),
        reinterpret_cast<const NpuRasterizer::float2*>(points_xy_image.data_ptr<float>()),
        reinterpret_cast<const NpuRasterizer::float4*>(conic_opacity.data_ptr<float>()),
        colors.data_ptr<float>(),
        final_Ts.data_ptr<float>(), n_contrib.data_ptr<uint32_t>(),
        bg_color.data_ptr<float>(),
        compute_locally.defined() ? compute_locally.data_ptr<bool>() : nullptr,
        params,
        dL_dmean2D.data_ptr<float>(), dL_dconic.data_ptr<float>(),
        dL_dopacity.data_ptr<float>(), dL_dcolors.data_ptr<float>(),
        workspace_ptr, (aclrtStream)CURRENT_NPU_STREAM, num_cores
    );
    return std::make_tuple(dL_dmean2D, dL_dconic, dL_dopacity, dL_dcolors);
}

// Wrapper for NPU ComputeCov2D Backward
std::tuple<at::Tensor, at::Tensor>
compute_cov2d_backward_npu(
    const at::Tensor& dL_dconics, // Px3 dL/d(conic_a,b,c)
    const at::Tensor& means3D, const at::Tensor& radii, const at::Tensor& cov3Ds_fwd,
    const at::Tensor& viewmatrix, const at::Tensor& projmatrix, const at::Tensor& cam_pos, // For params
    int W, int H, float focal_x, float focal_y, float scale_modifier, bool prefiltered, // For params
    int P,
    void* workspace_ptr, uint32_t num_cores)
{
    auto options_float = means3D.options().dtype(torch::kFloat32);
    at::Tensor dL_dmean3D_cov = torch::zeros({P,3}, options_float); // Zero initialize
    at::Tensor dL_dcov3D = torch::zeros({P,6}, options_float);      // Zero initialize

    NpuRasterizer::Rasterizer::RasterizationParams params = createParamsFromArgs(
        viewmatrix, projmatrix, cam_pos, W, H, focal_x, focal_y, scale_modifier, prefiltered);

    NpuRasterizer::Rasterizer::computeCov2DBackward(
        dL_dconics.data_ptr<float>(), means3D.data_ptr<float>(), radii.data_ptr<int>(),
        cov3Ds_fwd.data_ptr<float>(), params,
        dL_dmean3D_cov.data_ptr<float>(), dL_dcov3D.data_ptr<float>(),
        workspace_ptr, (aclrtStream)CURRENT_NPU_STREAM, num_cores
    );
    return std::make_tuple(dL_dmean3D_cov, dL_dcov3D);
}


// Wrapper for NPU Preprocess Backward
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
preprocess_gaussians_backward_npu(
    int P, int D_sh, int M_sh,
    const at::Tensor& means3D, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& shs_fwd,
    const at::Tensor& radii_fwd, const at::Tensor& clamped_fwd,
    const at::Tensor& cam_pos, const at::Tensor& projmatrix, float scale_modifier_val,
    const at::Tensor& dL_dmean2D_render, const at::Tensor& dL_dmean3D_cov,
    const at::Tensor& dL_dcolors_render, const at::Tensor& dL_dcov3D_cov,
    void* workspace_ptr, uint32_t num_cores)
{
    auto options_float = means3D.options().dtype(torch::kFloat32);
    at::Tensor dL_dmeans3D_total = torch::zeros({P,3}, options_float);
    at::Tensor dL_dshs = torch::zeros({P, M_sh, 3}, options_float); // Or P * M_sh * 3 if flattened
    at::Tensor dL_dscales = torch::zeros({P,3}, options_float);
    at::Tensor dL_drots = torch::zeros({P,4}, options_float);

    NpuRasterizer::Rasterizer::preprocessBackward(
        P, D_sh, M_sh,
        means3D.data_ptr<float>(), scales.data_ptr<float>(), rotations.data_ptr<float>(), shs_fwd.data_ptr<float>(),
        radii_fwd.data_ptr<int>(), clamped_fwd.data_ptr<bool>(),
        cam_pos.data_ptr<float>(), projmatrix.data_ptr<float>(), scale_modifier_val,
        dL_dmean2D_render.data_ptr<float>(), dL_dmean3D_cov.data_ptr<float>(),
        dL_dcolors_render.data_ptr<float>(), dL_dcov3D_cov.data_ptr<float>(),
        dL_dmeans3D_total.data_ptr<float>(), dL_dshs.data_ptr<float>(),
        dL_dscales.data_ptr<float>(), dL_drots.data_ptr<float>(),
        workspace_ptr, (aclrtStream)CURRENT_NPU_STREAM, num_cores
    );
    return std::make_tuple(dL_dmeans3D_total, dL_dshs, dL_dscales, dL_drots);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { // TORCH_EXTENSION_NAME will be diff_gaussian_rasterization._C_npu
  m.def("preprocess_gaussians", &preprocess_gaussians_npu);
  m.def("render_gaussians", &render_gaussians_npu);
  m.def("render_gaussians_backward", &render_gaussians_backward_npu);
  m.def("compute_cov2d_backward", &compute_cov2d_backward_npu);
  m.def("preprocess_gaussians_backward", &preprocess_gaussians_backward_npu);

  // mark_visible, get_distribution_strategy etc. are not directly mapped yet.
  // They might be integrated into preprocess or need separate NPU implementations if still required.
  // GetBlockXY is a config query, can be implemented if NPU_BLOCK_X/Y are fixed or from config.
   m.def("get_block_XY", []() {
        return std::make_tuple(NPU_BLOCK_X, NPU_BLOCK_Y);
    });
}