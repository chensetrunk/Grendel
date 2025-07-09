// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

#ifndef NPU_RASTERIZER_H_INCLUDED
#define NPU_RASTERIZER_H_INCLUDED

#include "acl/acl_rt.h" // For aclrtStream, aclrtMalloc, aclrtMemcpy etc.
#include "npu_config.h" // For NPU_NUM_CHANNELS etc.
#include "npu_auxiliary.h" // For float2, float3, float4, uint2 etc.

// Forward declare TIK kernel launch functions (defined in npu_rasterizer_kernels.cpp)
// These names must match the extern "C" __global__ __aicore__ function names.
extern "C" void npu_preprocess_forward(
    GM_ADDR نقاط_الاصل_gm, GM_ADDR المقاييس_gm, GM_ADDR الدورانات_gm, GM_ADDR العتامات_gm, GM_ADDR shs_gm,
    GM_ADDR المشبك_gm, GM_ADDR cov3D_precomp_gm, GM_ADDR colors_precomp_gm,
    GM_ADDR viewmatrix_gm, GM_ADDR projmatrix_gm, GM_ADDR cam_pos_gm,
    GM_ADDR انصاف_الاقطار_gm, GM_ADDR نقاط_صور_xy_gm, GM_ADDR الاعماق_gm, GM_ADDR cov3Ds_out_gm,
    GM_ADDR rgb_out_gm, GM_ADDR ضبابية_المخروط_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm);

extern "C" void npu_render_forward(
    GM_ADDR ranges_gm_in, GM_ADDR point_list_gm_in,
    GM_ADDR points_xy_image_gm_in, GM_ADDR features_gm_in, GM_ADDR conic_opacity_gm_in,
    GM_ADDR final_T_gm_out, GM_ADDR n_contrib_gm_out,
    GM_ADDR bg_color_gm_in, GM_ADDR out_color_gm_out,
    GM_ADDR compute_locally_gm_in,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm_in);

extern "C" void npu_render_backward(
    GM_ADDR ranges_gm_in, GM_ADDR point_list_gm_in,
    GM_ADDR points_xy_image_gm_in, GM_ADDR conic_opacity_gm_in, GM_ADDR colors_gm_in,
    GM_ADDR final_Ts_gm_in, GM_ADDR n_contrib_gm_in,
    GM_ADDR bg_color_gm_in, GM_ADDR dL_dpixels_gm_in,
    GM_ADDR dL_dmean2D_gm_out, GM_ADDR dL_dconic2D_gm_out, GM_ADDR dL_dopacity_gm_out, GM_ADDR dL_dcolors_gm_out,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm_in);

extern "C" void npu_compute_cov2d_backward(
    GM_ADDR means3D_gm_in, GM_ADDR radii_gm_in, GM_ADDR cov3Ds_fwd_gm_in, GM_ADDR viewmatrix_gm_in,
    GM_ADDR dL_dconics_gm_in, GM_ADDR dL_dmean3D_cov_gm_out, GM_ADDR dL_dcov3D_gm_out,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm_in);

extern "C" void npu_preprocess_backward(
     GM_ADDR means3D_gm_in, GM_ADDR scales_gm_in, GM_ADDR rotations_gm_in, GM_ADDR shs_fwd_gm_in,
     GM_ADDR radii_fwd_gm_in, GM_ADDR clamped_fwd_gm_in, GM_ADDR projmatrix_gm_in, GM_ADDR cam_pos_gm_in,
     GM_ADDR dL_dmean2D_render_gm_in, GM_ADDR dL_dmean3D_cov_gm_in, GM_ADDR dL_dcolors_render_gm_in, GM_ADDR dL_dcov3D_cov_gm_in,
     GM_ADDR dL_dmeans3D_total_gm_out, GM_ADDR dL_dshs_gm_out, GM_ADDR dL_dscales_gm_out, GM_ADDR dL_drots_gm_out,
     GM_ADDR workspace_gm, GM_ADDR tiling_gm_in);


namespace NpuRasterizer
{
	class Rasterizer
	{
	public:
        // struct for parameters that don't change per-Gaussian, but per-invocation
        struct RasterizationParams {
            const float* viewmatrix; // Points to GM buffer for view matrix (16 floats)
            const float* projmatrix; // Points to GM buffer for proj matrix (16 floats)
            const float* cam_pos;    // Points to GM buffer for camera position (3 floats)
            int image_width;
            int image_height;
            float tan_fovx;
            float tan_fovy;
            float focal_x; // Derived: image_width / (2.0f * tan_fovx)
            float focal_y; // Derived: image_height / (2.0f * tan_fovy)
            float scale_modifier;
            bool prefiltered; // If input Gaussians are already filtered
            // Add other global settings like debug flags if necessary
        };

        // PreprocessForward: Takes raw Gaussian parameters, computes intermediate representations.
        static void preprocessForward(
            // Inputs: Gaussian parameters
            int P, int D_sh, int M_sh, // Num Gaussians, SH degree, SH max coeffs
            const float* means3D_gm,      // float3 array [P]
            const float* scales_gm,       // float3 array [P]
            const float* rotations_gm,    // float4 array [P]
            const float* opacities_gm,    // float array [P]
            const float* shs_gm,          // float array [P * M_sh * 3], can be nullptr
            const float* cov3D_precomp_gm,// float array [P * 6], can be nullptr
            const float* colors_precomp_gm,// float array [P * 3], can be nullptr

            // Input: Rasterization settings & camera
            const RasterizationParams& params,

            // Outputs: Intermediate per-Gaussian data for rendering & backward pass
            float* means2D_out_gm,        // float2 array [P]
            float* depths_out_gm,         // float array [P]
            int* radii_out_gm,            // int array [P]
            float* cov3Ds_computed_out_gm,// float array [P * 6] (if not precomputed)
            float4* conic_opacity_out_gm, // float4 array [P]
            float* rgb_out_gm,            // float array [P * 3] (if not precomputed)
            bool* clamped_out_gm,         // bool array [P * 3]

            // Workspace & Stream
            void* workspace_gm,          // For tiling data, potentially other temp NPU buffers
            aclrtStream stream,
            uint32_t num_cores_to_use // blockDim for kernel launch
        );

        // RenderForward: Takes preprocessed Gaussians, renders image.
        static void renderForward(
            // Inputs: Data from binning/sorting and preprocessForward
            const uint32_t* ranges_gm,          // uint2 per tile: [start_idx, end_idx) in point_list
            const uint32_t* point_list_gm,      // sorted global Gaussian indices per tile
            const float2* points_xy_image_gm,   // From preprocess: screen XY for each Gaussian
            const float* features_gm,           // From preprocess: RGB or other features per Gaussian
            const float4* conic_opacity_gm,     // From preprocess: conic and opacity per Gaussian
            const bool* compute_locally_gm,     // Optional: map of tiles for this rank to compute
            const float* bg_color_gm,           // Background color (NUM_CHANNELS)

            // Input: Rasterization settings
            const RasterizationParams& params,

            // Outputs: Rendered image and auxiliary buffers
            float* final_T_out_gm,              // Transmittance per pixel [H * W]
            uint32_t* n_contrib_out_gm,         // Last contributing Gaussian index per pixel [H * W]
            float* out_color_out_gm,            // Final image [C * H * W]

            void* workspace_gm,
            aclrtStream stream,
            uint32_t num_cores_to_use
        );

        // Backward pass for rendering stage
        static void renderBackward(
            // Inputs: Gradients from loss, and data from forward passes
            const float* dL_dpixels_gm,         // Gradient of loss w.r.t. output pixels [C * H * W]
            // From forward binning & preprocess:
            const uint32_t* ranges_gm,
            const uint32_t* point_list_gm,
            const float2* points_xy_image_gm,
            const float4* conic_opacity_gm,
            const float* colors_gm,             // RGB features used in forward render
            // From forward render:
            const float* final_Ts_gm,
            const uint32_t* n_contrib_gm,
            // Other inputs:
            const float* bg_color_gm,
            const bool* compute_locally_gm,     // Optional
            const RasterizationParams& params,

            // Outputs: Gradients for inputs of forward render
            float* dL_dmean2D_out_gm,           // float2 per Gaussian [P]
            float* dL_dconic_out_gm,            // float3 per Gaussian (for conic a,b,c)
            float* dL_dopacity_out_gm,          // float per Gaussian
            float* dL_dcolors_out_gm,           // float3 per Gaussian

            void* workspace_gm,
            aclrtStream stream,
            uint32_t num_cores_to_use
        );

        // Backward pass for 2D covariance computation
        static void computeCov2DBackward(
            // Inputs: Gradients from renderBackward, and data from forward passes
            const float* dL_dconics_gm,         // float3 per Gaussian (dL/da, dL/db, dL/dc)
            const float* means3D_gm,            // Original world space means
            const int* radii_gm,                // Radii from forward preprocess (for culling)
            const float* cov3Ds_fwd_gm,         // 3D covariances from forward preprocess
            const RasterizationParams& params,

            // Outputs: Gradients
            float* dL_dmean3D_cov_out_gm,       // Contribution to dL/dMean3D from this path
            float* dL_dcov3D_out_gm,            // dL/dCov3D (6 components)

            void* workspace_gm,
            aclrtStream stream,
            uint32_t num_cores_to_use
        );

        // Backward pass for preprocessing stage (main part)
        static void preprocessBackward(
            // Inputs: Original Gaussian parameters, forward intermediates, and various gradients
            int P, int D_sh, int M_sh,
            const float* means3D_gm,
            const float* scales_gm,
            const float* rotations_gm,
            const float* shs_fwd_gm,
            const int* radii_fwd_gm,
            const bool* clamped_fwd_gm,
            const float* cam_pos_gm, // Only cam_pos and projmatrix needed from RasterizationParams here
            const float* projmatrix_gm,
            float scale_modifier_val, // Pass as value

            const float* dL_dmean2D_render_gm,    // From render_backward
            const float* dL_dmean3D_cov_gm,       // From cov2d_backward
            const float* dL_dcolors_render_gm,    // From render_backward (dL w.r.t. RGB colors)
            const float* dL_dcov3D_cov_gm,        // From cov2d_backward

            // Outputs: Final gradients for learnable parameters
            float* dL_dmeans3D_total_out_gm,
            float* dL_dshs_out_gm,
            float* dL_dscales_out_gm,
            float* dL_drots_out_gm,

            void* workspace_gm,
            aclrtStream stream,
            uint32_t num_cores_to_use
        );

	};
} // namespace NpuRasterizer

#endif // NPU_RASTERIZER_H_INCLUDED
