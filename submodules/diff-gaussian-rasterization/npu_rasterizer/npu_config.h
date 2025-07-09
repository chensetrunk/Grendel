// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

#ifndef NPU_RASTERIZER_CONFIG_H_INCLUDED
#define NPU_RASTERIZER_CONFIG_H_INCLUDED

// These constants are derived from the CUDA version (cuda_rasterizer/config.h)
// They might be adjusted for optimal NPU performance or passed via tiling data.

#define NPU_NUM_CHANNELS 3 // Default 3, RGB

// Block dimensions for tile-based rasterization on NPU
// These values were 16x16 in the CUDA version.
// Optimal values for NPU might differ and could be determined through experimentation
// or passed as TilingData parameters if they need to be dynamic.
#define NPU_BLOCK_X 16
#define NPU_BLOCK_Y 16

// Block size for 1D kernel launches (e.g., preprocessing Gaussians)
// CUDA version used 256. This also might need tuning for NPU.
#define NPU_ONE_DIM_BLOCK_SIZE 256

// Other constants from the CUDA implementation might be defined here if they are static
// and used across multiple NPU kernels. For example, SH coefficients, if not embedded directly.
// However, it's generally better to pass configurable parameters via TilingData.

// Spherical harmonics coefficients (from cuda_rasterizer/auxiliary.h)
// These can be defined as `static constexpr float` or similar in a .h or directly in .cpp
// If used by __aicore__ functions, they might need to be passed or loaded if not compile-time constants.
// For now, keeping them here for reference, but they'll likely move into a .cpp or a shared NPU utility header.
/*
static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
static const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};
*/

#endif // NPU_RASTERIZER_CONFIG_H_INCLUDED
