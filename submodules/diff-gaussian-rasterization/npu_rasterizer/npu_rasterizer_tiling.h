// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

/**
 * @file npu_rasterizer_tiling.h
 * This file will define the tiling data structure for the NPU rasterizer.
 */
#ifndef NPU_RASTERIZER_TILING_H
#define NPU_RASTERIZER_TILING_H

#include "register/tilingdata_base.h" // From add_custom

namespace optiling {

// Define the TilingData structure for the NPU rasterizer.
// This will contain parameters needed by the kernels, such as:
// - Number of Gaussians (P)
// - Image dimensions (W, H)
// - Camera parameters (focal_x, focal_y, tan_fovx, tan_fovy)
// - Pointers or offsets for input/output data
// - Core assignments, loop counts, data partition sizes per core
// This is a placeholder and will be expanded significantly.
BEGIN_TILING_DATA_DEF(NpuRasterizerTilingData)
  // Parameters from cuda_rasterizer/config.h
  TILING_DATA_FIELD_DEF(uint32_t, num_channels); // NUM_CHANNELS
  TILING_DATA_FIELD_DEF(uint32_t, block_x);      // BLOCK_X
  TILING_DATA_FIELD_DEF(uint32_t, block_y);      // BLOCK_Y

  // Basic parameters
  TILING_DATA_FIELD_DEF(uint32_t, P); // Number of Gaussians
  TILING_DATA_FIELD_DEF(uint32_t, D); // Degree of Spherical Harmonics
  TILING_DATA_FIELD_DEF(uint32_t, M); // Max SH coefficients
  TILING_DATA_FIELD_DEF(uint32_t, image_width);
  TILING_DATA_FIELD_DEF(uint32_t, image_height);

  // Camera parameters might also be part of tiling or passed directly
  TILING_DATA_FIELD_DEF(float, focal_x);
  TILING_DATA_FIELD_DEF(float, focal_y);
  TILING_DATA_FIELD_DEF(float, tan_fovx);
  TILING_DATA_FIELD_DEF(float, tan_fovy);

  // Tiling strategy related parameters (similar to add_custom_tiling.h)
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum); // Number of Gaussians per small core (example)
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);   // Number of Gaussians per big core (example)
  TILING_DATA_FIELD_DEF(uint64_t, ubPartDataNum);    // Data num for UB part (example)
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum); // Number of blocks with more data (big cores)

  // Add more fields as needed for different kernels (preprocess, render)
  // and for forward/backward passes.
  // For example, for the render kernel, parameters related to tile division:
  TILING_DATA_FIELD_DEF(uint32_t, tile_grid_x); // Number of tiles in x (for render kernels)
  TILING_DATA_FIELD_DEF(uint32_t, tile_grid_y); // Number of tiles in y (for render kernels)

  // Parameters for per-core tile processing in render kernels
  TILING_DATA_FIELD_DEF(uint32_t, core_tile_start_idx_flat); // Flat start index of tiles for this core
  TILING_DATA_FIELD_DEF(uint32_t, core_num_tiles_to_process); // Number of tiles this core handles

  // Specific UB part data numbers for different kernels if they vary significantly
  // If they are mostly the same, one ubPartDataNum can be used, interpreted by context.
  // Or, could have ubPartDataNum_preprocess, ubPartDataNum_render etc.
  // For now, using one ubPartDataNum and assuming kernel context or further tiling attributes clarify.

  TILING_DATA_FIELD_DEF(float, scale_modifier); // From RasterizationParams

END_TILING_DATA_DEF;

// Register TilingData for each specific kernel type that will have an OpDef.
// This links the string name used in OpDef registration to this TilingData structure.
REGISTER_TILING_DATA_CLASS(NpuRasterizerPreprocessForwardOp, NpuRasterizerTilingData)
REGISTER_TILING_DATA_CLASS(NpuRasterizerRenderForwardOp, NpuRasterizerTilingData)
REGISTER_TILING_DATA_CLASS(NpuRasterizerRenderBackwardOp, NpuRasterizerTilingData)
REGISTER_TILING_DATA_CLASS(NpuRasterizerComputeCov2DBackwardOp, NpuRasterizerTilingData)
REGISTER_TILING_DATA_CLASS(NpuRasterizerPreprocessBackwardOp, NpuRasterizerTilingData)


} // namespace optiling

#endif // NPU_RASTERIZER_TILING_H
