// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

/**
 * @file npu_auxiliary.cpp
 * This file can contain implementations of helper functions declared in npu_auxiliary.h,
 * especially if they are more complex and not suitable for being inline in the header.
 * For many simple math functions, keeping them inline in the header is often preferred
 * for potential performance benefits, especially if they are used by TIK kernels.
 *
 * TIK __aicore__ device functions must typically be defined in the same compilation unit
 * where they are used by kernels, or be templated / inline in headers.
 * This file is more for host-side C++ helpers or if we decide to have non-inline
 * versions of some auxiliary functions that are not directly part of TIK kernels.
 */

#include "npu_auxiliary.h"

// Currently, all functions in npu_auxiliary.h are inline.
// If any of them were complex enough to warrant a separate .cpp implementation,
// their definitions would go here.

// For example, if `transformPoint4x4` was not inline:
/*
float4 transformPoint4x4(const float3& p, const float* matrix) {
    // ... implementation ...
}
*/

// Most of the math helpers are small and benefit from being inlined,
// so this file might remain empty or be used for more complex host-side utilities
// related to the NPU rasterizer setup or data preparation if any arise.
