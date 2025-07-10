// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

#ifndef NPU_RASTERIZER_AUXILIARY_H_INCLUDED
#define NPU_RASTERIZER_AUXILIARY_H_INCLUDED

#include "npu_config.h"
// For NPU_BLOCK_X, NPU_BLOCK_Y etc. if still needed globally,
// though many should come from TilingData.

// Includes for Ascend C math functions or types if necessary
// For example, if using specific NPU vector types or intrinsics.
// #include <aclnn/aclnn_math.h> // Or similar headers

// Standard C++ math might be sufficient for many operations.
#include <cmath> // For std::sqrt, std::exp, std::min, std::max
#include <algorithm> // For std::min, std::max

// Define float2, float3, float4, uint2 etc. if not provided by a common NPU header.
// These are simple structs often used in graphics code.
// TIK might have its own vector types, or simple structs can be used.
// For now, defining basic ones. GLM types were used in CUDA version,
// but for direct NPU code, simpler structs or NPU specific types are better.

struct float2 {
    float x, y;
};

struct float3 {
    float x, y, z;
};

struct float4 {
    float x, y, z, w;
};

struct uint2 {
    unsigned int x, y;
};

struct uint3 { // If needed
    unsigned int x, y, z;
};

struct dim3_npu { // Naming to avoid conflict if dim3 is defined elsewhere
    unsigned int x, y, z;
};


// Constants for Spherical Harmonics (moved here for potential kernel use)
// These should be `static constexpr` if used in templates or header-only functions.
// If used in .cpp files, `const` is fine.
// For __aicore__ device code, how these are accessed needs care.
// They might need to be loaded into registers or local memory if too large or numerous.

static constexpr float SH_C0 = 0.28209479177387814f;
static constexpr float SH_C1 = 0.4886025119029199f;
// For arrays, they might need to be passed as arguments or be part of a constant memory segment accessible by NPU.
// For simplicity in a header, small arrays can be constexpr.
static constexpr float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
static constexpr float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};


// NPU-compatible helper functions (prototypes)
// Implementations will be in npu_auxiliary.cpp or inline if simple.

// Example: Convert NDC to Pixel coordinates
// __aicore__ or __aicore__ inline if it's a device function for TIK kernels
// For host-side C++ (like tiling funcs), just inline or regular function.
__aicore__ inline float ndc2Pix(float v, int S) {
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

// Get bounding rectangle for a 2D point with a radius
// `grid` here would be tile_grid dimensions.
// block_x_dim, block_y_dim would be NPU_BLOCK_X, NPU_BLOCK_Y (or from tiling)
__aicore__ inline void getRect(
    const float2 p,
    int max_radius,
    uint2& rect_min,
    uint2& rect_max,
    const dim3_npu grid_dim,
    unsigned int block_x_dim,
    unsigned int block_y_dim)
{
    rect_min.x = std::min(grid_dim.x, std::max(0, (int)((p.x - max_radius) / block_x_dim)));
    rect_min.y = std::min(grid_dim.y, std::max(0, (int)((p.y - max_radius) / block_y_dim)));

    rect_max.x = std::min(grid_dim.x, std::max(0, (int)((p.x + max_radius + block_x_dim - 1) / block_x_dim)));
    rect_max.y = std::min(grid_dim.y, std::max(0, (int)((p.y + max_radius + block_y_dim - 1) / block_y_dim)));
}


// Matrix transformation functions (4x4, 4x3)
// These will operate on simple float arrays representing matrices (row-major or col-major)
// and our float3/float4 structs.
// Example: transformPoint4x4
__aicore__ inline float4 transformPoint4x4(const float3& p, const float* matrix) {
    // Assuming matrix is row-major:
    // m[0] m[1] m[2] m[3]
    // m[4] m[5] m[6] m[7]
    // m[8] m[9] m[10] m[11]
    // m[12] m[13] m[14] m[15]
    // If col-major (like OpenGL/GLM default), access pattern changes: matrix[0], matrix[4], matrix[8], matrix[12] for first row of calculation
    // The CUDA code used: matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z + matrix[12]
    // This implies the matrix was passed effectively column-major or indices are for column-major access.
    // Let's stick to that convention for direct translation.
    float4 transformed;
    transformed.x = matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12];
    transformed.y = matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13];
    transformed.z = matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14];
    transformed.w = matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15];
    return transformed;
}

__aicore__ inline float3 transformPoint4x3(const float3& p, const float* matrix) {
    // Assuming column-major like access (matrix has 12 elements for a 4x3 projection)
    // m[0] m[1] m[2]
    // m[3] m[4] m[5]
    // m[6] m[7] m[8]
    // m[9] m[10] m[11]  -- these are translation components
    // The CUDA code: matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z + matrix[12]
    // This means the CUDA matrix was likely a float[16] (4x4) but only first 3 rows used, or it's a float[12] for affine.
    // If viewmatrix (float* viewmatrix) is float[16], then matrix[12], matrix[13], matrix[14] are translation.
    float3 transformed;
    transformed.x = matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12];
    transformed.y = matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13];
    transformed.z = matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14];
    return transformed;
}


// Sigmoid function
__aicore__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Frustum culling check
// `prefiltered` flag might not be relevant initially for TIK version.
// `p_view` is an output parameter (point in view space).
// This function will be used by the preprocess kernel.
__aicore__ inline bool in_frustum(
    // int idx, // Gaussian index, not directly needed by this math function if p_orig is passed
    const float3 p_orig, // Original point coordinates
    const float* viewmatrix, // 4x4 view matrix (16 floats)
    const float* projmatrix, // 4x4 projection matrix (16 floats)
    // bool prefiltered, // We might ignore this for now
    float3& p_view       // Output: point in view space
) {
    // Bring points to screen space
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f); // Add epsilon for stability
    // float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // Projected point (NDC like)

    p_view = transformPoint4x3(p_orig, viewmatrix);

    // Near plane culling (original used 0.2f, might need adjustment)
    // Far plane culling is implicitly handled by p_hom.w > 0 and p_proj.z < 1 (or similar)
    // View frustum culling (original used -1.3 to 1.3 in NDC-like space for x,y)
    // A common check is -p_hom.w <= p_hom.x <= p_hom.w and -p_hom.w <= p_hom.y <= p_hom.w
    // and near_plane_dist_val <= p_hom.w (or view_z) <= far_plane_dist_val

    if (p_view.z <= 0.2f) { // Basic near plane culling in view space
        return false;
    }

    // More robust culling against normalized device coordinates (NDC)
    // Check if x, y, z are within [-w, w] for perspective projection before division.
    bool culled = (p_hom.x < -p_hom.w || p_hom.x > p_hom.w ||
                   p_hom.y < -p_hom.w || p_hom.y > p_hom.w ||
                   p_hom.w < 0.0f ); // W check also covers points behind camera if not caught by p_view.z

    return !culled;
}


// Add other math utilities as needed (e.g., for covariance, SH, gradients)

// Vector transformation for directional vectors (no translation)
__aicore__ inline float3 transformVec4x3(const float3& v, const float* matrix) {
    // Assuming column-major like access for rotation part of a 4x4 matrix
    float3 transformed;
    transformed.x = matrix[0] * v.x + matrix[4] * v.y + matrix[8] * v.z;
    transformed.y = matrix[1] * v.x + matrix[5] * v.y + matrix[9] * v.z;
    transformed.z = matrix[2] * v.x + matrix[6] * v.y + matrix[10] * v.z;
    return transformed;
}

// Transposed vector transformation (useful for transforming normals with inverse transpose)
// Or, as used in CUDA code, for transforming gradients.
__aicore__ inline float3 transformVec4x3Transpose(const float3& v, const float* matrix) {
    // Effectively multiplying by the transpose of the 3x3 submatrix
    // matrix is still accessed as if it's column-major storage of a 4x4
    // Original:
    //   transformed.x = matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z;
    //   transformed.y = matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z;
    //   transformed.z = matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z;
    float3 transformed;
    transformed.x = matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z; // Row 0 of K^T . v
    transformed.y = matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z; // Row 1 of K^T . v
    transformed.z = matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z; // Row 2 of K^T . v
    return transformed;
}


// Gradient of |v|/|v|^3 dotted with dv, specifically for z component.
// Used in SH gradient calculation w.r.t mean.
// v: original vector (e.g., view direction)
// dv: gradient vector (dL_ddir)
__aicore__ inline float dnormvdz(float3 v, float3 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (sum2 == 0.0f) return 0.0f; // Avoid division by zero
    float invsum32 = 1.0f / (sum2 * std::sqrt(sum2)); // 1 / |v|^3
    float dnorm_val = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnorm_val;
}

// Gradient of v/|v| dotted with dv. (d(v/|v|)/dv_k * dv_k)
// v: original vector
// dv: gradient vector (dL_ddir)
__aicore__ inline float3 dnormvdv(float3 v, float3 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (sum2 == 0.0f) return {0.0f, 0.0f, 0.0f}; // Avoid division by zero
    float invsum32 = 1.0f / (sum2 * std::sqrt(sum2)); // 1 / |v|^3

    float3 dnorm_val;
    // Derivative of (v_i / |v|) w.r.t v_j is (delta_ij * |v|^2 - v_i * v_j) / |v|^3
    // dL/dv_j = sum_i (dL/d(normalized_v_i) * d(normalized_v_i)/dv_j)
    // Here, dv is dL/d(normalized_v)
    dnorm_val.x = ((sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
    dnorm_val.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
    dnorm_val.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnorm_val;
}

// Overload for float4 (though typically dL_drot in CUDA code was float4, its norm operations might be specific)
// The CUDA code `dnormvdv(float4 v, float4 dv)` was used for quaternion gradients.
// A quaternion q = (r, x, y, z) has |q|^2 = r^2+x^2+y^2+z^2.
// The normalization derivative logic remains similar.
__aicore__ inline float4 dnormvdv(float4 v, float4 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    if (sum2 == 0.0f) return {0.0f, 0.0f, 0.0f, 0.0f}; // Avoid division by zero
    float invsum32 = 1.0f / (sum2 * std::sqrt(sum2)); // 1 / |v|^3

    // vdv_sum = v.x*dv.x + v.y*dv.y + v.z*dv.z + v.w*dv.w (which is dot(v, dv))
    // The formula ( (sum2 - v_i*v_i)*dv_i - v_i * (dot(v,dv) - v_i*dv_i) ) * invsum32 simplifies to
    // ( sum2*dv_i - v_i * dot(v,dv) ) * invsum32 which is ( (|v|^2 Identity - v outer_product v) dv ) / |v|^3
    // Let's re-verify the CUDA code's formula:
    // dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
    // This is correct. vdv_sum - vdv.x is sum of v_k*dv_k for k != x.
    float vdv_x = v.x * dv.x;
    float vdv_y = v.y * dv.y;
    float vdv_z = v.z * dv.z;
    float vdv_w = v.w * dv.w;
    float vdv_sum = vdv_x + vdv_y + vdv_z + vdv_w;

    float4 dnorm_val;
    dnorm_val.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv_x)) * invsum32;
    dnorm_val.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv_y)) * invsum32;
    dnorm_val.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv_z)) * invsum32;
    dnorm_val.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv_w)) * invsum32;
    return dnorm_val;
}


#endif // NPU_RASTERIZER_AUXILIARY_H_INCLUDED
