// npu_simple_knn_auxiliary.h

#ifndef NPU_SIMPLE_KNN_AUXILIARY_H_INCLUDED
#define NPU_SIMPLE_KNN_AUXILIARY_H_INCLUDED

#include "npu_simple_knn_config.h" // For NPU_KNN_K etc.
#include <cmath>     // For std::sqrt, std::abs, std::min, std::max
#include <algorithm> // For std::min, std::max

// Basic data structures (if not using a common NPU math types header)
struct float3_knn { // Suffixed to avoid potential conflicts if float3 is defined elsewhere differently
    float x, y, z;
};

struct MinMax_knn {
    float3_knn min_coord;
    float3_knn max_coord;
};

// --- Morton Code Helpers (from CUDA simple_knn.cu) ---

// Prepares a single coordinate for Morton encoding by interleaving bits.
// __aicore__ inline if used directly in TIK device code.
inline uint32_t prepMorton_npu(uint32_t x) {
    x = (x | (x << 16)) & 0x030000FF; // Shift and mask to spread bits
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    return x;
}

// Converts a 3D coordinate to a 30-bit Morton code (10 bits per dimension).
// Assumes coord is within [min_bound, max_bound].
// __aicore__ inline if used directly in TIK device code.
inline uint32_t coordToMorton_npu(
    const float3_knn& coord,
    const float3_knn& min_bound,
    const float3_knn& max_bound)
{
    // Normalize coordinates to [0, 1] range
    // Handle cases where max_bound == min_bound to avoid division by zero
    float range_x = max_bound.x - min_bound.x;
    float range_y = max_bound.y - min_bound.y;
    float range_z = max_bound.z - min_bound.z;

    float norm_x = (range_x == 0) ? 0.0f : (coord.x - min_bound.x) / range_x;
    float norm_y = (range_y == 0) ? 0.0f : (coord.y - min_bound.y) / range_y;
    float norm_z = (range_z == 0) ? 0.0f : (coord.z - min_bound.z) / range_z;

    // Scale to 10-bit integer range [0, 1023]
    // Clamp to ensure it's within bounds, especially due to float precision.
    uint32_t ix = static_cast<uint32_t>(std::min(1023.0f, std::max(0.0f, norm_x * 1023.0f)));
    uint32_t iy = static_cast<uint32_t>(std::min(1023.0f, std::max(0.0f, norm_y * 1023.0f)));
    uint32_t iz = static_cast<uint32_t>(std::min(1023.0f, std::max(0.0f, norm_z * 1023.0f)));

    // Interleave bits
    uint32_t morton_x = prepMorton_npu(ix);
    uint32_t morton_y = prepMorton_npu(iy);
    uint32_t morton_z = prepMorton_npu(iz);

    return morton_x | (morton_y << 1) | (morton_z << 2);
}


// --- Distance and k-NN Helpers (from CUDA simple_knn.cu) ---

// Squared Euclidean distance between two float3_knn points.
// __aicore__ inline if used directly in TIK device code.
inline float distSq_npu(const float3_knn& p1, const float3_knn& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return dx * dx + dy * dy + dz * dz;
}

// Squared distance from a point 'p' to a bounding box 'box'.
// Returns 0 if the point is inside the box.
// __aicore__ inline if used directly in TIK device code.
inline float distBoxPointSq_npu(const MinMax_knn& box, const float3_knn& p) {
    float3_knn diff = {0.0f, 0.0f, 0.0f};
    if (p.x < box.min_coord.x) diff.x = p.x - box.min_coord.x;
    else if (p.x > box.max_coord.x) diff.x = p.x - box.max_coord.x;

    if (p.y < box.min_coord.y) diff.y = p.y - box.min_coord.y;
    else if (p.y > box.max_coord.y) diff.y = p.y - box.max_coord.y;

    if (p.z < box.min_coord.z) diff.z = p.z - box.min_coord.z;
    else if (p.z > box.max_coord.z) diff.z = p.z - box.max_coord.z;

    return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

// Updates a list of K best squared distances.
// `knn_dists_sq` is an array of size K, assumed to be sorted (or at least max is easily found).
// The original CUDA code used a simple insertion sort like approach.
// __aicore__ inline if used directly in TIK device code.
template<int K_val> // K_val would be NPU_KNN_K
inline void updateKBest_npu(float current_dist_sq, float* knn_dists_sq) {
    // Assumes knn_dists_sq[K_val-1] is the largest among the current K best.
    // This is a common way to manage top-K: keep sorted or find max and replace.
    // The CUDA code iterates and shifts elements if new_dist is smaller.
    for (int j = 0; j < K_val; ++j) {
        if (knn_dists_sq[j] > current_dist_sq) {
            // Shift elements to make space and insert
            // This is what the CUDA code's loop effectively does:
            // float t = knn[j]; knn[j] = dist; dist = t;
            // This implies knn_dists_sq is not necessarily sorted, but the new dist ripples through.
            // A more robust way for small K is to keep it sorted or always find the max to replace.
            // For K=3, direct comparison and swap is fine.
            // Let's replicate the ripple/swap logic for direct translation.
            float temp = knn_dists_sq[j];
            knn_dists_sq[j] = current_dist_sq;
            current_dist_sq = temp; // The 'dist' that was replaced continues to find its spot or get discarded
        }
    }
}


#endif // NPU_SIMPLE_KNN_AUXILIARY_H_INCLUDED
