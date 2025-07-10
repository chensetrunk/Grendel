// npu_simple_knn.h

#ifndef NPU_SIMPLE_KNN_H_INCLUDED
#define NPU_SIMPLE_KNN_H_INCLUDED

#include "acl/acl_rt.h"       // For aclrtStream, GM_ADDR (typically void*)
#include "npu_simple_knn_auxiliary.h" // For float3_knn etc.

// Forward declare TIK kernel launchers (names must match extern "C" in .cpp)
// These are placeholders; actual kernels will be more specific.
// Example: if bounding box is one kernel, morton another, etc.
extern "C" void npu_knn_min_max_reduce_stage1(GM_ADDR points_gm, GM_ADDR partial_min_max_gm, GM_ADDR tiling_gm, aclrtStream stream);
extern "C" void npu_knn_min_max_reduce_stage2(GM_ADDR partial_min_max_gm, GM_ADDR final_min_max_gm, GM_ADDR tiling_gm, aclrtStream stream);
extern "C" void npu_knn_coord_to_morton(GM_ADDR points_gm, GM_ADDR morton_codes_gm, GM_ADDR tiling_gm, aclrtStream stream);
// extern "C" void npu_knn_radix_sort_pairs(GM_ADDR keys_gm, GM_ADDR values_gm, uint32_t P, GM_ADDR tiling_gm, aclrtStream stream); // If custom sort
extern "C" void npu_knn_box_min_max(GM_ADDR points_gm, GM_ADDR sorted_indices_gm, GM_ADDR boxes_gm, GM_ADDR tiling_gm, aclrtStream stream);
extern "C" void npu_knn_box_mean_dist(GM_ADDR points_gm, GM_ADDR sorted_indices_gm, GM_ADDR boxes_gm, GM_ADDR mean_dists_out_gm, GM_ADDR tiling_gm, aclrtStream stream);
extern "C" void npu_knn_thrust_sequence_equivalent(GM_ADDR out_indices_gm, uint32_t P, GM_ADDR tiling_gm, aclrtStream stream);


namespace NpuSimpleKNN {

class KNN {
public:
    // Main k-NN function (K is fixed to NPU_KNN_K from config for now)
    // Takes NPU global memory pointers for inputs and outputs.
    // Intermediate buffers (morton codes, sorted indices, boxes) are allocated
    // and managed internally or passed via a workspace pointer.
    static void knn_npu(
        int P,                          // Number of points
        const float* points_gm,         // Input points (Px3) on NPU GM
        float* mean_dists_out_gm,     // Output mean distances (P) on NPU GM
        void* workspace_gm,             // Pre-allocated NPU workspace for intermediate data & tiling
        size_t workspace_size,          // Size of the workspace
        aclrtStream stream,
        uint32_t num_cores_for_stages // General indication of cores, tiling will refine
    );

    // Helper to calculate required workspace size
    static size_t getWorkspaceSizeBytes(int P, int num_boxes);

private:
    // Internal stages can be private static methods if this class manages the whole flow.
    // Or, they could be free functions in an anonymous namespace in the .cpp.
    static void computeBoundingBoxNPU(
        int P, const float* points_gm,
        float3_knn& min_coord, float3_knn& max_coord,
        void* workspace_gm, aclrtStream stream, uint32_t num_cores);

    static void computeMortonCodesNPU(
        int P, const float* points_gm,
        const float3_knn& min_coord, const float3_knn& max_coord,
        uint32_t* morton_codes_gm,
        void* workspace_gm, aclrtStream stream, uint32_t num_cores);

    // Sorting will be a major part. If using a library, this call wraps it.
    // If custom, it launches the custom sort kernels.
    static void sortMortonCodesAndIndicesNPU(
        int P, uint32_t* morton_codes_gm, uint32_t* indices_gm,
        uint32_t* sorted_morton_codes_gm, uint32_t* sorted_indices_gm,
        void* workspace_gm, aclrtStream stream, uint32_t num_cores);

    static void computeSpatialBoxesNPU(
        int P, const float* points_gm, const uint32_t* sorted_indices_gm,
        MinMax_knn* boxes_gm, uint32_t num_boxes,
        void* workspace_gm, aclrtStream stream, uint32_t num_cores);

    static void computeMeanDistancesNPU(
        int P, const float* points_gm, const uint32_t* sorted_indices_gm,
        const MinMax_knn* boxes_gm, uint32_t num_boxes,
        float* mean_dists_out_gm,
        void* workspace_gm, aclrtStream stream, uint32_t num_cores);

    static void generateSequenceNPU(
        uint32_t* out_indices_gm, int P,
        void* workspace_gm, aclrtStream stream, uint32_t num_cores);

};

} // namespace NpuSimpleKNN

#endif // NPU_SIMPLE_KNN_H_INCLUDED
