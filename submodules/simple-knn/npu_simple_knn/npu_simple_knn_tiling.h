// npu_simple_knn_tiling.h

#ifndef NPU_SIMPLE_KNN_TILING_H_INCLUDED
#define NPU_SIMPLE_KNN_TILING_H_INCLUDED

#include "register/tilingdata_base.h" // From Ascend C/TBE includes

namespace optiling {

// TilingData structure for NPU Simple k-NN kernels
BEGIN_TILING_DATA_DEF(NpuSimpleKnnTilingData)
    // Global parameters
    TILING_DATA_FIELD_DEF(uint32_t, P); // Total number of points
    TILING_DATA_FIELD_DEF(uint32_t, K); // Number of neighbors (e.g., NPU_KNN_K)

    // Parameters for Morton code calculation (if min/max are passed via tiling)
    // Alternatively, min/max can be computed by a preceding kernel.
    // float3 min_coord_x, min_coord_y, min_coord_z; (Can't use float3 directly here)
    // float3 max_coord_x, max_coord_y, max_coord_z;
    TILING_DATA_FIELD_DEF(float, min_coord_x);
    TILING_DATA_FIELD_DEF(float, min_coord_y);
    TILING_DATA_FIELD_DEF(float, min_coord_z);
    TILING_DATA_FIELD_DEF(float, max_coord_x);
    TILING_DATA_FIELD_DEF(float, max_coord_y);
    TILING_DATA_FIELD_DEF(float, max_coord_z);

    // Core distribution and loop parameters (similar to add_custom or rasterizer)
    // These will define how 'P' points are split among cores for various stages.
    TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum); // Points per small core
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);   // Points per big core
    TILING_DATA_FIELD_DEF(uint64_t, ubPartDataNum);    // Points processed per UB load in a loop
    TILING_DATA_FIELD_DEF(uint64_t, smallCoreTailDataNum);
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreTailDataNum);
    TILING_DATA_FIELD_DEF(uint64_t, smallCoreLoopNum);
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreLoopNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum); // Number of "big" cores

    // Parameters for specific kernels, e.g., number of boxes for boxMinMax and boxMeanDist
    TILING_DATA_FIELD_DEF(uint32_t, num_boxes);
    TILING_DATA_FIELD_DEF(uint32_t, box_size); // NPU_KNN_BOX_SIZE

    // Add more fields if different kernels have vastly different tiling needs
    // or if certain stages (like custom sort/reduce) need their own parameters.

END_TILING_DATA_DEF;

// Register TilingData for each kernel type that will have an OpDef
// For simple_knn, we might have ops for:
// 1. Full k-NN (if it's one monolithic Op)
// 2. Or separate Ops for stages: BoundingBox, MortonCoding, Sort, Boxing, KnnSearch
// Assuming a single Op for the whole k-NN process for now for simplicity,
// though this Op's TilingFunc would be complex or it would call sub-kernels.
// Let's name it based on the main function.
REGISTER_TILING_DATA_CLASS(NpuSimpleKnnOp, NpuSimpleKnnTilingData)

// If we break it down:
// REGISTER_TILING_DATA_CLASS(NpuMinMaxReduceOp, NpuSimpleKnnTilingData)
// REGISTER_TILING_DATA_CLASS(NpuCoordToMortonOp, NpuSimpleKnnTilingData)
// REGISTER_TILING_DATA_CLASS(NpuSortPairsOp, NpuSimpleKnnTilingData) // If custom sort
// REGISTER_TILING_DATA_CLASS(NpuBoxMinMaxOp, NpuSimpleKnnTilingData)
// REGISTER_TILING_DATA_CLASS(NpuBoxMeanDistOp, NpuSimpleKnnTilingData)


} // namespace optiling

#endif // NPU_SIMPLE_KNN_TILING_H_INCLUDED
