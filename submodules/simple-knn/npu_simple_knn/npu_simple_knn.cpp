// npu_simple_knn.cpp

#include "npu_simple_knn.h"
#include "npu_simple_knn_tiling.h" // For optiling::NpuSimpleKnnTilingData
#include "kernel_operator.h"     // For CALL_KERNEL or aclrtLaunchKernel
#include <acl/acl.h>             // For ACL API
#include <iostream>              // For error messages

// Define NPU_CHECK_ERROR if not already globally available
#ifndef NPU_CHECK_ERROR
#define NPU_CHECK_ERROR(status, msg) \
    if ((status) != ACL_SUCCESS) { \
        std::cerr << "NPU Error in simple_knn: " << (msg) << " | ACL code: " << (status) << std::endl; \
        throw std::runtime_error(std::string("NPU Error in simple_knn: ") + msg); \
    }
#endif

namespace NpuSimpleKNN {

// Helper to copy tiling data to a GM_ADDR (typically start of workspace)
template<typename TilingDataType>
static void copyTilingDataToNpu(
    TilingDataType& tiling_data_host,
    void* tiling_gm_target, // GM_ADDR for tiling data
    aclrtStream stream)
{
    size_t tiling_data_size = tiling_data_host.GetDataSize();
    NPU_CHECK_ERROR(aclrtMemcpy(tiling_gm_target, tiling_data_size, &tiling_data_host, tiling_data_size, ACL_MEMCPY_HOST_TO_DEVICE),
                    "Failed to copy tiling data to NPU for simple_knn.");
}

// --- Private static methods for internal stages ---

void KNN::generateSequenceNPU(
    uint32_t* out_indices_gm, int P,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores)
{
    optiling::NpuSimpleKnnTilingData tiling_data;
    tiling_data.set_P(P);
    // Populate other relevant fields for sequence generation tiling if any
    // (e.g. how P elements are split among cores)
    // KnnPointsIterTilingFunc might be adaptable if it sets up coreDataNum etc.
    // uint32_t bytes_per_element = sizeof(uint32_t); // output size per P
    // optiling::KnnPointsIterTilingFunc(nullptr, tiling_data, bytes_per_element); // Need mock context or direct setup

    // Simplified tiling for sequence generation
    tiling_data.set_ubPartDataNum(256); // Example
    uint64_t elements_per_core = (P + num_cores -1) / num_cores;
    tiling_data.set_smallCoreDataNum(elements_per_core);
    tiling_data.set_bigCoreDataNum(elements_per_core);
    tiling_data.set_smallCoreLoopNum((elements_per_core + 255)/256);
    tiling_data.set_bigCoreLoopNum((elements_per_core + 255)/256);
    uint64_t tail_data = elements_per_core % 256;
    tiling_data.set_smallCoreTailDataNum(tail_data == 0 && elements_per_core > 0 ? 256 : tail_data);
    tiling_data.set_bigCoreTailDataNum(tiling_data.smallCoreTailDataNum());
    tiling_data.set_tailBlockNum(0);


    void* tiling_data_gm_ptr = workspace_gm; // Use start of workspace for this tiling data
    copyTilingDataToNpu(tiling_data, tiling_data_gm_ptr, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_knn_thrust_sequence_equivalent", num_cores, stream,
        out_indices_gm, static_cast<uint32_t>(P),
        tiling_data_gm_ptr), "Launch npu_knn_thrust_sequence_equivalent failed.");
}


void KNN::computeBoundingBoxNPU(
    int P, const float* points_gm,
    float3_knn& min_coord_host, float3_knn& max_coord_host,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores)
{
    // This would launch reduction kernels.
    // Stage 1: Each core reduces a part of P points.
    // Stage 2: A single core reduces the partial results.
    // Workspace needs to hold partial results and tiling data for each stage.
    // For simplicity, this detailed TIK kernel launch sequence for reduction is TBD
    // and depends heavily on the custom reduction kernel design.
    // For now, placeholder:
    std::cerr << "KNN::computeBoundingBoxNPU - NPU reduction not fully implemented in interface yet." << std::endl;
    // As a fallback or initial step, this could be done on host or via a library call if available.
    // If using library reduce:
    // aclMathStatBoundingBoxFloat3(points_gm, P, &min_coord_device, &max_coord_device, stream);
    // aclrtMemcpy(&min_coord_host, ..., DEVICE_TO_HOST);
    // aclrtMemcpy(&max_coord_host, ..., DEVICE_TO_HOST);
    // For now, returning dummy values.
    min_coord_host = {0,0,0}; max_coord_host = {1,1,1};
}

void KNN::computeMortonCodesNPU(
    int P, const float* points_gm,
    const float3_knn& min_coord, const float3_knn& max_coord,
    uint32_t* morton_codes_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores)
{
    optiling::NpuSimpleKnnTilingData tiling_data;
    tiling_data.set_P(P);
    tiling_data.set_min_coord_x(min_coord.x); tiling_data.set_min_coord_y(min_coord.y); tiling_data.set_min_coord_z(min_coord.z);
    tiling_data.set_max_coord_x(max_coord.x); tiling_data.set_max_coord_y(max_coord.y); tiling_data.set_max_coord_z(max_coord.z);
    // uint32_t bytes_per_point = sizeof(float3_knn) + sizeof(uint32_t); // Input + output
    // optiling::KnnPointsIterTilingFunc(nullptr, tiling_data, bytes_per_point); // Mock context

    // Simplified tiling for morton codes
    tiling_data.set_ubPartDataNum(256); // Example
    uint64_t elements_per_core = (P + num_cores -1) / num_cores;
    tiling_data.set_smallCoreDataNum(elements_per_core); // ... (rest of simple tiling)
    // ... (fill remaining tiling fields based on P and num_cores as in generateSequenceNPU)


    void* tiling_data_gm_ptr = static_cast<char*>(workspace_gm) + 0; // Example offset
    copyTilingDataToNpu(tiling_data, tiling_data_gm_ptr, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_knn_coord_to_morton", num_cores, stream,
        const_cast<float*>(points_gm), morton_codes_gm,
        tiling_data_gm_ptr), "Launch npu_knn_coord_to_morton failed.");
}

void KNN::sortMortonCodesAndIndicesNPU(
    int P, uint32_t* morton_codes_gm, uint32_t* indices_gm, // Input arrays (indices is 0..P-1)
    uint32_t* sorted_morton_codes_gm, uint32_t* sorted_indices_gm, // Output arrays
    void* workspace_gm, aclrtStream stream, uint32_t num_cores)
{
    // This is where a call to an NPU accelerated sort library would go.
    // e.g., aclmSortPairs(morton_codes_gm, indices_gm, sorted_morton_codes_gm, sorted_indices_gm, P, stream);
    // If implementing custom sort, it would be a series of kernel launches.
    // For now, placeholder:
    std::cerr << "KNN::sortMortonCodesAndIndicesNPU - NPU sort not fully implemented in interface yet." << std::endl;
    // Fallback: copy to host, sort, copy back (very slow, for testing logic only)
    // As a HACK for now, just copy input to output to allow pipeline to proceed if sort is complex
    NPU_CHECK_ERROR(aclrtMemcpyAsync(sorted_morton_codes_gm, P * sizeof(uint32_t), morton_codes_gm, P * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "Hack copy morton failed");
    NPU_CHECK_ERROR(aclrtMemcpyAsync(sorted_indices_gm, P * sizeof(uint32_t), indices_gm, P * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "Hack copy indices failed");
    NPU_CHECK_ERROR(aclrtSynchronizeStream(stream),"Sync after hack copy failed");
}


void KNN::computeSpatialBoxesNPU(
    int P, const float* points_gm, const uint32_t* sorted_indices_gm,
    MinMax_knn* boxes_gm, uint32_t num_boxes,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores_for_boxing) // num_cores might be num_boxes
{
    optiling::NpuSimpleKnnTilingData tiling_data;
    tiling_data.set_P(P);
    tiling_data.set_num_boxes(num_boxes);
    tiling_data.set_box_size(NPU_KNN_BOX_SIZE);
    // Tiling: each core handles one box (num_cores_for_boxing should = num_boxes)
    // ubPartDataNum here refers to how many points within a box are processed per UB load by that core.
    // uint32_t bytes_per_point_in_box = sizeof(float3_knn); // Input point
    // uint32_t ub_per_box_kernel = (NPU_KNN_BOX_SIZE * bytes_per_point_in_box); // Rough estimate
    // tiling_data.set_ubPartDataNum(...); // Needs careful calculation
    // ... (set core/loop params for processing points within a box if BOX_SIZE > ubPartDataNum)
    // For boxMinMax, one core processes one box, and the box_size points are its total data.
    // So DataNum for the kernel launch is P (total points being indexed into), but blockDim is num_boxes.
    // The kernel itself uses blockIdx.x to pick its box and processes NPU_KNN_BOX_SIZE elements from sorted_indices_gm.
    // This needs a different tiling setup than PerGaussianKernelTilingFunc.
    // For now, assuming blockDim = num_boxes for the kernel launch. Tiling data might be minimal or just P.

    void* tiling_data_gm_ptr = static_cast<char*>(workspace_gm) + 2048; // Example offset
    copyTilingDataToNpu(tiling_data, tiling_data_gm_ptr, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_knn_box_min_max", num_boxes, stream, // Launch one block per box
        const_cast<float*>(points_gm), const_cast<uint32_t*>(sorted_indices_gm),
        boxes_gm, tiling_data_gm_ptr), "Launch npu_knn_box_min_max failed.");
}

void KNN::computeMeanDistancesNPU(
    int P, const float* points_gm, const uint32_t* sorted_indices_gm,
    const MinMax_knn* boxes_gm, uint32_t num_boxes,
    float* mean_dists_out_gm,
    void* workspace_gm, aclrtStream stream, uint32_t num_cores)
{
    optiling::NpuSimpleKnnTilingData tiling_data;
    tiling_data.set_P(P);
    tiling_data.set_num_boxes(num_boxes);
    tiling_data.set_box_size(NPU_KNN_BOX_SIZE);
    tiling_data.set_K(NPU_KNN_K);
    // Tiling similar to boxMinMax: blockDim is num_boxes (or P / NPU_KNN_BOX_SIZE)
    // Kernel iterates P points, each doing a search. This is the main compute kernel.
    // So, it's more like PerGaussianKernelTilingFunc for P points.
    // uint32_t bytes_per_point_knn_search = ... estimate ...;
    // KnnPointsIterTilingFunc(nullptr, tiling_data, bytes_per_point_knn_search);

    // Simplified tiling for boxMeanDist
    tiling_data.set_ubPartDataNum(1); // Process one point at a time in UB (complex search per point)
    uint64_t elements_per_core = (P + num_cores -1) / num_cores;
    tiling_data.set_smallCoreDataNum(elements_per_core); // ... (rest of simple tiling)
    // ...

    void* tiling_data_gm_ptr = static_cast<char*>(workspace_gm) + 3072; // Example offset
    copyTilingDataToNpu(tiling_data, tiling_data_gm_ptr, stream);

    NPU_CHECK_ERROR(aclrtLaunchKernel(
        "npu_knn_box_mean_dist", num_cores, stream, // Launch P threads effectively, distributed over cores
        const_cast<float*>(points_gm), const_cast<uint32_t*>(sorted_indices_gm),
        const_cast<MinMax_knn*>(boxes_gm), mean_dists_out_gm,
        tiling_data_gm_ptr), "Launch npu_knn_box_mean_dist failed.");
}


// Main public static method
void KNN::knn_npu(
    int P,
    const float* points_gm,
    float* mean_dists_out_gm,
    void* workspace_gm, // Must be large enough for all intermediates + tiling data copies
    size_t workspace_size,
    aclrtStream stream,
    uint32_t num_cores_to_use)
{
    if (P == 0) return;

    // Calculate required sizes for intermediate buffers within workspace
    size_t morton_codes_size = P * sizeof(uint32_t);
    size_t indices_size = P * sizeof(uint32_t); // For original indices 0..P-1
    size_t sorted_morton_codes_size = P * sizeof(uint32_t);
    size_t sorted_indices_size = P * sizeof(uint32_t);
    uint32_t num_boxes = (P + NPU_KNN_BOX_SIZE - 1) / NPU_KNN_BOX_SIZE;
    size_t boxes_size = num_boxes * sizeof(MinMax_knn);
    size_t min_max_reduce_buf_size = 2 * sizeof(float3_knn); // For final min/max
    // Partial reduction buffer for min/max (num_cores * sizeof(MinMax_knn)) - simplified for now

    // Simple linear allocation from workspace. Proper sub-allocation needed.
    char* current_ws_ptr = static_cast<char*>(workspace_gm);

    float3_knn* min_max_reduce_gm = reinterpret_cast<float3_knn*>(current_ws_ptr);
    current_ws_ptr += min_max_reduce_buf_size; // (sizeof(float3_knn) * 2)

    uint32_t* morton_codes_gm_ptr = reinterpret_cast<uint32_t*>(current_ws_ptr);
    current_ws_ptr += morton_codes_size;

    uint32_t* indices_gm_ptr = reinterpret_cast<uint32_t*>(current_ws_ptr);
    current_ws_ptr += indices_size;

    uint32_t* sorted_morton_codes_gm_ptr = reinterpret_cast<uint32_t*>(current_ws_ptr);
    current_ws_ptr += sorted_morton_codes_size;

    uint32_t* sorted_indices_gm_ptr = reinterpret_cast<uint32_t*>(current_ws_ptr);
    current_ws_ptr += sorted_indices_size;

    MinMax_knn* boxes_gm_ptr = reinterpret_cast<MinMax_knn*>(current_ws_ptr);
    current_ws_ptr += boxes_size;

    // Check if workspace is sufficient (basic check)
    if (static_cast<size_t>(current_ws_ptr - static_cast<char*>(workspace_gm)) > workspace_size) {
        NPU_CHECK_ERROR(ACL_ERROR_INVALID_PARAM, "Workspace size too small for simple_knn internal buffers.");
    }
    // Note: Tiling data copies also need space in workspace_gm, handled by each function for now by using base.

    // 1. Compute Bounding Box (min_coord, max_coord)
    float3_knn min_coord_host, max_coord_host; // Will be computed and used on device or copied to host then back.
                                        // For now, assume computeBoundingBoxNPU fills these if it were fully NPU.
                                        // The CUDA code copies to host, then passes to Morton kernel.
                                        // Let's assume they are computed and kept on device, then passed to morton.
                                        // This means `min_max_reduce_gm` holds the final result.
    computeBoundingBoxNPU(P, points_gm, min_coord_host, max_coord_host, workspace_gm, stream, num_cores_to_use);
    // For NPU version, min_coord_host/max_coord_host would be GM_ADDRs if reduction is fully on device
    // For now, using host versions as placeholders for values needed by Morton kernel's tiling data.
    // A proper NPU reduction kernel would output to min_max_reduce_gm.

    // 2. Compute Morton Codes
    computeMortonCodesNPU(P, points_gm, min_coord_host, max_coord_host, morton_codes_gm_ptr, workspace_gm, stream, num_cores_to_use);

    // 3. Generate initial indices (0 to P-1)
    generateSequenceNPU(indices_gm_ptr, P, workspace_gm, stream, num_cores_to_use);

    // 4. Sort Morton codes and indices
    // This is the major CUB dependency.
    sortMortonCodesAndIndicesNPU(P, morton_codes_gm_ptr, indices_gm_ptr,
                                 sorted_morton_codes_gm_ptr, sorted_indices_gm_ptr,
                                 workspace_gm, stream, num_cores_to_use);

    // 5. Compute Min/Max for each box of sorted points
    computeSpatialBoxesNPU(P, points_gm, sorted_indices_gm_ptr, boxes_gm_ptr, num_boxes, workspace_gm, stream, num_boxes); // One core per box

    // 6. Compute Mean Distances
    computeMeanDistancesNPU(P, points_gm, sorted_indices_gm_ptr, boxes_gm_ptr, num_boxes,
                            mean_dists_out_gm, workspace_gm, stream, num_cores_to_use);

    // Ensure all kernels are finished if caller expects synchronous behavior or will use results immediately
    // NPU_CHECK_ERROR(aclrtSynchronizeStream(stream), "Stream sync failed after simple_knn_npu.");
}


size_t KNN::getWorkspaceSizeBytes(int P, int num_boxes_ignored) {
    // Estimate based on intermediate buffers needed.
    // This needs to be accurate if caller pre-allocates.
    size_t size = 0;
    size += P * sizeof(uint32_t); // morton_codes
    size += P * sizeof(uint32_t); // indices
    size += P * sizeof(uint32_t); // sorted_morton_codes
    size += P * sizeof(uint32_t); // sorted_indices
    uint32_t num_boxes_calc = (P + NPU_KNN_BOX_SIZE - 1) / NPU_KNN_BOX_SIZE;
    size += num_boxes_calc * sizeof(MinMax_knn); // boxes
    size += 2 * sizeof(float3_knn); // global min/max from reduction
    // Add space for any partial reduction buffers if needed
    // Add space for multiple tiling data structs if they are copied to workspace sequentially
    size += 4096; // Generous buffer for all tiling data copies and small alignment gaps
    return size;
}


} // namespace NpuSimpleKNN
