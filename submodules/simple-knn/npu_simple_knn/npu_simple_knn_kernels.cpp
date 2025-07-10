// npu_simple_knn_kernels.cpp

#include "kernel_operator.h"
#include "npu_simple_knn_tiling.h"
#include "npu_simple_knn_auxiliary.h" // For float3_knn, MinMax_knn, helpers
#include <algorithm> // for std::min, std::max

// Define buffer numbers for pipe if used (may not be needed for all simple_knn kernels)
constexpr int32_t KNN_BUFFER_NUM = 2;


// --- Kernel for generating sequence (0 to P-1) ---
template<bool IsExistBigCore>
class KernelGenerateSequence {
public:
    __aicore__ inline KernelGenerateSequence() {}
    __aicore__ inline void Init(GM_ADDR out_indices_gm, GM_ADDR tiling_gm) {
        GET_TILING_DATA(tiling, tiling_gm, optiling::NpuSimpleKnnTilingData);
        this->tilingData = tiling;
        // Core/loop setup from tilingData (smallCoreDataNum, etc.)
        uint64_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalBufferOffset = 0;
        if constexpr (IsExistBigCore) { /* ... (standard core/loop setup) ... */
            if (coreNum < tiling.tailBlockNum) {
                this->numElementsPerCore = tiling.bigCoreDataNum; this->numLoops = tiling.bigCoreLoopNum;
                this->tailDataNumInLoop = tiling.bigCoreTailDataNum; globalBufferOffset = tiling.bigCoreDataNum * coreNum;
            } else {
                this->numElementsPerCore = tiling.smallCoreDataNum; this->numLoops = tiling.smallCoreLoopNum;
                this->tailDataNumInLoop = tiling.smallCoreTailDataNum; globalBufferOffset = tiling.bigCoreDataNum * tiling.tailBlockNum + tiling.smallCoreDataNum * (coreNum - tiling.tailBlockNum);
            }
        } else { /* ... (standard core/loop setup) ... */
            this->numElementsPerCore = tiling.smallCoreDataNum; this->numLoops = tiling.smallCoreLoopNum;
            this->tailDataNumInLoop = tiling.smallCoreTailDataNum; globalBufferOffset = tiling.smallCoreDataNum * coreNum;
        }
        this->ubPartDataNum = tiling.ubPartDataNum;
        this->startValueOffset = globalBufferOffset; // Each core starts its sequence from its global offset

        output_indices_gm.SetGlobalBuffer((__gm uint32_t*)out_indices_gm + this->startValueOffset, this->numElementsPerCore);
        pipe.InitBuffer(outQueueIndices, KNN_BUFFER_NUM, this->ubPartDataNum * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        this->currentLoopDataNum = this->ubPartDataNum;
        for (uint32_t i = 0; i < this->numLoops; ++i) {
            if (i == this->numLoops - 1) {
                this->currentLoopDataNum = this->tailDataNumInLoop;
            }
            if (this->currentLoopDataNum == 0) continue;

            AscendC::LocalTensor<uint32_t> indices_local = outQueueIndices.AllocTensor<uint32_t>();
            uint32_t loop_offset = i * this->ubPartDataNum;
            for (uint32_t j = 0; j < this->currentLoopDataNum; ++j) {
                indices_local.SetScalar(j, this->startValueOffset + loop_offset + j);
            }
            outQueueIndices.EnQue(indices_local);

            AscendC::LocalTensor<uint32_t> indices_to_copy = outQueueIndices.DeQue<uint32_t>();
            AscendC::DataCopy(output_indices_gm[loop_offset], indices_to_copy, this->currentLoopDataNum);
            outQueueIndices.FreeTensor(indices_to_copy);
        }
    }
private:
    AscendC::GlobalTensor<uint32_t> output_indices_gm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, KNN_BUFFER_NUM> outQueueIndices;
    optiling::NpuSimpleKnnTilingData tilingData;
    uint64_t numElementsPerCore, numLoops, ubPartDataNum, tailDataNumInLoop, currentLoopDataNum;
    uint64_t startValueOffset;
};

extern "C" __global__ __aicore__ void npu_knn_thrust_sequence_equivalent(
    GM_ADDR out_indices_gm, uint32_t P_total_unused, /* P comes from tiling */
    GM_ADDR tiling_gm, aclrtStream stream_unused) {
    if (TILING_KEY_IS(1)) { KernelGenerateSequence<true> op; op.Init(out_indices_gm, tiling_gm); op.Process(); }
    else { KernelGenerateSequence<false> op; op.Init(out_indices_gm, tiling_gm); op.Process(); }
}


// --- Kernel for Morton Coding ---
template<bool IsExistBigCore>
class KernelCoordToMorton {
public:
    __aicore__ inline KernelCoordToMorton() {}
    __aicore__ inline void Init(GM_ADDR points_gm_in, GM_ADDR morton_codes_gm_out, GM_ADDR tiling_gm) {
        GET_TILING_DATA(tiling, tiling_gm, optiling::NpuSimpleKnnTilingData);
        this->tilingData = tiling;
        // Core/loop setup from tilingData
        uint64_t coreNum = AscendC::GetBlockIdx(); uint64_t globalBufferOffset = 0;
        // ... (standard core/loop setup for P points, similar to KernelGenerateSequence) ...
        if constexpr (IsExistBigCore) { /* ... */
            if (coreNum < tiling.tailBlockNum) {
                this->numElementsPerCore = tiling.bigCoreDataNum; this->numLoops = tiling.bigCoreLoopNum;
                this->tailDataNumInLoop = tiling.bigCoreTailDataNum; globalBufferOffset = tiling.bigCoreDataNum * coreNum;
            } else {
                this->numElementsPerCore = tiling.smallCoreDataNum; this->numLoops = tiling.smallCoreLoopNum;
                this->tailDataNumInLoop = tiling.smallCoreTailDataNum; globalBufferOffset = tiling.bigCoreDataNum * tiling.tailBlockNum + tiling.smallCoreDataNum * (coreNum - tiling.tailBlockNum);
            }
        } else { /* ... */
            this->numElementsPerCore = tiling.smallCoreDataNum; this->numLoops = tiling.smallCoreLoopNum;
            this->tailDataNumInLoop = tiling.smallCoreTailDataNum; globalBufferOffset = tiling.smallCoreDataNum * coreNum;
        }
        this->ubPartDataNum = tiling.ubPartDataNum;

        points_gm.SetGlobalBuffer((__gm float*)points_gm_in + globalBufferOffset * 3, this->numElementsPerCore * 3);
        morton_codes_gm.SetGlobalBuffer((__gm uint32_t*)morton_codes_gm_out + globalBufferOffset, this->numElementsPerCore);

        pipe.InitBuffer(inQueuePoints, KNN_BUFFER_NUM, this->ubPartDataNum * 3 * sizeof(float));
        pipe.InitBuffer(outQueueMorton, KNN_BUFFER_NUM, this->ubPartDataNum * sizeof(uint32_t));

        min_bound = {tiling.min_coord_x(), tiling.min_coord_y(), tiling.min_coord_z()};
        max_bound = {tiling.max_coord_x(), tiling.max_coord_y(), tiling.max_coord_z()};
    }

    __aicore__ inline void Process() {
        this->currentLoopDataNum = this->ubPartDataNum;
        for (uint32_t i = 0; i < this->numLoops; ++i) {
            if (i == this->numLoops - 1) this->currentLoopDataNum = this->tailDataNumInLoop;
            if (this->currentLoopDataNum == 0) continue;

            // CopyIn
            AscendC::LocalTensor<float> points_local = inQueuePoints.AllocTensor<float>();
            AscendC::DataCopy(points_local, points_gm[i * this->ubPartDataNum * 3], this->currentLoopDataNum * 3);
            inQueuePoints.EnQue(points_local);

            // Compute
            AscendC::LocalTensor<float> current_points = inQueuePoints.DeQue<float>();
            AscendC::LocalTensor<uint32_t> morton_local = outQueueMorton.AllocTensor<uint32_t>();
            for (uint32_t j = 0; j < this->currentLoopDataNum; ++j) {
                float3_knn p = {current_points.GetScalar(j*3+0), current_points.GetScalar(j*3+1), current_points.GetScalar(j*3+2)};
                morton_local.SetScalar(j, coordToMorton_npu(p, min_bound, max_bound));
            }
            outQueueMorton.EnQue(morton_local);
            inQueuePoints.FreeTensor(current_points);

            // CopyOut
            AscendC::LocalTensor<uint32_t> morton_to_copy = outQueueMorton.DeQue<uint32_t>();
            AscendC::DataCopy(morton_codes_gm[i * this->ubPartDataNum], morton_to_copy, this->currentLoopDataNum);
            outQueueMorton.FreeTensor(morton_to_copy);
        }
    }
private:
    AscendC::GlobalTensor<float> points_gm;
    AscendC::GlobalTensor<uint32_t> morton_codes_gm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, KNN_BUFFER_NUM> inQueuePoints;
    AscendC::TQue<AscendC::QuePosition::VECOUT, KNN_BUFFER_NUM> outQueueMorton;
    optiling::NpuSimpleKnnTilingData tilingData;
    uint64_t numElementsPerCore, numLoops, ubPartDataNum, tailDataNumInLoop, currentLoopDataNum;
    float3_knn min_bound, max_bound;
};

extern "C" __global__ __aicore__ void npu_knn_coord_to_morton(
    GM_ADDR points_gm, GM_ADDR morton_codes_gm,
    GM_ADDR tiling_gm, aclrtStream stream_unused) {
    if (TILING_KEY_IS(1)) { KernelCoordToMorton<true> op; op.Init(points_gm, morton_codes_gm, tiling_gm); op.Process(); }
    else { KernelCoordToMorton<false> op; op.Init(points_gm, morton_codes_gm, tiling_gm); op.Process(); }
}


// --- Kernel for BoxMinMax ---
// This kernel is launched with blockDim = num_boxes. Each block processes one box.
// Tiling data mainly provides P and box_size.
template<bool IsExistBigCore_Unused> // This kernel's structure doesn't use big/small core logic for elements
class KernelBoxMinMax {
public:
    __aicore__ inline KernelBoxMinMax() {}
    __aicore__ inline void Init(GM_ADDR points_gm_in, GM_ADDR sorted_indices_gm_in, GM_ADDR boxes_gm_out, GM_ADDR tiling_gm) {
        GET_TILING_DATA(tiling, tiling_gm, optiling::NpuSimpleKnnTilingData);
        this->tilingData = tiling;

        uint32_t box_idx = AscendC::GetBlockIdx(); // Each block is one box

        // Points to all points and all sorted_indices, kernel will select based on box_idx
        points_gm.SetGlobalBuffer((__gm float*)points_gm_in);
        sorted_indices_gm.SetGlobalBuffer((__gm uint32_t*)sorted_indices_gm_in);
        boxes_gm.SetGlobalBuffer((__gm MinMax_knn*)boxes_gm_out + box_idx, 1); // Each block writes one MinMax_knn

        // UB for points within a box. Max size is NPU_KNN_BOX_SIZE points.
        // If NPU_KNN_BOX_SIZE * sizeof(float3_knn) > UB, this needs batching.
        // Assuming it fits for now, as in CUDA shared memory.
        // Max points to load: tiling.box_size()
        // If box_size is large, this needs internal looping & UB management.
        // For now, assume box_size points fit in one UB load.
        uint32_t points_in_this_box = tiling.box_size();
        if (box_idx == tiling.num_boxes() - 1) { // Last box might be smaller
            uint32_t remainder = tiling.P() % tiling.box_size();
            if (remainder != 0) points_in_this_box = remainder;
        }
        this->num_points_this_box = points_in_this_box;
        this->box_start_offset_in_sorted_indices = box_idx * tiling.box_size();

        // Allocate LocalTensor for points of the current box
        // Size: num_points_this_box * 3 floats
        // This is a simplification. The CUDA kernel used shared memory of fixed BOX_SIZE.
        // TIK local memory is UB.
        // If box_size is too large for UB, this kernel needs internal loops.
        // For now, assume it fits.
        current_box_points_local.AllocTensor<float>({this->num_points_this_box * 3});
    }

    __aicore__ inline void Process() {
        if (this->num_points_this_box == 0) return;

        // Simplified CopyIn: Load all points for this box into UB
        // This is a gather operation based on sorted_indices_gm
        for(uint32_t i=0; i < this->num_points_this_box; ++i) {
            uint32_t point_original_idx = sorted_indices_gm.GetScalar(this->box_start_offset_in_sorted_indices + i);
            current_box_points_local.SetScalar(i*3+0, points_gm.GetScalar(point_original_idx*3+0));
            current_box_points_local.SetScalar(i*3+1, points_gm.GetScalar(point_original_idx*3+1));
            current_box_points_local.SetScalar(i*3+2, points_gm.GetScalar(point_original_idx*3+2));
        }

        // Compute MinMax within the UB data (current_box_points_local)
        MinMax_knn current_box_min_max;
        if (this->num_points_this_box > 0) {
            current_box_min_max.min_coord = {current_box_points_local.GetScalar(0), current_box_points_local.GetScalar(1), current_box_points_local.GetScalar(2)};
            current_box_min_max.max_coord = current_box_min_max.min_coord;
            for (uint32_t i = 1; i < this->num_points_this_box; ++i) {
                float px = current_box_points_local.GetScalar(i*3+0);
                float py = current_box_points_local.GetScalar(i*3+1);
                float pz = current_box_points_local.GetScalar(i*3+2);
                current_box_min_max.min_coord.x = std::min(current_box_min_max.min_coord.x, px);
                current_box_min_max.min_coord.y = std::min(current_box_min_max.min_coord.y, py);
                current_box_min_max.min_coord.z = std::min(current_box_min_max.min_coord.z, pz);
                current_box_min_max.max_coord.x = std::max(current_box_min_max.max_coord.x, px);
                current_box_min_max.max_coord.y = std::max(current_box_min_max.max_coord.y, py);
                current_box_min_max.max_coord.z = std::max(current_box_min_max.max_coord.z, pz);
            }
        } else { // Should not happen if num_points_this_box > 0 check passed
            current_box_min_max.min_coord = {3.402823466e+38F, 3.402823466e+38F, 3.402823466e+38F}; // FLT_MAX
            current_box_min_max.max_coord = {-3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F}; // -FLT_MAX
        }

        // CopyOut result to GM (this kernel writes one MinMax_knn struct)
        // This direct GM write for a struct needs careful handling or writing element by element.
        // Using a LocalTensor for the output struct.
        AscendC::LocalTensor<MinMax_knn> box_result_local;
        box_result_local.AllocTensor<MinMax_knn>({1});
        box_result_local.AsVec<MinMax_knn>()[0] = current_box_min_max; // If AssignPointer was used
        // Or element-wise:
        // box_result_local.SetScalar(0, current_box_min_max.min_coord.x); ... etc. (tedious for structs)

        // Assuming GlobalTensor boxes_gm points to the correct single MinMax_knn slot for this box_idx
        AscendC::DataCopy(boxes_gm[0], box_result_local, 1); // Copy 1 MinMax_knn struct
        box_result_local.FreeTensor();
        current_box_points_local.FreeTensor();
    }
private:
    AscendC::GlobalTensor<float> points_gm;
    AscendC::GlobalTensor<uint32_t> sorted_indices_gm;
    AscendC::GlobalTensor<MinMax_knn> boxes_gm; // Points to a single MinMax_knn slot
    AscendC::LocalTensor<float> current_box_points_local;
    optiling::NpuSimpleKnnTilingData tilingData;
    uint32_t num_points_this_box;
    uint32_t box_start_offset_in_sorted_indices;
};

extern "C" __global__ __aicore__ void npu_knn_box_min_max(
    GM_ADDR points_gm, GM_ADDR sorted_indices_gm, GM_ADDR boxes_gm,
    GM_ADDR tiling_gm, aclrtStream stream_unused) {
    // This kernel is launched with num_boxes blocks. Tiling key might not be used.
    KernelBoxMinMax<false> op;
    op.Init(points_gm, sorted_indices_gm, boxes_gm, tiling_gm);
    op.Process();
}


// --- Kernel for BoxMeanDist (Main k-NN search) ---
// Launched with P points distributed across cores.
template<bool IsExistBigCore>
class KernelBoxMeanDist {
public:
    __aicore__ inline KernelBoxMeanDist() {}
    __aicore__ inline void Init(GM_ADDR points_gm_in, GM_ADDR sorted_indices_gm_in,
                                GM_ADDR boxes_gm_in, GM_ADDR mean_dists_out_gm_in,
                                GM_ADDR tiling_gm) {
        GET_TILING_DATA(tiling, tiling_gm, optiling::NpuSimpleKnnTilingData);
        this->tilingData = tiling;
        // Core/loop setup for P points
        uint64_t coreNum = AscendC::GetBlockIdx(); uint64_t globalBufferOffset = 0;
        // ... (standard core/loop setup for P points, similar to KernelGenerateSequence) ...
         if constexpr (IsExistBigCore) { /* ... */
            if (coreNum < tiling.tailBlockNum) {
                this->numElementsPerCore = tiling.bigCoreDataNum; this->numLoops = tiling.bigCoreLoopNum;
                this->tailDataNumInLoop = tiling.bigCoreTailDataNum; globalBufferOffset = tiling.bigCoreDataNum * coreNum;
            } else {
                this->numElementsPerCore = tiling.smallCoreDataNum; this->numLoops = tiling.smallCoreLoopNum;
                this->tailDataNumInLoop = tiling.smallCoreTailDataNum; globalBufferOffset = tiling.bigCoreDataNum * tiling.tailBlockNum + tiling.smallCoreDataNum * (coreNum - tiling.tailBlockNum);
            }
        } else { /* ... */
            this->numElementsPerCore = tiling.smallCoreDataNum; this->numLoops = tiling.smallCoreLoopNum;
            this->tailDataNumInLoop = tiling.smallCoreTailDataNum; globalBufferOffset = tiling.smallCoreDataNum * coreNum;
        }
        this->ubPartDataNum = tiling.ubPartDataNum; // Should be 1 for this kernel (one point processed at a time)
        this->global_point_idx_offset = globalBufferOffset;


        // Global Tensors
        points_gm.SetGlobalBuffer((__gm float*)points_gm_in); // Access all points
        sorted_indices_gm.SetGlobalBuffer((__gm uint32_t*)sorted_indices_gm_in); // Access all sorted indices
        boxes_gm.SetGlobalBuffer((__gm MinMax_knn*)boxes_gm_in); // Access all boxes
        mean_dists_out_gm.SetGlobalBuffer((__gm float*)mean_dists_out_gm_in + globalBufferOffset, this->numElementsPerCore);

        // UB (LocalTensors)
        // We process one query point at a time. Its data and k-best distances are in UB.
        // Boxes and points from boxes are loaded iteratively.
        // If NPU_KNN_BOX_SIZE is large, points_from_one_box_local needs to be batched.
        // Assuming NPU_KNN_BOX_SIZE points fit in UB for now.
        pipe.InitBuffer(inQueueQueryPointData, KNN_BUFFER_NUM, 1 * 3 * sizeof(float)); // float3 for current query point
        pipe.InitBuffer(outQueueMeanDist, KNN_BUFFER_NUM, 1 * sizeof(float)); // Output for current query point

        // Local storage for K best distances
        knn_dists_sq_local.AllocTensor<float>({(uint32_t)NPU_KNN_K});
        // Local storage for one box's data (MinMax)
        one_box_data_local.AllocTensor<MinMax_knn>({1});
        // Local storage for points within one box (up to NPU_KNN_BOX_SIZE)
        points_from_one_box_local.AllocTensor<float>({(uint32_t)NPU_KNN_BOX_SIZE * 3});
    }

    __aicore__ inline void Process() {
        this->currentLoopDataNum = this->ubPartDataNum; // Should be 1
        for (uint32_t i = 0; i < this->numLoops; ++i) { // Loop over points assigned to this core
            if (i == this->numLoops - 1) this->currentLoopDataNum = this->tailDataNumInLoop;
            if (this->currentLoopDataNum == 0) continue;

            // Each iteration of this outer loop processes ONE query point
            uint32_t current_sorted_idx_in_P = this->global_point_idx_offset + i; // This is the index in the *sorted* list
            uint32_t query_point_original_idx = sorted_indices_gm.GetScalar(current_sorted_idx_in_P);

            float3_knn query_point_val; // Load the query point
            query_point_val.x = points_gm.GetScalar(query_point_original_idx * 3 + 0);
            query_point_val.y = points_gm.GetScalar(query_point_original_idx * 3 + 1);
            query_point_val.z = points_gm.GetScalar(query_point_original_idx * 3 + 2);

            // Initialize K-best distances for this query point
            for(int k=0; k<NPU_KNN_K; ++k) knn_dists_sq_local.SetScalar(k, 3.402823466e+38F); // FLT_MAX

            // Initial local search (small window around current_sorted_idx_in_P)
            for (int local_search_offset = -NPU_KNN_K; local_search_offset <= NPU_KNN_K; ++local_search_offset) {
                int neighbor_sorted_idx = current_sorted_idx_in_P + local_search_offset;
                if (neighbor_sorted_idx == current_sorted_idx_in_P || neighbor_sorted_idx < 0 || neighbor_sorted_idx >= (int)tilingData.P()) continue;

                uint32_t neighbor_original_idx = sorted_indices_gm.GetScalar(neighbor_sorted_idx);
                float3_knn neighbor_point = {points_gm.GetScalar(neighbor_original_idx*3+0), points_gm.GetScalar(neighbor_original_idx*3+1), points_gm.GetScalar(neighbor_original_idx*3+2)};
                updateKBest_npu<NPU_KNN_K>(query_point_val, neighbor_point, knn_dists_sq_local.GetBuffer());
            }

            float reject_dist_sq = knn_dists_sq_local.GetScalar(NPU_KNN_K - 1); // Current K'th best distance

            // Iterate through all spatial boxes
            for (uint32_t box_j = 0; box_j < tilingData.num_boxes(); ++box_j) {
                // Load current box's MinMax data
                // AscendC::DataCopy(one_box_data_local, boxes_gm[box_j], 1); // If GlobalTensor was setup for one box
                // Or scalar reads if boxes_gm is full array:
                one_box_data_local.AsVec<MinMax_knn>()[0].min_coord.x = boxes_gm.GetScalar(box_j * sizeof(MinMax_knn) / sizeof(float) + 0); // This indexing is tricky for structs
                // Safer: Load struct elements one by one or use proper GlobalTensor indexing if it supports struct members
                // This part requires careful handling of struct layout in GM.
                // For now, assume one_box_data_local is filled correctly by some means for box_j.
                // Load current box's MinMax data into one_box_data_local
                // Assuming boxes_gm is GlobalTensor<MinMax_knn> and points to the array of boxes
                AscendC::DataCopy(one_box_data_local, boxes_gm[box_j], 1); // Copy one MinMax_knn struct
                MinMax_knn current_eval_box = one_box_data_local.AsVec<MinMax_knn>()[0];

                float dist_to_box_sq = distBoxPointSq_npu(current_eval_box, query_point_val);
                if (dist_to_box_sq > reject_dist_sq || dist_to_box_sq > knn_dists_sq_local.GetScalar(NPU_KNN_K-1) ) {
                    continue;
                }

                // If box is promising, iterate points within this box
                uint32_t box_points_start = box_j * tilingData.box_size();
                uint32_t points_in_this_eval_box = tilingData.box_size();
                if (box_j == tilingData.num_boxes() - 1) { // Last box might be smaller
                    uint32_t remainder = tilingData.P() % tilingData.box_size();
                    if (remainder != 0) points_in_this_eval_box = remainder;
                }

                // Iterate points within the current promising box, fetching directly from GM
                for(uint32_t pt_k_idx_in_box = 0; pt_k_idx_in_box < points_in_this_eval_box; ++pt_k_idx_in_box) {
                    uint32_t point_in_box_sorted_idx = box_points_start + pt_k_idx_in_box;
                    uint32_t point_in_box_original_idx = sorted_indices_gm.GetScalar(point_in_box_sorted_idx);

                    if (point_in_box_original_idx == query_point_original_idx) continue;

                    float3_knn p_in_box = {
                        points_gm.GetScalar(point_in_box_original_idx * 3 + 0),
                        points_gm.GetScalar(point_in_box_original_idx * 3 + 1),
                        points_gm.GetScalar(point_in_box_original_idx * 3 + 2)
                    };
                    updateKBest_npu<NPU_KNN_K>(query_point_val, p_in_box, knn_dists_sq_local.GetBuffer());
                }
                reject_dist_sq = knn_dists_sq_local.GetScalar(NPU_KNN_K - 1); // Update reject distance
            }

            // Calculate mean of K best distances
            float sum_knn_dists_sq = 0; // Distances are squared
            for(int k=0; k<NPU_KNN_K; ++k) {
                float d_sq = knn_dists_sq_local.GetScalar(k);
                if (d_sq < 3.3e+38F) { // Check if it's not FLT_MAX (i.e., a valid distance found)
                     sum_knn_dists_sq += d_sq; // Original code sums squared distances then divides
                }
            }
            // The original CUDA code returns mean of squared distances.
            // If actual mean distance is needed, sqrt should be applied before summing or after averaging.
            // Following original: mean of squared distances.
            float mean_dist_val = (NPU_KNN_K > 0) ? (sum_knn_dists_sq / NPU_KNN_K) : 0.0f;

            // Store result
            mean_dists_out_gm.SetScalar(i, mean_dist_val);
        }
    }

private:
    AscendC::GlobalTensor<float> points_gm;
    AscendC::GlobalTensor<uint32_t> sorted_indices_gm;
    AscendC::GlobalTensor<MinMax_knn> boxes_gm;
    AscendC::GlobalTensor<float> mean_dists_out_gm;

    AscendC::TPipe pipe;
    // Queues might not be strictly necessary if ubPartDataNum is 1 and data is small for one point.
    // For now, kept for consistency, but query_point_val could be direct local vars.
    AscendC::TQue<AscendC::QuePosition::VECIN, KNN_BUFFER_NUM> inQueueQueryPointData;
    AscendC::TQue<AscendC::QuePosition::VECOUT, KNN_BUFFER_NUM> outQueueMeanDist;

    AscendC::LocalTensor<float> knn_dists_sq_local; // Stores K best squared distances
    AscendC::LocalTensor<MinMax_knn> one_box_data_local; // To load one box's MinMax
    // AscendC::LocalTensor<float> points_from_one_box_local; // Removed, points fetched directly

    optiling::NpuSimpleKnnTilingData tilingData;
    uint64_t numElementsPerCore, numLoops, ubPartDataNum, tailDataNumInLoop, currentLoopDataNum;
    uint64_t global_point_idx_offset;
};

extern "C" __global__ __aicore__ void npu_knn_box_mean_dist(
    GM_ADDR points_gm, GM_ADDR sorted_indices_gm, GM_ADDR boxes_gm,
    GM_ADDR mean_dists_out_gm, GM_ADDR tiling_gm, aclrtStream stream_unused) {
    if (TILING_KEY_IS(1)) { KernelBoxMeanDist<true> op; op.Init(points_gm, sorted_indices_gm, boxes_gm, mean_dists_out_gm, tiling_gm); op.Process(); }
    else { KernelBoxMeanDist<false> op; op.Init(points_gm, sorted_indices_gm, boxes_gm, mean_dists_out_gm, tiling_gm); op.Process(); }
}


// TODO: Kernels for MinMax reduction and RadixSort (if custom implementation is needed)
// extern "C" __global__ __aicore__ void npu_knn_min_max_reduce_stage1(...)
// extern "C" __global__ __aicore__ void npu_knn_min_max_reduce_stage2(...)
// extern "C" __global__ __aicore__ void npu_knn_radix_sort_pairs_kernel(...)

// These would be complex and involve multiple stages/kernel launches.
// For now, they are assumed to be handled by library calls or (for sort) a placeholder in the C++ interface.
