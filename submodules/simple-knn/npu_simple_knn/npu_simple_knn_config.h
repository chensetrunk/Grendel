// npu_simple_knn_config.h

#ifndef NPU_SIMPLE_KNN_CONFIG_H_INCLUDED
#define NPU_SIMPLE_KNN_CONFIG_H_INCLUDED

// K for k-Nearest Neighbors (fixed at 3 in the original CUDA implementation)
#define NPU_KNN_K 3

// Default block size for 1D kernel launches (e.g., for processing P points)
// This might be adjusted based on NPU architecture or determined by tiling.
#define NPU_KNN_DEFAULT_BLOCK_SIZE 256

// Default box size for spatial boxing (was 1024 in CUDA)
// This affects how many points are grouped for min/max box calculation and search.
#define NPU_KNN_BOX_SIZE 1024

// Add any other compile-time constants specific to the npu_simple_knn module.
// Most parameters (like total point count P) will come from runtime arguments
// and be part of the TilingData.

#endif // NPU_SIMPLE_KNN_CONFIG_H_INCLUDED
