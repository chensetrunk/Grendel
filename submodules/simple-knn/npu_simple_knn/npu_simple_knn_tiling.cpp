// npu_simple_knn_tiling.cpp

#include "npu_simple_knn_tiling.h"
#include "npu_simple_knn_config.h" // For NPU_KNN_K, NPU_KNN_BOX_SIZE
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>

namespace optiling {

// Constants for Tiling
const uint64_t TILING_BLOCK_SIZE_BYTES = 32; // NPU memory alignment
const uint64_t TILING_BUFFER_NUM = 2;      // For double buffering in UB

// Generic Tiling Function for Simple KNN stages that iterate over P points
// (e.g., Morton Coding, part of BoxMinMax, part of BoxMeanDist)
static ge::graphStatus KnnPointsIterTilingFunc(gert::TilingContext* context, NpuSimpleKnnTilingData& tiling_data, uint32_t bytes_per_point_ub) {
    uint64_t ubLength = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto totalCoreNum = ascendcPlatform.GetCoreNum();

    int64_t P_attr = 0;
    context->GetOpDesc()->GetAttr("P", P_attr); // Assuming P is an attribute
    uint64_t P = static_cast<uint64_t>(P_attr);
    tiling_data.set_P(P);

    if (totalCoreNum == 0 || (P > 0 && bytes_per_point_ub == 0)) {
        return ge::GRAPH_FAILED;
    }
     if (P == 0) {
         tiling_data.set_smallCoreDataNum(0); tiling_data.set_bigCoreDataNum(0); tiling_data.set_ubPartDataNum(0);
         tiling_data.set_smallCoreLoopNum(0); tiling_data.set_bigCoreLoopNum(0);
         tiling_data.set_smallCoreTailDataNum(0); tiling_data.set_bigCoreTailDataNum(0);
         tiling_data.set_tailBlockNum(0);
         context->SetBlockDim(1); context->SetTilingKey(0);
         tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
         context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());
         size_t *ws = context->GetWorkspaceSizes(1); ws[0] = 0;
         return ge::GRAPH_SUCCESS;
    }

    uint64_t ubPartDataNum = (ubLength / TILING_BUFFER_NUM) / bytes_per_point_ub;
    if (ubPartDataNum == 0) ubPartDataNum = 1;

    uint64_t effectiveCoreNum = totalCoreNum;
    if (P < totalCoreNum) effectiveCoreNum = P;
    if (P <= ubPartDataNum && P > 0) effectiveCoreNum = 1;

    uint64_t pointsPerCoreBase = P / effectiveCoreNum;
    uint64_t tailPointsExtra = P % effectiveCoreNum;

    tiling_data.set_smallCoreDataNum(pointsPerCoreBase);
    tiling_data.set_bigCoreDataNum(pointsPerCoreBase + (tailPointsExtra > 0 ? 1 : 0));
    tiling_data.set_tailBlockNum(tailPointsExtra);

    if (tiling_data.smallCoreDataNum() == 0 && tiling_data.bigCoreDataNum() > 0) {
        tiling_data.set_smallCoreDataNum(tiling_data.bigCoreDataNum());
    } else if (tiling_data.smallCoreDataNum() == 0 && tiling_data.bigCoreDataNum() == 0 && P > 0) {
        return ge::GRAPH_FAILED;
    }

    tiling_data.set_ubPartDataNum(ubPartDataNum);
    tiling_data.set_smallCoreLoopNum((tiling_data.smallCoreDataNum() + ubPartDataNum - 1) / ubPartDataNum);
    if (tiling_data.smallCoreDataNum() > 0 && tiling_data.smallCoreLoopNum() == 0) tiling_data.set_smallCoreLoopNum(1);
    uint64_t small_tail = tiling_data.smallCoreDataNum() % ubPartDataNum;
    tiling_data.set_smallCoreTailDataNum((small_tail == 0 && tiling_data.smallCoreDataNum() > 0) ? ubPartDataNum : small_tail);

    tiling_data.set_bigCoreLoopNum((tiling_data.bigCoreDataNum() + ubPartDataNum - 1) / ubPartDataNum);
     if (tiling_data.bigCoreDataNum() > 0 && tiling_data.bigCoreLoopNum() == 0) tiling_data.set_bigCoreLoopNum(1);
    uint64_t big_tail = tiling_data.bigCoreDataNum() % ubPartDataNum;
    tiling_data.set_bigCoreTailDataNum((big_tail == 0 && tiling_data.bigCoreDataNum() > 0) ? ubPartDataNum : big_tail);

    context->SetBlockDim(effectiveCoreNum);
    context->SetTilingKey(tiling_data.tailBlockNum() > 0 ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}


// Tiling function for the main NpuSimpleKnnOp (if monolithic)
// This would be complex as it needs to tile for multiple internal stages.
// For now, let's assume it primarily tiles for the most dominant stage, e.g., boxMeanDist.
static ge::graphStatus NpuSimpleKnnOpTilingFunc(gert::TilingContext* context) {
    NpuSimpleKnnTilingData tiling_data;

    // Populate attributes P, K, min_coord, max_coord etc. from OpDesc
    auto op_desc = context->GetOpDesc();
    int64_t P_attr; op_desc->GetAttr("P", P_attr); tiling_data.set_P(P_attr);
    // K is fixed for now by NPU_KNN_K, but could be an attribute
    tiling_data.set_K(NPU_KNN_K);
    // Min/max coords might be computed by a prior kernel, or passed as attributes if known
    // For simplicity, assume they might come from attributes or a small fixed input tensor
    // If they are computed by a first kernel stage within this Op, this tiling func gets more complex.
    // Let's assume they are attributes for now for this example tiling.
    float min_x, min_y, min_z, max_x, max_y, max_z;
    op_desc->GetAttr("min_coord_x", min_x); tiling_data.set_min_coord_x(min_x);
    op_desc->GetAttr("min_coord_y", min_y); tiling_data.set_min_coord_y(min_y);
    op_desc->GetAttr("min_coord_z", min_z); tiling_data.set_min_coord_z(min_z);
    op_desc->GetAttr("max_coord_x", max_x); tiling_data.set_max_coord_x(max_x);
    op_desc->GetAttr("max_coord_y", max_y); tiling_data.set_max_coord_y(max_y);
    op_desc->GetAttr("max_coord_z", max_z); tiling_data.set_max_coord_z(max_z);

    tiling_data.set_box_size(NPU_KNN_BOX_SIZE);
    uint32_t num_boxes_val = (P_attr + NPU_KNN_BOX_SIZE -1) / NPU_KNN_BOX_SIZE;
    tiling_data.set_num_boxes(num_boxes_val);

    // Estimate UB usage for the most intensive part, e.g., boxMeanDist
    // Input: point (3f), k-best distances (Kf). Output: meanDist (1f).
    // Temp: box (6f). Inner loop loads points (3f).
    // This estimation is very rough.
    uint32_t bytes_per_point_boxmeandist = (3 + NPU_KNN_K + 1 + 6 + 3) * sizeof(float);
    KnnPointsIterTilingFunc(context, tiling_data, bytes_per_point_boxmeandist);

    tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());
    size_t *ws = context->GetWorkspaceSizes(1); ws[0] = 0; // Example, no extra workspace from tiling
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// --- InferShape and InferDataType Functions ---
namespace ge {

static ge::graphStatus NpuSimpleKnnOpInferShape(gert::InferShapeContext* context) {
    const gert::Shape* points_shape = context->GetInputShape(0); // "points" input
    int64_t P = points_shape->GetDim(0); // Number of points

    // Output is mean distances, one per point
    context->SetOutputShape("mean_dists", gert::Shape({P}));
    return GRAPH_SUCCESS;
}

static ge::graphStatus NpuSimpleKnnOpInferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType("mean_dists", ge::DT_FLOAT); // Output is float distances
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

// --- Operator Definition ---
namespace ops {

class NpuSimpleKnnOp : public OpDef {
public:
    explicit NpuSimpleKnnOp(const char* name) : OpDef(name) {
        this->Input("points") // Input point cloud
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // Shape (P, 3)

        this->Output("mean_dists") // Output mean distances
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // Shape (P)

        // Attributes
        this->Attr("P", ge::AttrValue::INT{0}); // Number of points, derived from input tensor shape
        // Min/max coordinates for Morton coding, can be attributes or computed internally
        this->Attr("min_coord_x", ge::AttrValue::FLOAT{0.0f});
        this->Attr("min_coord_y", ge::AttrValue::FLOAT{0.0f});
        this->Attr("min_coord_z", ge::AttrValue::FLOAT{0.0f});
        this->Attr("max_coord_x", ge::AttrValue::FLOAT{0.0f});
        this->Attr("max_coord_y", ge::AttrValue::FLOAT{0.0f});
        this->Attr("max_coord_z", ge::AttrValue::FLOAT{0.0f});
        // K is fixed by NPU_KNN_K in config for now, could be an attribute if variable K is needed.

        this->SetInferShape(ge::NpuSimpleKnnOpInferShape);
        this->SetInferDataType(ge::NpuSimpleKnnOpInferDataType);

        this->AICore()
            .SetTiling(optiling::NpuSimpleKnnOpTilingFunc)
            .AddConfig("ascend910b") // Example NPU target
            .AddConfig("ascend310p");
            // Add other supported NPU configs
    }
};
OP_ADD(NpuSimpleKnnOp); // Register the operator

} // namespace ops
