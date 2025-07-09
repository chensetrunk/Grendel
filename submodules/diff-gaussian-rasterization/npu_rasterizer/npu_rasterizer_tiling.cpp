// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use
// under the terms of the LICENSE.md file.
//
// For inquiries contact  george.drettakis@inria.fr

#include "npu_rasterizer_tiling.h"
#include "npu_config.h" // For NPU_NUM_CHANNELS
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector> // Required for std::vector

namespace optiling {

// Define constants, similar to add_custom_tiling.cpp
const uint64_t BLOCK_SIZE_BYTES = 32; // NPU memory alignment / block size in bytes
const uint64_t BUFFER_NUM = 2;      // For double buffering in UB

// Helper to estimate UB size needed per element for various kernels
uint32_t estimateBytesPerElementPreprocess(uint32_t M_sh) {
    uint32_t bytes = 0;
    bytes += 3 * sizeof(float); // means3D
    bytes += 3 * sizeof(float); // scales
    bytes += 4 * sizeof(float); // rotations
    bytes += 1 * sizeof(float); // opacities
    if (M_sh > 0) bytes += M_sh * 3 * sizeof(float); // shs
    bytes += 2 * sizeof(float); // means2D_out
    bytes += 1 * sizeof(float); // depths_out
    bytes += 1 * sizeof(int);   // radii_out
    bytes += 6 * sizeof(float); // cov3Ds_computed_out
    bytes += 4 * sizeof(float); // conic_opacity_out
    bytes += 3 * sizeof(float); // rgb_out
    bytes += 3 * sizeof(bool);  // clamped_out (practical size might be 3 bytes or more)
    return bytes;
}

uint32_t estimateBytesPerElementRender() {
    uint32_t bytes = 0;
    bytes += 2 * sizeof(float); // points_xy_image
    bytes += NPU_NUM_CHANNELS * sizeof(float); // features
    bytes += 4 * sizeof(float); // conic_opacity
    // Add size for collected_ids if that's also a main buffer in UB loops
    bytes += 1 * sizeof(uint32_t); // collected_id
    return bytes;
}

uint32_t estimateBytesPerElementRenderBackward() {
    uint32_t bytes = estimateBytesPerElementRender(); // Similar inputs as forward
    // Additional for dL_dpixel, accum_rec etc.
    // This is per-pixel, not per-Gaussian batch, so UB for Gaussian data is key.
    return bytes;
}

uint32_t estimateBytesPerElementCov2DBackward() {
    uint32_t bytes = 0;
    bytes += 3 * sizeof(float); // means3D
    bytes += 1 * sizeof(int);   // radii
    bytes += 6 * sizeof(float); // cov3Ds_fwd
    bytes += 4 * sizeof(float); // dL_dconics (actually 3 for conic, 1 for opacity handled separately)
    // Outputs
    bytes += 3 * sizeof(float); // dL_dmean3D_cov
    bytes += 6 * sizeof(float); // dL_dcov3D
    return bytes;
}

uint32_t estimateBytesPerElementPreprocessBackward(uint32_t M_sh) {
    uint32_t bytes = 0;
    // Inputs
    bytes += 3 * sizeof(float); // means3D
    bytes += 3 * sizeof(float); // scales
    bytes += 4 * sizeof(float); // rotations
    if (M_sh > 0) bytes += M_sh * 3 * sizeof(float); // shs_fwd
    bytes += 1 * sizeof(int);   // radii_fwd
    bytes += 3 * sizeof(bool);  // clamped_fwd
    bytes += 2 * sizeof(float); // dL_dmean2D_render
    bytes += 3 * sizeof(float); // dL_dmean3D_cov
    bytes += 3 * sizeof(float); // dL_dcolors_render
    bytes += 6 * sizeof(float); // dL_dcov3D_cov
    // Outputs
    bytes += 3 * sizeof(float); // dL_dmeans3D_total
    if (M_sh > 0) bytes += M_sh * 3 * sizeof(float); // dL_dshs
    bytes += 3 * sizeof(float); // dL_dscales
    bytes += 4 * sizeof(float); // dL_drots
    return bytes;
}


// Generic tiling logic for per-Gaussian kernels
static ge::graphStatus PerGaussianKernelTilingFunc(gert::TilingContext* context, NpuRasterizerTilingData& tiling, uint32_t bytes_per_gaussian_ub_approx) {
    uint64_t ubLength = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto totalCoreNum = ascendcPlatform.GetCoreNum();

    int64_t P_attr = 0;
    context->GetOpDesc()->GetAttr("P", P_attr);
    uint64_t P = static_cast<uint64_t>(P_attr);

    if (totalCoreNum == 0 || (P > 0 && bytes_per_gaussian_ub_approx == 0) ) {
        return ge::GRAPH_FAILED;
    }
    if (P == 0) { // No work to do
         tiling.set_P(0);
         tiling.set_smallCoreDataNum(0); tiling.set_bigCoreDataNum(0); tiling.set_ubPartDataNum(0);
         tiling.set_smallCoreLoopNum(0); tiling.set_bigCoreLoopNum(0);
         tiling.set_smallCoreTailDataNum(0); tiling.set_bigCoreTailDataNum(0);
         tiling.set_tailBlockNum(0);
         context->SetBlockDim(1);
         context->SetTilingKey(0);
         tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
         context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
         size_t *ws = context->GetWorkspaceSizes(1); ws[0] = 0;
         return ge::GRAPH_SUCCESS;
    }

    uint64_t ubPartDataNum = (ubLength / BUFFER_NUM) / bytes_per_gaussian_ub_approx;
    if (ubPartDataNum == 0) ubPartDataNum = 1;

    uint64_t effectiveCoreNum = totalCoreNum;
    if (P < totalCoreNum) effectiveCoreNum = P;
    if (P <= ubPartDataNum && P > 0) effectiveCoreNum = 1;


    uint64_t gaussiansPerCoreBase = P / effectiveCoreNum;
    uint64_t tailGaussiansExtra = P % effectiveCoreNum;

    tiling.set_smallCoreDataNum(gaussiansPerCoreBase);
    tiling.set_bigCoreDataNum(gaussiansPerCoreBase + (tailGaussiansExtra > 0 ? 1 : 0));
    tiling.set_tailBlockNum(tailGaussiansExtra);

    if (tiling.smallCoreDataNum() == 0 && tiling.bigCoreDataNum() > 0) {
        tiling.set_smallCoreDataNum(tiling.bigCoreDataNum());
    } else if (tiling.smallCoreDataNum() == 0 && tiling.bigCoreDataNum() == 0 && P > 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_ubPartDataNum(ubPartDataNum);

    tiling.set_smallCoreLoopNum((tiling.smallCoreDataNum() + ubPartDataNum - 1) / ubPartDataNum);
    if (tiling.smallCoreDataNum() > 0 && tiling.smallCoreLoopNum() == 0) tiling.set_smallCoreLoopNum(1); // Ensure at least 1 loop if data exists
    uint64_t small_tail = tiling.smallCoreDataNum() % ubPartDataNum;
    tiling.set_smallCoreTailDataNum( (small_tail == 0 && tiling.smallCoreDataNum() > 0) ? ubPartDataNum : small_tail);

    tiling.set_bigCoreLoopNum((tiling.bigCoreDataNum() + ubPartDataNum - 1) / ubPartDataNum);
    if (tiling.bigCoreDataNum() > 0 && tiling.bigCoreLoopNum() == 0) tiling.set_bigCoreLoopNum(1);
    uint64_t big_tail = tiling.bigCoreDataNum() % ubPartDataNum;
    tiling.set_bigCoreTailDataNum( (big_tail == 0 && tiling.bigCoreDataNum() > 0) ? ubPartDataNum : big_tail);

    context->SetBlockDim(effectiveCoreNum);
    context->SetTilingKey(tiling.tailBlockNum() > 0 ? 1 : 0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// Tiling function for PreprocessForward
static ge::graphStatus PreprocessForwardTilingFunc(gert::TilingContext* context) {
    NpuRasterizerTilingData tiling;
    auto op_desc = context->GetOpDesc();
    int64_t P_attr, D_attr, M_attr, img_w_attr, img_h_attr, blk_x_attr, blk_y_attr;
    float tan_fx_attr, tan_fy_attr, foc_x_attr, foc_y_attr, scale_mod_attr;

    op_desc->GetAttr("P", P_attr); tiling.set_P(P_attr);
    op_desc->GetAttr("D_sh", D_attr); tiling.set_D(D_attr);
    op_desc->GetAttr("M_sh", M_attr); tiling.set_M(M_attr);
    op_desc->GetAttr("image_width", img_w_attr); tiling.set_image_width(img_w_attr);
    op_desc->GetAttr("image_height", img_h_attr); tiling.set_image_height(img_h_attr);
    op_desc->GetAttr("tan_fovx", tan_fx_attr); tiling.set_tan_fovx(tan_fx_attr);
    op_desc->GetAttr("tan_fovy", tan_fy_attr); tiling.set_tan_fovy(tan_fy_attr);
    op_desc->GetAttr("focal_x", foc_x_attr); tiling.set_focal_x(foc_x_attr);
    op_desc->GetAttr("focal_y", foc_y_attr); tiling.set_focal_y(foc_y_attr);
    op_desc->GetAttr("scale_modifier", scale_mod_attr); tiling.set_scale_modifier(scale_mod_attr);
    op_desc->GetAttr("block_x_render", blk_x_attr); tiling.set_block_x(blk_x_attr);
    op_desc->GetAttr("block_y_render", blk_y_attr); tiling.set_block_y(blk_y_attr);
    tiling.set_num_channels(NPU_NUM_CHANNELS);

    uint32_t bytes_per_gauss_ub = estimateBytesPerElementPreprocess(tiling.M());
    return PerGaussianKernelTilingFunc(context, tiling, bytes_per_gauss_ub);
}

// Generic tiling for Render kernels (Forward and Backward)
static ge::graphStatus RenderKernelTilingFunc(gert::TilingContext* context, NpuRasterizerTilingData& tiling) {
    uint64_t ubLength = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto totalCoreNum = ascendcPlatform.GetCoreNum();

    auto op_desc = context->GetOpDesc();
    int64_t img_w_attr, img_h_attr, blk_x_attr, blk_y_attr;
    op_desc->GetAttr("image_width", img_w_attr); tiling.set_image_width(img_w_attr);
    op_desc->GetAttr("image_height", img_h_attr); tiling.set_image_height(img_h_attr);
    op_desc->GetAttr("block_x_render", blk_x_attr); tiling.set_block_x(blk_x_attr);
    op_desc->GetAttr("block_y_render", blk_y_attr); tiling.set_block_y(blk_y_attr);
    tiling.set_num_channels(NPU_NUM_CHANNELS);

    uint32_t tile_grid_x = (tiling.image_width() + tiling.block_x() - 1) / tiling.block_x();
    uint32_t tile_grid_y = (tiling.image_height() + tiling.block_y() - 1) / tiling.block_y();
    tiling.set_tile_grid_x(tile_grid_x);
    tiling.set_tile_grid_y(tile_grid_y);
    uint64_t total_tiles = tile_grid_x * tile_grid_y;

    if (totalCoreNum == 0 || total_tiles == 0) {
        context->SetBlockDim(1); context->SetTilingKey(0);
        tiling.set_core_tile_start_idx_flat(0); tiling.set_core_num_tiles_to_process(0);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *ws = context->GetWorkspaceSizes(1); ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t effectiveCoreNum = (totalCoreNum < total_tiles) ? totalCoreNum : total_tiles;

    // Kernels will derive their tile range from coreIdx and effectiveCoreNum.
    // No need to store per-core tile ranges in TilingData if this derivation is simple.

    uint32_t bytes_per_gaussian_render_ub = estimateBytesPerElementRender(); // Could be different for backward
    uint64_t ubPartDataNum_render = (ubLength / BUFFER_NUM) / bytes_per_gaussian_render_ub;
    if (ubPartDataNum_render == 0) ubPartDataNum_render = 1;
    tiling.set_ubPartDataNum(ubPartDataNum_render);

    context->SetBlockDim(effectiveCoreNum);
    context->SetTilingKey(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RenderForwardTilingFunc(gert::TilingContext* context) {
    NpuRasterizerTilingData tiling;
    return RenderKernelTilingFunc(context, tiling);
}

static ge::graphStatus RenderBackwardTilingFunc(gert::TilingContext* context) {
    NpuRasterizerTilingData tiling;
    // Note: estimateBytesPerElementRenderBackward() might be different.
    // For now, using the same as forward for ubPartDataNum calculation.
    return RenderKernelTilingFunc(context, tiling);
}

static ge::graphStatus ComputeCov2DBackwardTilingFunc(gert::TilingContext* context) {
    NpuRasterizerTilingData tiling;
    auto op_desc = context->GetOpDesc();
    int64_t P_attr; float tan_fx_attr, tan_fy_attr, foc_x_attr, foc_y_attr;
    op_desc->GetAttr("P", P_attr); tiling.set_P(P_attr);
    op_desc->GetAttr("tan_fovx", tan_fx_attr); tiling.set_tan_fovx(tan_fx_attr);
    op_desc->GetAttr("tan_fovy", tan_fy_attr); tiling.set_tan_fovy(tan_fy_attr);
    op_desc->GetAttr("focal_x", foc_x_attr); tiling.set_focal_x(foc_x_attr);
    op_desc->GetAttr("focal_y", foc_y_attr); tiling.set_focal_y(foc_y_attr);
    tiling.set_num_channels(NPU_NUM_CHANNELS); // Not directly used but good to have

    uint32_t bytes_per_gauss_ub = estimateBytesPerElementCov2DBackward();
    return PerGaussianKernelTilingFunc(context, tiling, bytes_per_gauss_ub);
}

static ge::graphStatus PreprocessBackwardTilingFunc(gert::TilingContext* context) {
    NpuRasterizerTilingData tiling;
    auto op_desc = context->GetOpDesc();
    int64_t P_attr, D_attr, M_attr; float scale_mod_attr;
    op_desc->GetAttr("P", P_attr); tiling.set_P(P_attr);
    op_desc->GetAttr("D_sh", D_attr); tiling.set_D(D_attr);
    op_desc->GetAttr("M_sh", M_attr); tiling.set_M(M_attr);
    op_desc->GetAttr("scale_modifier", scale_mod_attr); tiling.set_scale_modifier(scale_mod_attr);
    tiling.set_num_channels(NPU_NUM_CHANNELS);

    uint32_t bytes_per_gauss_ub = estimateBytesPerElementPreprocessBackward(tiling.M());
    return PerGaussianKernelTilingFunc(context, tiling, bytes_per_gauss_ub);
}

} // namespace optiling

// --- InferShape and InferDataType Functions ---
namespace ge {

static ge::graphStatus NpuRasterizerPreprocessForwardInferShape(gert::InferShapeContext* context) {
    const gert::Shape* means3D_shape = context->GetInputShape(0);
    int64_t P = means3D_shape->GetDim(0);
    auto op_desc = context->GetOpDesc();
    int64_t M_sh_attr; op_desc->GetAttr("M_sh", M_sh_attr);


    context->SetOutputShape("means2D_out", gert::Shape({P, 2}));
    context->SetOutputShape("depths_out", gert::Shape({P}));
    context->SetOutputShape("radii_out", gert::Shape({P}));
    context->SetOutputShape("cov3Ds_computed_out", gert::Shape({P, 6}));
    context->SetOutputShape("conic_opacity_out", gert::Shape({P, 4}));
    context->SetOutputShape("rgb_out", gert::Shape({P, (int64_t)NPU_NUM_CHANNELS}));
    context->SetOutputShape("clamped_out", gert::Shape({P, (int64_t)NPU_NUM_CHANNELS})); // 3 bools for RGB
    return GRAPH_SUCCESS;
}

static ge::graphStatus NpuRasterizerPreprocessForwardInferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType("means2D_out", ge::DT_FLOAT);
    context->SetOutputDataType("depths_out", ge::DT_FLOAT);
    context->SetOutputDataType("radii_out", ge::DT_INT32);
    context->SetOutputDataType("cov3Ds_computed_out", ge::DT_FLOAT);
    context->SetOutputDataType("conic_opacity_out", ge::DT_FLOAT);
    context->SetOutputDataType("rgb_out", ge::DT_FLOAT);
    context->SetOutputDataType("clamped_out", ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NpuRasterizerRenderForwardInferShape(gert::InferShapeContext* context) {
    auto op_desc = context->GetOpDesc();
    int64_t H_attr, W_attr;
    op_desc->GetAttr("image_height", H_attr);
    op_desc->GetAttr("image_width", W_attr);

    context->SetOutputShape("out_color", gert::Shape({(int64_t)NPU_NUM_CHANNELS, H_attr, W_attr}));
    context->SetOutputShape("final_T", gert::Shape({H_attr, W_attr}));
    context->SetOutputShape("n_contrib", gert::Shape({H_attr, W_attr}));
    return GRAPH_SUCCESS;
}

static ge::graphStatus NpuRasterizerRenderForwardInferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType("out_color", ge::DT_FLOAT);
    context->SetOutputDataType("final_T", ge::DT_FLOAT);
    context->SetOutputDataType("n_contrib", ge::DT_UINT32);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NpuRasterizerRenderBackwardInferShape(gert::InferShapeContext* context) {
    int64_t P_attr; context->GetOpDesc()->GetAttr("P", P_attr);
    context->SetOutputShape("dL_dmean2D", gert::Shape({P_attr, 2}));
    context->SetOutputShape("dL_dconic", gert::Shape({P_attr, 3}));
    context->SetOutputShape("dL_dopacity", gert::Shape({P_attr}));
    context->SetOutputShape("dL_dcolors", gert::Shape({P_attr, (int64_t)NPU_NUM_CHANNELS}));
    return GRAPH_SUCCESS;
}
static ge::graphStatus NpuRasterizerRenderBackwardInferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType("dL_dmean2D", ge::DT_FLOAT);
    context->SetOutputDataType("dL_dconic", ge::DT_FLOAT);
    context->SetOutputDataType("dL_dopacity", ge::DT_FLOAT);
    context->SetOutputDataType("dL_dcolors", ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NpuRasterizerComputeCov2DBackwardInferShape(gert::InferShapeContext* context) {
    int64_t P_attr; context->GetOpDesc()->GetAttr("P", P_attr);
    context->SetOutputShape("dL_dmean3D_cov", gert::Shape({P_attr, 3}));
    context->SetOutputShape("dL_dcov3D", gert::Shape({P_attr, 6}));
    return GRAPH_SUCCESS;
}
static ge::graphStatus NpuRasterizerComputeCov2DBackwardInferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType("dL_dmean3D_cov", ge::DT_FLOAT);
    context->SetOutputDataType("dL_dcov3D", ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NpuRasterizerPreprocessBackwardInferShape(gert::InferShapeContext* context) {
    int64_t P_attr; context->GetOpDesc()->GetAttr("P", P_attr);
    int64_t M_sh_attr; context->GetOpDesc()->GetAttr("M_sh", M_sh_attr);

    context->SetOutputShape("dL_dmeans3D_total", gert::Shape({P_attr,3}));
    context->SetOutputShape("dL_dshs", gert::Shape({P_attr, M_sh_attr, 3}));
    context->SetOutputShape("dL_dscales", gert::Shape({P_attr,3}));
    context->SetOutputShape("dL_drots", gert::Shape({P_attr,4}));
    return GRAPH_SUCCESS;
}
static ge::graphStatus NpuRasterizerPreprocessBackwardInferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType("dL_dmeans3D_total", ge::DT_FLOAT);
    context->SetOutputDataType("dL_dshs", ge::DT_FLOAT);
    context->SetOutputDataType("dL_dscales", ge::DT_FLOAT);
    context->SetOutputDataType("dL_drots", ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}


} // namespace ge

// --- Operator Definitions ---
namespace ops {

// --- OpDef for PreprocessForward ---
class NpuRasterizerPreprocessForwardOp : public OpDef {
public:
    explicit NpuRasterizerPreprocessForwardOp(const char* name) : OpDef(name) {
        this->Input("means3D").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("scales").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("rotations").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("opacities").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("shs").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("cov3D_precomp").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("colors_precomp").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("viewmatrix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("projmatrix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("cam_pos").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Output("means2D_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("depths_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("radii_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("cov3Ds_computed_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("conic_opacity_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("rgb_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("clamped_out").ParamType(REQUIRED).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND});

        this->Attr("P", ge::AttrValue::INT{0});
        this->Attr("D_sh", ge::AttrValue::INT{0});
        this->Attr("M_sh", ge::AttrValue::INT{0});
        this->Attr("image_width", ge::AttrValue::INT{0});
        this->Attr("image_height", ge::AttrValue::INT{0});
        this->Attr("tan_fovx", ge::AttrValue::FLOAT{0.0f});
        this->Attr("tan_fovy", ge::AttrValue::FLOAT{0.0f});
        this->Attr("focal_x", ge::AttrValue::FLOAT{0.0f});
        this->Attr("focal_y", ge::AttrValue::FLOAT{0.0f});
        this->Attr("scale_modifier", ge::AttrValue::FLOAT{1.0f});
        this->Attr("prefiltered", ge::AttrValue::BOOL{false}); // Not used by kernel yet
        this->Attr("block_x_render", ge::AttrValue::INT{16});
        this->Attr("block_y_render", ge::AttrValue::INT{16});

        this->SetInferShape(ge::NpuRasterizerPreprocessForwardInferShape);
        this->SetInferDataType(ge::NpuRasterizerPreprocessForwardInferDataType);

        this->AICore().SetTiling(optiling::PreprocessForwardTilingFunc)
            .AddConfig("ascend910").AddConfig("ascend910b").AddConfig("ascend310p");
    }
};
OP_ADD(NpuRasterizerPreprocessForwardOp);


// --- OpDef for RenderForward ---
class NpuRasterizerRenderForwardOp : public OpDef {
public:
    explicit NpuRasterizerRenderForwardOp(const char* name) : OpDef(name) {
        this->Input("ranges").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("point_list").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("points_xy_image").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("features").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("conic_opacity").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("compute_locally").ParamType(OPTIONAL).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND});
        this->Input("bg_color").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Output("out_color").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("final_T").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("n_contrib").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});

        this->Attr("image_width", ge::AttrValue::INT{0});
        this->Attr("image_height", ge::AttrValue::INT{0});
        this->Attr("block_x_render", ge::AttrValue::INT{16});
        this->Attr("block_y_render", ge::AttrValue::INT{16});

        this->SetInferShape(ge::NpuRasterizerRenderForwardInferShape);
        this->SetInferDataType(ge::NpuRasterizerRenderForwardInferDataType);
        this->AICore().SetTiling(optiling::RenderForwardTilingFunc)
            .AddConfig("ascend910").AddConfig("ascend910b").AddConfig("ascend310p");
    }
};
OP_ADD(NpuRasterizerRenderForwardOp);


// --- OpDef for RenderBackward ---
class NpuRasterizerRenderBackwardOp : public OpDef {
public:
    explicit NpuRasterizerRenderBackwardOp(const char* name) : OpDef(name) {
        this->Input("dL_dpixels").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("ranges").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("point_list").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("points_xy_image").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("conic_opacity").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("colors").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("final_Ts").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("n_contrib").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("bg_color").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("compute_locally").ParamType(OPTIONAL).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND});

        this->Output("dL_dmean2D").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("dL_dconic").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("dL_dopacity").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("dL_dcolors").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Attr("image_width", ge::AttrValue::INT{0});
        this->Attr("image_height", ge::AttrValue::INT{0});
        this->Attr("block_x_render", ge::AttrValue::INT{16});
        this->Attr("block_y_render", ge::AttrValue::INT{16});
        this->Attr("P", ge::AttrValue::INT{0});

        this->SetInferShape(ge::NpuRasterizerRenderBackwardInferShape);
        this->SetInferDataType(ge::NpuRasterizerRenderBackwardInferDataType);
        this->AICore().SetTiling(optiling::RenderBackwardTilingFunc)
            .AddConfig("ascend910").AddConfig("ascend910b").AddConfig("ascend310p");
    }
};
OP_ADD(NpuRasterizerRenderBackwardOp);


// --- OpDef for ComputeCov2DBackward ---
class NpuRasterizerComputeCov2DBackwardOp : public OpDef {
public:
    explicit NpuRasterizerComputeCov2DBackwardOp(const char* name) : OpDef(name) {
        this->Input("dL_dconics").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // Px4 (conic_a,b,opacity,c)
        this->Input("means3D").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("radii").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("cov3Ds_fwd").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // Px6
        this->Input("viewmatrix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // 16

        this->Output("dL_dmean3D_cov").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // Px3
        this->Output("dL_dcov3D").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // Px6

        this->Attr("P", ge::AttrValue::INT{0});
        this->Attr("tan_fovx", ge::AttrValue::FLOAT{0.0f});
        this->Attr("tan_fovy", ge::AttrValue::FLOAT{0.0f});
        this->Attr("focal_x", ge::AttrValue::FLOAT{0.0f});
        this->Attr("focal_y", ge::AttrValue::FLOAT{0.0f});

        this->SetInferShape(ge::NpuRasterizerComputeCov2DBackwardInferShape);
        this->SetInferDataType(ge::NpuRasterizerComputeCov2DBackwardInferDataType);
        this->AICore().SetTiling(optiling::ComputeCov2DBackwardTilingFunc)
            .AddConfig("ascend910").AddConfig("ascend910b").AddConfig("ascend310p");
    }
};
OP_ADD(NpuRasterizerComputeCov2DBackwardOp);


// --- OpDef for PreprocessBackward ---
class NpuRasterizerPreprocessBackwardOp : public OpDef {
public:
    explicit NpuRasterizerPreprocessBackwardOp(const char* name) : OpDef(name) {
        this->Input("means3D").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("scales").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("rotations").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("shs_fwd").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}); // From forward (or original)
        this->Input("radii_fwd").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("clamped_fwd").ParamType(REQUIRED).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND});
        this->Input("projmatrix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("cam_pos").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("dL_dmean2D_render").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("dL_dmean3D_cov").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("dL_dcolors_render").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("dL_dcov3D_cov").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Output("dL_dmeans3D_total").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("dL_dshs").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("dL_dscales").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("dL_drots").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Attr("P", ge::AttrValue::INT{0});
        this->Attr("D_sh", ge::AttrValue::INT{0});
        this->Attr("M_sh", ge::AttrValue::INT{0});
        this->Attr("scale_modifier", ge::AttrValue::FLOAT{1.0f});

        this->SetInferShape(ge::NpuRasterizerPreprocessBackwardInferShape);
        this->SetInferDataType(ge::NpuRasterizerPreprocessBackwardInferDataType);
        this->AICore().SetTiling(optiling::PreprocessBackwardTilingFunc)
            .AddConfig("ascend910").AddConfig("ascend910b").AddConfig("ascend310p");
    }
};
OP_ADD(NpuRasterizerPreprocessBackwardOp);


} // namespace ops
>>>>>>> REPLACE
