// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"

#include "PTOLowerToOpLibCalls.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

static int64_t getElemBytes(Type elemTy) {
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return (intTy.getWidth() + 7) / 8;
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return (floatTy.getWidth() + 7) / 8;
  return -1;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
  }
  return false;
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
  }
  return false;
}

static FailureOr<MemRefType> inferSimdBridgeMemRefType(pto::TileBufType tileTy,
                                                       MLIRContext *ctx) {
  if (tileTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> physicalShape = tileTy.getShape();
  if (physicalShape.size() != 2)
    return failure();
  if (physicalShape[0] == ShapedType::kDynamic ||
      physicalShape[1] == ShapedType::kDynamic)
    return failure();

  SmallVector<int64_t, 2> memShape(physicalShape.begin(), physicalShape.end());
  ArrayRef<int64_t> validShape = tileTy.getValidShape();
  if (validShape.size() == memShape.size()) {
    for (unsigned i = 0; i < validShape.size(); ++i)
      memShape[i] = validShape[i] < 0 ? physicalShape[i] : validShape[i];
  }

  auto cfg = tileTy.getConfigAttr();
  if (!cfg)
    cfg = pto::TileBufConfigAttr::getDefault(ctx);

  int32_t bl = 0;
  int32_t sl = 0;
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    fr = static_cast<int32_t>(attr.getInt());

  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (sl != 0) {
    int64_t elemBytes = getElemBytes(tileTy.getElementType());
    if (elemBytes <= 0)
      return failure();
    if (fr == 1024) {
      innerRows = 16;
      innerCols = 16;
    } else if (fr == 32) {
      innerRows = 16;
      innerCols = 2;
    } else if (fr == 512) {
      if (sl == 1) {
        innerRows = 16;
        innerCols = 32 / elemBytes;
      } else if (sl == 2) {
        innerRows = 32 / elemBytes;
        innerCols = 16;
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }

  SmallVector<int64_t, 2> strides;
  if (sl == 0) {
    if (bl == 1) {
      strides.push_back(1);
      strides.push_back(physicalShape[0]);
    } else {
      strides.push_back(physicalShape[1]);
      strides.push_back(1);
    }
  } else if (bl == 1) {
    if (sl != 1)
      return failure();
    strides.push_back(innerCols);
    strides.push_back(physicalShape[0]);
  } else {
    strides.push_back(physicalShape[1]);
    strides.push_back(innerRows);
  }

  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/0, strides);
  return MemRefType::get(memShape, tileTy.getElementType(), layout,
                         tileTy.getMemorySpace());
}

static bool areIntegerCarrierTypesCompatible(Type lhs, Type rhs) {
  auto lhsInt = dyn_cast<IntegerType>(lhs);
  auto rhsInt = dyn_cast<IntegerType>(rhs);
  if (!lhsInt || !rhsInt)
    return false;
  return lhsInt.getWidth() == rhsInt.getWidth();
}

static bool canRemapSimdBridgeViaCarrierCast(MemRefType actualTy,
                                             MemRefType templateTy) {
  if (actualTy.getRank() != templateTy.getRank())
    return false;
  if (actualTy.getMemorySpace() != templateTy.getMemorySpace())
    return false;
  return areIntegerCarrierTypesCompatible(actualTy.getElementType(),
                                          templateTy.getElementType());
}

static MemRefType remapMemRefToTemplateCarrier(MemRefType actualTy,
                                               MemRefType templateTy) {
  return MemRefType::get(actualTy.getShape(), templateTy.getElementType(),
                         actualTy.getLayout(), actualTy.getMemorySpace());
}

} // namespace

FailureOr<bool> mlir::pto::tryCloneOpLibInlineBridgeOp(OpBuilder &builder,
                                                       Operation &op,
                                                       IRMapping &mapping) {
  if (auto bridge = dyn_cast<pto::SimdTileToMemrefOp>(&op)) {
    Value mappedSrc = mapping.lookupOrNull(bridge.getSrc());
    if (!mappedSrc)
      return failure();

    auto templateMemTy = dyn_cast<MemRefType>(bridge.getDst().getType());
    if (auto mappedTileTy = dyn_cast<pto::TileBufType>(mappedSrc.getType())) {
      FailureOr<MemRefType> inferredTyOr =
          inferSimdBridgeMemRefType(mappedTileTy, builder.getContext());
      if (failed(inferredTyOr))
        return failure();

      auto inferredTy = *inferredTyOr;
      auto newBridge = builder.create<pto::SimdTileToMemrefOp>(
          bridge.getLoc(), inferredTy, mappedSrc);
      if (templateMemTy && inferredTy != templateMemTy &&
          canRemapSimdBridgeViaCarrierCast(inferredTy, templateMemTy)) {
        auto carrierTy = remapMemRefToTemplateCarrier(inferredTy, templateMemTy);
        auto cast = builder.create<UnrealizedConversionCastOp>(
            bridge.getLoc(), TypeRange{carrierTy}, ValueRange{newBridge.getDst()});
        mapping.map(bridge.getDst(), cast.getResult(0));
      } else {
        mapping.map(bridge.getDst(), newBridge.getDst());
      }
      return true;
    }

    auto mappedMemTy = dyn_cast<MemRefType>(mappedSrc.getType());
    auto dstMemTy = templateMemTy;
    if (!mappedMemTy || !dstMemTy)
      return failure();
    if (mappedMemTy.getRank() != dstMemTy.getRank())
      return failure();

    auto newBridge = builder.create<pto::SimdTileToMemrefOp>(
        bridge.getLoc(), mappedMemTy, mappedSrc);
    if (mappedMemTy.getElementType() == dstMemTy.getElementType()) {
      mapping.map(bridge.getDst(), newBridge.getDst());
      return true;
    }
    if (!canRemapSimdBridgeViaCarrierCast(mappedMemTy, dstMemTy))
      return failure();
    auto carrierTy = remapMemRefToTemplateCarrier(mappedMemTy, dstMemTy);
    auto cast = builder.create<UnrealizedConversionCastOp>(
        bridge.getLoc(), TypeRange{carrierTy}, ValueRange{newBridge.getDst()});
    mapping.map(bridge.getDst(), cast.getResult(0));
    return true;
  }

  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(&op)) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return failure();

    Value mappedSrc = mapping.lookupOrNull(cast.getOperand(0));
    if (!mappedSrc)
      return failure();

    Type dstTy = cast.getResult(0).getType();
    if (mappedSrc.getType() == dstTy) {
      mapping.map(cast.getResult(0), mappedSrc);
      return true;
    }

    auto clonedCast = builder.create<UnrealizedConversionCastOp>(
        cast.getLoc(), TypeRange{dstTy}, ValueRange{mappedSrc});
    mapping.map(cast.getResult(0), clonedCast.getResult(0));
    return true;
  }

  return false;
}
