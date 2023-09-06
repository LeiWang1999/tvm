/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inject_thread_rasterization.cc
 * \brief Transform annotated block into thread-rasterized block.
 */

#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../support/utils.h"
#include "../schedule/utils.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

namespace thread_rasterization {

/*! Structure that represents the provided annotation per block or loop. */
struct RasterizationInfo {
  int width;
};

class RasterizationInjector : private StmtExprMutator {
 public:
  static Stmt Inject(const PrimFunc& func) {
    RasterizationInjector injector;
    return injector(func->body);
  }

 private:
  explicit RasterizationInjector(){}
  Stmt VisitStmt_(const ForNode* op) final {
    // LOG(INFO) << "InjectThreadRasterization -> VisitStmt_(const ForNode* op)";
    // LOG(INFO) << "InjectThreadRasterization -> op->thread_binding" << op->thread_binding;
    // LOG(INFO) << "InjectThreadRasterization -> op->annotations" << op->annotations;

    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!HasRasterizationAnnotation(op)) {
      return std::move(for_node);
    }

    auto rasterization_stage = Downcast<Integer>(op->annotations.at(attr::thread_rasterization));
    int rasterization_width = rasterization_stage->value;
    // Create a RasterNode
    auto raster = Raster(ConstInt32(rasterization_width));
    Array<Stmt> seq;
    seq.push_back(raster);
    seq.push_back(for_node);
    auto seq_node = SeqStmt(seq);
    // Step 2: Rewrite the current node.
    // combine raster with for_node into a stmt

    return std::move(seq_node);
  }

  bool HasRasterizationAnnotation(const ForNode* op) const {
    auto it = op->annotations.find(attr::thread_rasterization);
    bool has_annotation = it != op->annotations.end();
    if (has_annotation) {
      return true;
    }
    return false;
  }
};

}  // namespace thread_rasterization

namespace transform {

/*!
 * \brief Transform annotated block into threa block rasteration form.
 * \return The IR transform pass.
 */
Pass InjectThreadRasterization() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    fptr->body =
        thread_rasterization::RasterizationInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectThreadRasterization", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectThreadRasterization").set_body_typed(InjectThreadRasterization);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
