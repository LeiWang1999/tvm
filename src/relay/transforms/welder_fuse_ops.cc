/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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

#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>
#include <queue>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

struct TaggedNode {
  TaggedNode(const ExprNode* node, int id)
    : node_(node), id_(id)
  {
    visited_ = false;
    inlined_ = false;
    reverse_inlined_ = false;
    group_id_ = -1;
    kind_ = kOpaque;
  }

  bool operator>(const TaggedNode& other) const { return id_ > other.id_; }

  const ExprNode* node_;
  int id_, group_id_;
  bool visited_, inlined_, reverse_inlined_;
  Shape shape_;
  OpPatternKind kind_;
  String op_type_;
  std::vector<TaggedNode*> out_edges_, in_edges_;
};

struct TaggedNodeGraph {
  std::vector<TaggedNode*> nodes;
  std::unordered_map<const ExprNode*, TaggedNode*> node_map;
  size_t num_group_ = 0;
  std::unordered_set<String> skip_ops;

  void RunFuse() {
    // phase 1: Fuse from complex Ops
    for (auto& node : nodes) {
      if (is_inlinable(node) || node->visited_ || node->op_type_ == "Tensor") continue;
      fuse_from_node(node);
    }
    // phase 2: Fuse from Sinple Ops
    for (auto& node : nodes) {
      node->visited_ = false;
      node->inlined_ = false;
      node->reverse_inlined_ = false;
    }

    update_inline_nodes();
    // LOG(INFO) << "after update inline nodes";
    for (auto& node : nodes) {
      // LOG(INFO) << "node " << node->op_type_ << " " << node->kind_ << " " << node->inlined_ << " " << node->reverse_inlined_ << " group_id " << node->group_id_;
      if (node->op_type_ == "Tensor" || node->visited_)
        continue;
      else if (node->group_id_ >= 0) {  // already fused
        node->visited_ = true;
        update_inline_nodes();
      } else if (node->inlined_ && node->out_edges_.size() == 1 &&
                 node->out_edges_[0]->group_id_ == -1) {
        // Process these nodes in the next phase
        continue;
      } else {
        fuse_from_node(node);
      }
    }
    // LOG(INFO) << "after update inline nodes stage 2";
    // for (auto& node : nodes) {
    //   LOG(INFO) << "node " << node->op_type_ << " " << node->kind_ << " " << node->inlined_ << " "
    //              << node->reverse_inlined_ << " group_id " << node->group_id_;
    // }
    // phase 3: handle reserve inlined nodes
    for (auto& node : nodes) {
      if (node->reverse_inlined_) {
        // assert the node has only one output
        // CHECK_EQ(node->out_edges_.size(), 1);
        auto out_node = node->out_edges_[0];
        // set the group id of the input node to the group id of the output node
        node->group_id_ = out_node->group_id_;
      }
    }
    // LOG(INFO) << "after handle reserve inlined nodes";
    // for (auto& node : nodes) {
    //   LOG(INFO) << "node " << node->op_type_ << " " << node->kind_ << " " << node->inlined_ << " "
    //             << node->reverse_inlined_ << " group_id " << node->group_id_;
    // }
    // phase 4: Fuse inline ops
    inline_lightweighted_ops(); 
    // phase 5: handle non-group nodes
    // LOG(INFO) << "after inline lightweighted ops";
    // for (auto& node : nodes) {
    //   LOG(INFO) << "node " << node->op_type_ << " " << node->kind_ << " " << node->inlined_ << " "
    //             << node->reverse_inlined_ << " group_id " << node->group_id_;
    // }
  }

  bool is_inlinable(const TaggedNode* node) {
    return node->kind_ <= kInjective;
  }

  void update_inline_nodes() {
    for (auto& node : nodes) {
      if (node->inlined_ || node->visited_) continue;
      if (is_inlinable(node)) {
        node->inlined_ = true;
        for (auto src_node : node->in_edges_) {
          if (!((src_node->inlined_ && src_node->out_edges_.size() == 1) || src_node->visited_)) {
            node->inlined_ = false;
            if (src_node->out_edges_.size() > 1 && node->in_edges_.size() == 1) {
              // dst node is not ladder_perfect
              auto dst_node_type = node->out_edges_[0]->op_type_;
              if (dst_node_type == "ladder.perfect_im2col_conv" ||
                  dst_node_type == "ladder.C2DImplicitGemm" ||
                  dst_node_type == "welder.C2DImplicitGemm") {
                node->reverse_inlined_ = false;
              } else
                node->reverse_inlined_ = true;
            }
            break;
          }
        }
      }
    }
  }

  void fuse_from_node(TaggedNode* top_node) {
    std::unordered_set<const TaggedNode*> block_list;
    auto cmp = [](const TaggedNode* a, const TaggedNode* b) { return a->id_ > b->id_; };
    std::priority_queue<TaggedNode*, std::vector<TaggedNode*>, decltype(cmp)> queue(cmp);
    if (top_node->reverse_inlined_) return;
    top_node->group_id_ = num_group_++;
    queue.push(top_node);

    while (!queue.empty()) {
      auto tnode = queue.top();
      queue.pop();
      if (block_list.count(tnode)) continue;

      // check fusible
      bool fusible = true;
      if (tnode != top_node) {
        fusible &= tnode->group_id_ == -1;
        fusible &= !tnode->visited_;
        fusible &= tnode->inlined_;
        fusible &= StructuralEqual()(tnode->shape_, top_node->shape_);
        fusible &= (!skip_ops.count(tnode->op_type_) && !skip_ops.count(top_node->op_type_));
        fusible &= (tnode->kind_ != kOpaque && top_node->kind_ != kOpaque);
        fusible &= !(top_node->kind_ == kOutEWiseFusable && tnode->kind_ > kBroadcast);
        fusible &= (!tnode->reverse_inlined_);
      }

      // add to group
      if (fusible) {
        tnode->group_id_ = top_node->group_id_;
        for (auto node : tnode->out_edges_) {
          queue.push(node);
        }
        tnode->visited_ = true;
        if (is_inlinable(tnode)) fuse_inline_dependent_nodes(tnode);
        update_inline_nodes();
      } else {
        update_block_list(block_list, tnode);
      }
    }
  }

  void fuse_inline_dependent_nodes(const TaggedNode* tnode) {
    for (auto in_node : tnode->in_edges_) {
      if (in_node->visited_) continue;
      if (!in_node->inlined_) continue;
      if (in_node->reverse_inlined_) continue;
      CHECK(in_node->inlined_);
      in_node->group_id_ = tnode->group_id_;
      in_node->visited_ = true;
      fuse_inline_dependent_nodes(in_node);
    }
  }

  void update_block_list(std::unordered_set<const TaggedNode*>& block_list, const TaggedNode* tnode) {
    block_list.insert(tnode);
    for (auto out_node : tnode->out_edges_) {
      if (block_list.count(out_node) == 0)
        update_block_list(block_list, out_node);
    }
  }

  void Dump() {
    std::unordered_map<int, std::unordered_set<TaggedNode*>> table;
    for (auto node : nodes) {
      if (node->group_id_ >= 0) {
        if (!table.count(node->group_id_)) table[node->group_id_] = {};
        table[node->group_id_].insert(node);
      } else {

      }
    }
    std::cout << table.size() << std::endl;
    for (auto pair : table) {
      for (auto node : pair.second) {
        std::string name = node->op_type_ + std::to_string(node->id_);
        std::cout << name << ", ";
      }
      std::cout << std::endl;
    }
  }

  void inline_lightweighted_ops() {
    // Iterate over all independent groups
    // inline first group into second if:
    // 1. first group has one output
    // 2. first group are all light weighted ops
    // 3. all ops not in skip lists
    using NodeGroup = std::unordered_set<TaggedNode*>;
    std::unordered_map<int, std::shared_ptr<NodeGroup>> map;
    std::vector<std::shared_ptr<NodeGroup>> groups;
    for (auto node : nodes) {
      if (node->group_id_ < 0) node->group_id_ = num_group_++;
      if (!map.count(node->group_id_)) {
        map[node->group_id_] = std::make_shared<NodeGroup>();
      }
      map[node->group_id_]->insert(node);
    }
    for (auto& kv : map) groups.push_back(kv.second);

    for (auto group : groups) {
      bool group_is_lightweighted = true;
      NodeGroup group_outputs;
      for (auto node : *group) {
        group_is_lightweighted &= is_lightweighted_op(node);
        for (auto out_node : node->out_edges_)
          if (!group->count(out_node)) group_outputs.insert(out_node);
      }
      if (group_outputs.size() == 0) continue;  
      auto output_node = *group_outputs.begin();
      bool op_skip = skip_ops.count(output_node->op_type_) || output_node->kind_ == kOpaque;
      for (auto node : *group) {
        op_skip |= skip_ops.count(node->op_type_);
        op_skip |= node->kind_ == kOpaque;
      }

      if (group_is_lightweighted && !op_skip && group_outputs.size() == 1) {
        for (auto node : *group)
          node->group_id_ = output_node->group_id_;
      }
    }
  }

  void CleanUp() {
    for (auto node : nodes) {
      delete node;
    }
  }
  bool is_lightweighted_op(const TaggedNode* node) {
    auto type = node->op_type_;
    if (type == "strided_slice" || type == "Scalar" ||
        type == "expand_dims" || type == "squeeze") return true;
    if (type == "transpose") {
      Expr expr = GetRef<Expr>(node->node_);
      if (auto attrs = expr.as<CallNode>()->attrs.as<TransposeAttrs>()) {
        bool is_lower_dim_kept = attrs->axes.back().IntValue() == int(attrs->axes.size()) -1;
        return is_lower_dim_kept;
      }
    }
    return false;
  }
};

// Quickly check special properties of the fused function.
// A pass to check if the fused op contains only reshape ops.
class CheckReshapeOnly : public ExprVisitor {
  public:
  void VisitExpr_(const CallNode* cn) final {
    this->has_call = true;
    static auto freshape_op = Op::GetAttrMap<TReshapeOp>("TReshapeOp");

    if (!freshape_op.get(cn->op, false)) {
      this->reshape_only = false;
    }

    if (!this->reshape_only) return;
    ExprVisitor::VisitExpr_(cn);
  }

  void VisitExpr_(const VarNode* vn) final {
    if (!vn->type_annotation.defined() || !vn->type_annotation->IsInstance<TensorTypeNode>()) {
      this->reshape_only = false;
    }
  }

  bool reshape_only = true;
  bool has_call = false;
};

/*! \brief Temporary information from each group. */
struct GroupInfo {
public:
  // The parameters of the function.
  Array<Var> params;
  // The arguments to call the functions.
  Array<Expr> arguments;
  // Get a new parameter or allocate an old one
  Var GetOrAllocParam(const Expr& expr, const Type& type) {
    // run linear scan as most fused groups contain only a few inputs.
    for (size_t i = 0; i < arguments.size(); ++i) {
      if (expr.same_as(arguments[i])) return params[i];
    }
    // create a new parameter.
    std::ostringstream os;
    os << "p" << params.size();
    auto var = Var(os.str(), type);
    params.push_back(var);
    arguments.push_back(expr);
    return var;
  }
};

class WelderFuseMutator : private ExprMutator {
public:
  WelderFuseMutator() {}
  // Run the transform
  Expr Transform(const Function& body);
private:
  class TopoCreator;
  TaggedNodeGraph g_;
  Expr VisitExpr_(const CallNode* call) final;
  Expr VisitExpr_(const TupleNode* tuple) final;
  Expr VisitExpr_(const TupleGetItemNode* tuple) final;
  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args, int group_id);
  std::unordered_map<int, GroupInfo> ginfo_;
};

class WelderFuseMutator::TopoCreator : private ExprVisitor {
public:
  TopoCreator() {}
  TaggedNodeGraph Prepare(const Function& func) {
    this->VisitExpr(func->body);
    this->GetFunctionGlobalOutputs(func);
    return std::move(this->g_);
  }
private:
  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    auto node = new TaggedNode(call, g_.nodes.size());
    g_.nodes.push_back(node);
    g_.node_map[call] = node;

    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto op = GetRef<Op>(opnode);
      if (IsDynamic(call->checked_type()) && IsDataDependent(call)) {
        // output of a shape func can't be fed to a data-dependent shape func
        node->kind_ = kOpaque;
      } else {
        node->kind_ = static_cast<OpPatternKind>(fpattern[op]);
      }
    } else {
      node->kind_ = kOpaque;
    }
    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    node->shape_ = rtype->shape;
    if (call->op.as<OpNode>()) {
      node->op_type_ = call->op.as<OpNode>()->name;
    } else {
      node->op_type_ = "Opaque";
    }

    for (auto arg: call->args) {
      if (auto tup = arg.as<TupleNode>()) {
        for (auto item: tup->fields) {
          if (g_.node_map.count(item.get())) {
            auto src_node = g_.node_map[item.get()];
            src_node->out_edges_.push_back(node);
            node->in_edges_.push_back(src_node);
          }
        }
        if (g_.node_map.count(arg.get())) {
          CHECK(g_.node_map[arg.get()] == node);
        } else {
          g_.node_map[arg.get()] = node;
        }
      } else if (g_.node_map.count(arg.get())) {
        auto src_node = g_.node_map[arg.get()];
        src_node->out_edges_.push_back(node);
        node->in_edges_.push_back(src_node);
      }
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
    if (op->is_scalar()) {
      auto node = new TaggedNode(op, g_.nodes.size());
      g_.nodes.push_back(node);
      g_.node_map[op] = node;
      node->kind_ = kElemWise;
      const auto* rtype = op->checked_type().as<TensorTypeNode>();
      ICHECK(rtype);
      node->shape_ = rtype->shape;
      node->op_type_ = "Constant";
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      return;
    }
  }

  void VisitExpr_(const FunctionNode* op) final {
    // treat as opaque call node,do not recurse
    return;
  }

  void VisitExpr_(const LetNode* op) final {
    ICHECK(0) << "Not Implemented.";
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ICHECK(0) << "Not Implemented.";
  }

  void GetFunctionGlobalOutputs(const Function& func) {
    std::unordered_set<TaggedNode*> global_outputs;
    std::queue<Expr> queue;
    queue.push(func->body);
    while (!queue.empty()) {
      auto expr = queue.front();
      queue.pop();
      if (auto tup = expr.as<TupleNode>()) {
        for (auto field : tup->fields) queue.push(field);
      } else if (auto en = expr.as<ExprNode>()) {
        if (g_.node_map.count(en)) {
          auto output_node = new TaggedNode(nullptr, g_.nodes.size());
          g_.nodes.push_back(output_node);
          output_node->kind_ = kOpaque;
          output_node->op_type_ = "Output";
          output_node->in_edges_.push_back(g_.node_map[en]);
          g_.node_map[en]->out_edges_.push_back(output_node);
        }
      }
    }
  }

  TaggedNodeGraph g_;
};

Expr WelderFuseMutator::Transform(const Function& body) {
  this->g_ = TopoCreator().Prepare(body);
  this->g_.RunFuse();
  this->g_.Dump();
  // collect nodes in each group
  std::unordered_map<int, std::vector<TaggedNode*>> gnodes;
  for (auto node : this->g_.nodes) {
    if (!this->ginfo_.count(node->group_id_)) {
      this->ginfo_[node->group_id_] = GroupInfo();
      gnodes[node->group_id_] = {};
    }
    gnodes[node->group_id_].push_back(node);
  }
  // We cannot visit the body with DFS order
  // We need to visit the fusion group one by one
  // Now prepare the fuse group topo order
  std::vector<int> group_order;
  std::unordered_map<int, int> dep_cnt;
  for (auto node: this->g_.nodes) {
    if (!dep_cnt.count(node->group_id_)) dep_cnt[node->group_id_] = 0;
    for (auto in_node: node->in_edges_) {
      if (node->group_id_ != in_node->group_id_) dep_cnt[node->group_id_]++;
    }
  }
  std::stack<int> stack;
  for (auto kv: dep_cnt) {
    if (kv.second == 0) stack.push(kv.first);
  }
  while (!stack.empty()) {
    int group_id = stack.top();
    stack.pop();
    group_order.push_back(group_id);
    for (auto node: gnodes[group_id]) {
      for (auto out_node: node->out_edges_) {
        if (out_node->group_id_ == node->group_id_) continue;
        CHECK(dep_cnt[out_node->group_id_] > 0);
        if (--dep_cnt[out_node->group_id_] == 0) stack.push(out_node->group_id_);
      }
    }
  }
  CHECK(group_order.size() == dep_cnt.size()) << group_order.size() << " " << dep_cnt.size();

  for (int group_id : group_order) {
    bool is_opaque = false;
    std::vector<TaggedNode*> outputs;
    for (auto node : gnodes[group_id]) {
      if (node->kind_ == kOpaque) {
        is_opaque = true;
        break;
      }
      this->Mutate(GetRef<Expr>(node->node_));
      bool is_group_output = false;
      for (auto node : node->out_edges_) {
        is_group_output |= node->group_id_ != group_id;
      }
      if (is_group_output) {
        outputs.push_back(node);
      }
    }
    if (is_opaque) continue;
    CHECK(outputs.size() >= 1);
    if (outputs.size() > 1) {
      Array<Expr> fields;
      Array<Type> types;
      for (auto node: outputs) {
        fields.push_back(this->Mutate(GetRef<Expr>(node->node_)));
        types.push_back(node->node_->checked_type());
      }
      auto function = Function(ginfo_[group_id].params, Tuple(fields), TupleType(types), {});
      auto call = Call(function, ginfo_[group_id].arguments, Attrs());
      for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
        Expr field = GetRef<Expr>(outputs[i]->node_);
        this->memo_[field] = TupleGetItem(call, i);
      }
    } else {
      Expr field = GetRef<Expr>(outputs[0]->node_);
      auto function = Function(ginfo_[group_id].params, this->Mutate(field), field->checked_type(), {});
      CheckReshapeOnly checker;
      checker.VisitExpr(this->Mutate(field));
      if (checker.has_call && checker.reshape_only)
        function = WithAttr(std::move(function), attr::kReshapeOnly, tvm::Integer(checker.reshape_only));
      auto call = Call(function, ginfo_[group_id].arguments, Attrs());
      this->memo_[field] = call;
    }
  }
  this->ginfo_.clear();
  this->g_.CleanUp();
  return this->Mutate(body);
}

Expr WelderFuseMutator::VisitExpr_(const CallNode* call) {
  ICHECK(g_.node_map.count(call));
  if (g_.node_map[call]->kind_ == kOpaque) {
      Array<Expr> new_args;
      for (auto arg : call->args) {
        auto new_arg = this->Mutate(arg);
        new_args.push_back(new_arg);
      }
      return Call(call->op, new_args, call->attrs, call->type_args, call->span);
  }
  auto new_args = GetNewArguments(call->args, g_.node_map[call]->group_id_);
  auto new_call = Call(call->op, new_args, call->attrs, call->type_args, call->span);
  return std::move(new_call);
}

Expr WelderFuseMutator::VisitExpr_(const TupleNode* tuple_node) {
  if (!g_.node_map.count(tuple_node)) {
    return ExprMutator::VisitExpr_(tuple_node);
  }
  Array<Expr> new_fields = GetNewArguments(tuple_node->fields, g_.node_map[tuple_node]->group_id_);
  return WithFields(GetRef<Tuple>(tuple_node), new_fields);
}

Expr WelderFuseMutator::VisitExpr_(const TupleGetItemNode* tuple_get) {
  CHECK(0) << "Not implemented.";
  return ExprMutator::VisitExpr_(tuple_get);
}

Array<Expr> WelderFuseMutator::GetNewArguments(const tvm::Array<Expr>& args, int group_id) {
  Array<Expr> new_args;
  for (auto arg : args) {
    auto type = arg->checked_type();
    Expr new_arg = this->Mutate(arg);
    if (g_.node_map.count(arg.get()) && g_.node_map[arg.get()]->group_id_ == group_id) {
      new_args.push_back(new_arg);
    } else {
      auto var = ginfo_[group_id].GetOrAllocParam(new_arg, type);
      new_args.push_back(var);
    }
  }
  return new_args;
}

namespace transform {

Pass WelderFuseOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(WelderFuseMutator().Transform(f));;
    };
  return CreateFunctionPass(pass_func, 0, "WelderFuseOps", { "InferType" });
}

TVM_REGISTER_GLOBAL("relay._transform.WelderFuseOps").set_body_typed(WelderFuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
