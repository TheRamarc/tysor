#include "ops.h"

#include <string_view>
#include <utility>

namespace {

Type tensor_any() {
  return Type::tensor(std::nullopt, std::nullopt, std::nullopt);
}

/**
 * @brief Helper to construct a BuiltinSignature.
 */
BuiltinSignature signature(std::string name, Type returnType,
                           std::vector<Type> argTypes, std::size_t minArity,
                           std::size_t maxArity) {
  return BuiltinSignature{std::move(name), std::move(returnType),
                          std::move(argTypes), minArity, maxArity};
}

/**
 * @brief Helper to construct an OpDefinition.
 */
OpDefinition op(OpId id, OpCategory category, BuiltinSignature sig,
                bool primitive, bool library, bool callable,
                bool preserves_first_arg, bool runtime_library,
                bool plan_supported,
                RuntimePrimitiveKind runtime_primitive_kind =
                    RuntimePrimitiveKind::Unsupported) {
  return OpDefinition{
      id,
      category,
      std::move(sig),
      primitive,
      library,
      callable,
      preserves_first_arg,
      runtime_library,
      runtime_primitive_kind,
      plan_supported,
  };
}

/**
 * @brief Returns the expected argument types for the reshape operation.
 *
 * Reshape accepts a tensor and up to 7 integer dimension sizes.
 */
std::vector<Type> reshape_arg_types() {
  return {
      tensor_any(),    Type::intType(), Type::intType(), Type::intType(),
      Type::intType(), Type::intType(), Type::intType(), Type::intType(),
  };
}

/**
 * @brief Defines the table of all available built-in operations.
 *
 * Includes layers (like Linear, Embedding), activation functions, tensor ops
 * (Reshape, Sum), and utilities (Print). Each definition specifies its typing
 * and lowering properties.
 */
std::vector<OpDefinition> make_ops() {
  return {
      op(OpId::Linear, OpCategory::LayerCtor,
         signature("linear", Type::callable(tensor_any()),
                   {Type::intType(), Type::intType(), Type::boolType()}, 1, 3),
         false, true, true, false, true, true),
      op(OpId::Tensor, OpCategory::LayerCtor,
         signature("tensor", tensor_any(),
                   {Type::intType(), Type::intType(), Type::boolType()}, 1, 3),
         false, true, true, false, true, false),
      op(OpId::Matmul, OpCategory::Primitive,
         signature("matmul", tensor_any(), {tensor_any(), tensor_any()}, 2, 2),
         true, false, false, false, false, true, RuntimePrimitiveKind::Matmul),
      op(OpId::Relu, OpCategory::Primitive,
         signature("relu", tensor_any(), {tensor_any()}, 1, 1), true, false,
         false, true, false, true, RuntimePrimitiveKind::Relu),
      op(OpId::Scale, OpCategory::Primitive,
         signature("scale", tensor_any(), {tensor_any(), Type::floatType()}, 2,
                   2),
         true, true, false, true, false, true, RuntimePrimitiveKind::Scale),
      op(OpId::Silu, OpCategory::Activation,
         signature("SiLU", Type::callable(tensor_any()), {}, 0, 0), false, true,
         true, false, true, true),
      op(OpId::Gelu, OpCategory::Activation,
         signature("GELU", Type::callable(tensor_any()), {}, 0, 0), false, true,
         true, false, true, true),
      op(OpId::Tanh, OpCategory::Activation,
         signature("Tanh", Type::callable(tensor_any()), {}, 0, 0), false, true,
         true, false, true, true),
      op(OpId::Sigmoid, OpCategory::Activation,
         signature("Sigmoid", Type::callable(tensor_any()), {}, 0, 0), false,
         true, true, false, true, true),
      op(OpId::Softmax, OpCategory::Activation,
         signature("Softmax", Type::callable(tensor_any()), {}, 0, 0), false,
         true, true, false, true, true),
      op(OpId::RmsNorm, OpCategory::Activation,
         signature("RMSNorm", Type::callable(tensor_any()), {}, 0, 0), false,
         true, true, false, true, false),
      op(OpId::RmsNormCall, OpCategory::LibraryOp,
         signature("rms_norm", tensor_any(), {tensor_any(), Type::intType()}, 2,
                   2),
         false, true, false, true, false, true),
      op(OpId::CrossEntropy, OpCategory::LibraryOp,
         signature("cross_entropy", tensor_any(), {tensor_any(), tensor_any()},
                   2, 2),
         false, true, false, false, false, true),
      op(OpId::Embedding, OpCategory::LayerCtor,
         signature("Embedding", Type::callable(tensor_any()),
                   {Type::intType(), Type::intType()}, 2, 2),
         false, true, true, false, true, true),
      op(OpId::Dropout, OpCategory::Activation,
         signature("Dropout", Type::callable(tensor_any()), {Type::floatType()},
                   1, 1),
         false, true, true, false, true, true),
      op(OpId::Rope, OpCategory::LibraryOp,
         signature("rope", tensor_any(),
                   {tensor_any(), Type::intType(), Type::floatType()}, 3, 3),
         false, true, false, true, false, true),
      op(OpId::Reshape, OpCategory::LibraryOp,
         signature("reshape", tensor_any(), reshape_arg_types(), 2, 8), false,
         true, false, false, false, true),
      op(OpId::Transpose, OpCategory::LibraryOp,
         signature("transpose", tensor_any(), {tensor_any()}, 1, 1), false,
         true, false, false, false, true),
      op(OpId::Sum, OpCategory::LibraryOp,
         signature("sum", Type::tensor(std::nullopt, std::nullopt, 1),
                   {tensor_any(), Type::intType()}, 1, 2),
         false, true, false, false, false, true),
      op(OpId::Mean, OpCategory::LibraryOp,
         signature("mean", Type::tensor(std::nullopt, std::nullopt, 1),
                   {tensor_any(), Type::intType()}, 1, 2),
         false, true, false, false, false, true),
      op(OpId::Sqrt, OpCategory::LibraryOp,
         signature("sqrt", tensor_any(), {tensor_any()}, 1, 1), false, true,
         false, true, false, true),
      op(OpId::Rsqrt, OpCategory::LibraryOp,
         signature("rsqrt", tensor_any(), {tensor_any()}, 1, 1), false, true,
         false, true, false, true),
      op(OpId::CausalMask, OpCategory::LibraryOp,
         signature("causal_mask", tensor_any(), {tensor_any()}, 1, 1), false,
         true, false, true, false, true),
      op(OpId::FlattenHeads, OpCategory::LibraryOp,
         signature("flatten_heads", tensor_any(), {tensor_any()}, 1, 1), false,
         true, false, true, false, true),
      op(OpId::RepeatKv, OpCategory::LibraryOp,
         signature("repeat_kv", tensor_any(), {tensor_any(), Type::intType()},
                   2, 2),
         false, true, false, true, false, true),
      op(OpId::Print, OpCategory::Utility,
         signature("print", Type::noneType(), {}, 0, 8), true, false, false,
         false, false, true),
  };
}

} // namespace

const OpRegistry &OpRegistry::instance() {
  static const OpRegistry registry;
  return registry;
}

OpRegistry::OpRegistry() : ops_(make_ops()) {
  for (std::size_t idx = 0; idx < ops_.size(); ++idx) {
    nameMap_[ops_[idx].signature.name] = idx;
    idMap_[static_cast<std::size_t>(ops_[idx].id)] = idx;
  }
}

const OpDefinition *OpRegistry::lookup(std::string_view name) const {
  auto found = nameMap_.find(std::string(name));
  if (found == nameMap_.end()) {
    return nullptr;
  }
  return &ops_[found->second];
}

const OpDefinition *OpRegistry::lookup(OpId id) const {
  auto found = idMap_.find(static_cast<std::size_t>(id));
  if (found == idMap_.end()) {
    return nullptr;
  }
  return &ops_[found->second];
}

std::optional<OpId> OpRegistry::lookupId(std::string_view name) const {
  const OpDefinition *def = lookup(name);
  if (def == nullptr) {
    return std::nullopt;
  }
  return def->id;
}

Type OpRegistry::inferResultType(OpId id, const Type &declared_type,
                                 const std::vector<Type> &argTypes) const {
  if (declared_type.base == TypeBase::Callable && !argTypes.empty()) {
    if (declared_type.callableReturn &&
        declared_type.callableReturn->base == TypeBase::Tensor &&
        argTypes[0].base == TypeBase::Tensor) {
      return Type::callable(
          Type::tensor(argTypes[0].tensorDtype,
                       declared_type.callableReturn->tensorShapeExpr,
                       argTypes[0].tensorRank));
    }
    return declared_type;
  }

  if (declared_type.base != TypeBase::Tensor || argTypes.empty() ||
      argTypes[0].base != TypeBase::Tensor) {
    return declared_type;
  }

  const Type &first = argTypes[0];
  if (id == OpId::Reshape) {
    return Type::tensor(first.tensorDtype, std::nullopt, std::nullopt);
  }
  if (id == OpId::Sum || id == OpId::Mean) {
    return Type::tensor(first.tensorDtype, std::nullopt, 1);
  }
  if (id == OpId::Matmul) {
    return Type::tensor(first.tensorDtype, declared_type.tensorShapeExpr,
                        first.tensorRank ? first.tensorRank
                                         : declared_type.tensorRank);
  }
  if (id == OpId::CrossEntropy) {
    return Type::tensor(first.tensorDtype, declared_type.tensorShapeExpr,
                        declared_type.tensorRank);
  }

  const OpDefinition *def = lookup(id);
  if (def != nullptr && def->preservesFirstTensorArg) {
    return first;
  }

  return Type::tensor(first.tensorDtype, declared_type.tensorShapeExpr,
                      first.tensorRank);
}

const std::vector<OpDefinition> &allOps() {
  return OpRegistry::instance().allOps();
}

std::vector<BuiltinSignature> allBuiltinSignatures() {
  std::vector<BuiltinSignature> signatures;
  for (const auto &item : allOps()) {
    signatures.push_back(item.signature);
  }
  return signatures;
}

const OpDefinition *lookupOp(std::string_view name) {
  return OpRegistry::instance().lookup(name);
}

std::optional<OpId> lookupOpId(std::string_view name) {
  return OpRegistry::instance().lookupId(name);
}

const char *opIdName(OpId id) {
  switch (id) {
  case OpId::Linear:
    return "Linear";
  case OpId::Tensor:
    return "Tensor";
  case OpId::Matmul:
    return "Matmul";
  case OpId::MatmulRelu:
    return "MatmulRelu";
  case OpId::Relu:
    return "Relu";
  case OpId::Scale:
    return "Scale";
  case OpId::Silu:
    return "Silu";
  case OpId::Gelu:
    return "Gelu";
  case OpId::Tanh:
    return "Tanh";
  case OpId::Sigmoid:
    return "Sigmoid";
  case OpId::Softmax:
    return "Softmax";
  case OpId::RmsNorm:
    return "RmsNorm";
  case OpId::RmsNormCall:
    return "RmsNormCall";
  case OpId::CrossEntropy:
    return "CrossEntropy";
  case OpId::Embedding:
    return "Embedding";
  case OpId::Dropout:
    return "Dropout";
  case OpId::Rope:
    return "Rope";
  case OpId::Reshape:
    return "Reshape";
  case OpId::Transpose:
    return "Transpose";
  case OpId::Sum:
    return "Sum";
  case OpId::Mean:
    return "Mean";
  case OpId::Sqrt:
    return "Sqrt";
  case OpId::Rsqrt:
    return "Rsqrt";
  case OpId::CausalMask:
    return "CausalMask";
  case OpId::FlattenHeads:
    return "FlattenHeads";
  case OpId::RepeatKv:
    return "RepeatKv";
  case OpId::Print:
    return "Print";
  }
  return "Unknown";
}

bool identifiesOp(std::string_view name, OpId id) {
  const std::optional<OpId> resolved = lookupOpId(name);
  return resolved && *resolved == id;
}

bool identifiesOp(std::string_view name, const std::optional<std::string> &opId,
                  OpId id) {
  if (identifiesOp(name, id)) {
    return true;
  }
  return opId && *opId == opIdName(id);
}

bool isPlanActivation(std::string_view name) {
  const OpDefinition *def = lookupOp(name);
  return def != nullptr && def->planSupported &&
         def->category == OpCategory::Activation;
}

namespace {

std::vector<std::string> collectPlanOps(bool wantPrimitive, bool wantLibrary,
                                        bool wantConstructor, bool ids) {
  std::vector<std::string> names;
  for (const OpDefinition &def : allOps()) {
    if (!def.planSupported) {
      continue;
    }
    const bool primitive = def.isPrimitiveTensorOp;
    const bool constructor = def.isCallableLibraryOp;
    const bool library = def.isLibraryOp && !primitive && !constructor;
    const bool match = (wantPrimitive && primitive) ||
                       (wantLibrary && library) ||
                       (wantConstructor && constructor);
    if (!match) {
      continue;
    }
    names.push_back(ids ? opIdName(def.id) : def.signature.name);
  }
  return names;
}

} // namespace

const std::vector<std::string> &planSupportedPrimitiveOps() {
  static const std::vector<std::string> names =
      collectPlanOps(true, false, false, false);
  return names;
}

const std::vector<std::string> &planSupportedPrimitiveOpIds() {
  static const std::vector<std::string> names =
      collectPlanOps(true, false, false, true);
  return names;
}

const std::vector<std::string> &planSupportedLibraryOps() {
  static const std::vector<std::string> names =
      collectPlanOps(false, true, false, false);
  return names;
}

const std::vector<std::string> &planSupportedLibraryOpIds() {
  static const std::vector<std::string> names =
      collectPlanOps(false, true, false, true);
  return names;
}

const std::vector<std::string> &planSupportedConstructorOps() {
  static const std::vector<std::string> names =
      collectPlanOps(false, false, true, false);
  return names;
}

const std::vector<std::string> &planSupportedConstructorOpIds() {
  static const std::vector<std::string> names =
      collectPlanOps(false, false, true, true);
  return names;
}

bool isBuiltinOp(std::string_view name) { return lookupOp(name) != nullptr; }

bool isPrimitiveTensorOp(std::string_view name) {
  const OpDefinition *item = lookupOp(name);
  return item != nullptr && item->isPrimitiveTensorOp;
}

bool isLibraryOp(std::string_view name) {
  const OpDefinition *item = lookupOp(name);
  return item != nullptr && item->isLibraryOp;
}

bool isCallableLibraryOp(std::string_view name) {
  const OpDefinition *item = lookupOp(name);
  return item != nullptr && item->isCallableLibraryOp;
}

bool preservesFirstTensorArg(std::string_view name) {
  const OpDefinition *item = lookupOp(name);
  return item != nullptr && item->preservesFirstTensorArg;
}

bool runtimeSupportsLibraryOp(std::string_view name) {
  const OpDefinition *item = lookupOp(name);
  return item != nullptr && item->runtimeSupportsLibraryOp;
}

RuntimePrimitiveKind runtimePrimitive(std::string_view name) {
  const OpDefinition *item = lookupOp(name);
  return item == nullptr ? RuntimePrimitiveKind::Unsupported
                         : item->runtimePrimitiveKind;
}
