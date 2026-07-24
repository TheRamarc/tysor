#include "ops.h"

#include <algorithm>
#include <string_view>
#include <utility>

namespace {

Type tensor_any() {
    return Type::tensor(std::nullopt, std::nullopt, std::nullopt);
}

/**
 * @brief Helper to construct a BuiltinSignature.
 */
BuiltinSignature signature(
    std::string name,
    Type returnType,
    std::vector<Type> argTypes,
    std::size_t minArity,
    std::size_t maxArity
) {
    return BuiltinSignature{std::move(name), std::move(returnType), std::move(argTypes), minArity, maxArity};
}

/**
 * @brief Helper to construct an OpDefinition.
 */
OpDefinition op(
    OpId id,
    BuiltinSignature sig,
    bool primitive,
    bool library,
    bool callable,
    bool preserves_first_arg,
    bool runtime_library,
    RuntimePrimitiveKind runtime_primitive_kind = RuntimePrimitiveKind::Unsupported
) {
    return OpDefinition{
        id,
        std::move(sig),
        primitive,
        library,
        callable,
        preserves_first_arg,
        runtime_library,
        runtime_primitive_kind,
    };
}

/**
 * @brief Returns the expected argument types for the reshape operation.
 * 
 * Reshape accepts a tensor and up to 7 integer dimension sizes.
 */
std::vector<Type> reshape_arg_types() {
    return {
        tensor_any(),
        Type::intType(),
        Type::intType(),
        Type::intType(),
        Type::intType(),
        Type::intType(),
        Type::intType(),
        Type::intType(),
    };
}

/**
 * @brief Defines the table of all available built-in operations.
 * 
 * Includes layers (like Linear, Embedding), activation functions, tensor ops (Reshape, Sum),
 * and utilities (Print). Each definition specifies its typing and lowering properties.
 */
std::vector<OpDefinition> make_ops() {
    return {
        op(OpId::Linear, signature("linear", Type::callable(tensor_any()), {Type::intType(), Type::intType(), Type::boolType()}, 1, 3), false, true, true, false, true),
        op(OpId::Tensor, signature("tensor", tensor_any(), {Type::intType(), Type::intType(), Type::boolType()}, 1, 3), false, true, true, false, true),
        op(OpId::Matmul, signature("matmul", tensor_any(), {tensor_any(), tensor_any()}, 2, 2), true, false, false, false, false, RuntimePrimitiveKind::Matmul),
        op(OpId::Relu, signature("relu", tensor_any(), {tensor_any()}, 1, 1), true, false, false, true, false, RuntimePrimitiveKind::Relu),
        op(OpId::Scale, signature("scale", tensor_any(), {tensor_any(), Type::floatType()}, 2, 2), true, true, false, true, false, RuntimePrimitiveKind::Scale),
        op(OpId::Silu, signature("SiLU", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Gelu, signature("GELU", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Tanh, signature("Tanh", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Sigmoid, signature("Sigmoid", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Softmax, signature("Softmax", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::RmsNorm, signature("RMSNorm", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::RmsNorm, signature("rms_norm", tensor_any(), {tensor_any(), Type::intType()}, 2, 2), false, true, false, true, false),
        op(OpId::CrossEntropy, signature("cross_entropy", tensor_any(), {tensor_any(), tensor_any()}, 2, 2), false, true, false, false, false),
        op(OpId::Embedding, signature("Embedding", Type::callable(tensor_any()), {Type::intType(), Type::intType()}, 2, 2), false, true, true, false, true),
        op(OpId::Dropout, signature("Dropout", Type::callable(tensor_any()), {Type::floatType()}, 1, 1), false, true, true, false, true),
        op(OpId::Rope, signature("rope", tensor_any(), {tensor_any(), Type::intType(), Type::floatType()}, 3, 3), false, true, false, true, false),
        op(OpId::Reshape, signature("reshape", tensor_any(), reshape_arg_types(), 2, 8), false, true, false, false, false),
        op(OpId::Transpose, signature("transpose", tensor_any(), {tensor_any()}, 1, 1), false, true, false, false, false),
        op(OpId::Sum, signature("sum", Type::tensor(std::nullopt, std::nullopt, 1), {tensor_any(), Type::intType()}, 1, 2), false, true, false, false, false),
        op(OpId::Mean, signature("mean", Type::tensor(std::nullopt, std::nullopt, 1), {tensor_any(), Type::intType()}, 1, 2), false, true, false, false, false),
        op(OpId::Sqrt, signature("sqrt", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::Rsqrt, signature("rsqrt", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::CausalMask, signature("causal_mask", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::FlattenHeads, signature("flatten_heads", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::RepeatKv, signature("repeat_kv", tensor_any(), {tensor_any(), Type::intType()}, 2, 2), false, true, false, true, false),
        op(OpId::Print, signature("print", Type::noneType(), {}, 0, 1), false, false, false, false, false),
    };
}

} // namespace

const std::vector<OpDefinition>& allOps() {
    static const std::vector<OpDefinition> ops = make_ops();
    return ops;
}

std::vector<BuiltinSignature> allBuiltinSignatures() {
    std::vector<BuiltinSignature> signatures;
    for (const auto& item : allOps()) {
        signatures.push_back(item.signature);
    }
    return signatures;
}

const OpDefinition* lookupOp(std::string_view name) {
    const auto& ops = allOps();
    auto found = std::find_if(ops.begin(), ops.end(), [&](const OpDefinition& item) {
        return std::string_view(item.signature.name) == name;
    });
    return found == ops.end() ? nullptr : &*found;
}

std::optional<OpId> lookupOpId(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    if (item == nullptr) {
        return std::nullopt;
    }
    return item->id;
}

const char* opIdName(OpId id) {
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

bool isBuiltinOp(std::string_view name) {
    return lookupOp(name) != nullptr;
}

bool isPrimitiveTensorOp(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    return item != nullptr && item->isPrimitiveTensorOp;
}

bool isLibraryOp(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    return item != nullptr && item->isLibraryOp;
}

bool isCallableLibraryOp(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    return item != nullptr && item->isCallableLibraryOp;
}

bool preservesFirstTensorArg(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    return item != nullptr && item->preservesFirstTensorArg;
}

bool runtimeSupportsLibraryOp(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    return item != nullptr && item->runtimeSupportsLibraryOp;
}

RuntimePrimitiveKind runtimePrimitive(std::string_view name) {
    const OpDefinition* item = lookupOp(name);
    return item == nullptr ? RuntimePrimitiveKind::Unsupported : item->runtimePrimitiveKind;
}
