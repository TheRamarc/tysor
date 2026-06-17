#include "ops.h"

#include <algorithm>
#include <string_view>
#include <utility>

namespace {

Type tensor_any() {
    return Type::tensor(std::nullopt, std::nullopt, std::nullopt);
}

BuiltinSignature signature(
    std::string name,
    Type return_type,
    std::vector<Type> arg_types,
    std::size_t min_arity,
    std::size_t max_arity
) {
    return BuiltinSignature{std::move(name), std::move(return_type), std::move(arg_types), min_arity, max_arity};
}

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

std::vector<Type> reshape_arg_types() {
    return {
        tensor_any(),
        Type::int_type(),
        Type::int_type(),
        Type::int_type(),
        Type::int_type(),
        Type::int_type(),
        Type::int_type(),
        Type::int_type(),
    };
}

std::vector<OpDefinition> make_ops() {
    return {
        op(OpId::Linear, signature("linear", Type::callable(tensor_any()), {Type::int_type(), Type::int_type(), Type::bool_type()}, 1, 3), false, true, true, false, true),
        op(OpId::Matmul, signature("matmul", tensor_any(), {tensor_any(), tensor_any()}, 2, 2), true, false, false, false, false, RuntimePrimitiveKind::Matmul),
        op(OpId::Relu, signature("relu", tensor_any(), {tensor_any()}, 1, 1), true, false, false, true, false, RuntimePrimitiveKind::Relu),
        op(OpId::Scale, signature("scale", tensor_any(), {tensor_any(), Type::float_type()}, 2, 2), true, true, false, true, false, RuntimePrimitiveKind::Scale),
        op(OpId::Silu, signature("SiLU", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Gelu, signature("GELU", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Tanh, signature("Tanh", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Sigmoid, signature("Sigmoid", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::Softmax, signature("Softmax", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::RmsNorm, signature("RMSNorm", Type::callable(tensor_any()), {}, 0, 0), false, true, true, false, true),
        op(OpId::RmsNorm, signature("rms_norm", tensor_any(), {tensor_any(), Type::int_type()}, 2, 2), false, true, false, true, false),
        op(OpId::CrossEntropy, signature("cross_entropy", tensor_any(), {tensor_any(), tensor_any()}, 2, 2), false, true, false, false, false),
        op(OpId::Embedding, signature("Embedding", Type::callable(tensor_any()), {Type::int_type(), Type::int_type()}, 2, 2), false, true, true, false, true),
        op(OpId::Dropout, signature("Dropout", Type::callable(tensor_any()), {Type::float_type()}, 1, 1), false, true, true, false, true),
        op(OpId::Rope, signature("rope", tensor_any(), {tensor_any(), Type::int_type(), Type::float_type()}, 3, 3), false, true, false, true, false),
        op(OpId::Reshape, signature("reshape", tensor_any(), reshape_arg_types(), 2, 8), false, true, false, false, false),
        op(OpId::Transpose, signature("transpose", tensor_any(), {tensor_any()}, 1, 1), false, true, false, false, false),
        op(OpId::Sum, signature("sum", Type::tensor(std::nullopt, std::nullopt, 1), {tensor_any(), Type::int_type()}, 1, 2), false, true, false, false, false),
        op(OpId::Mean, signature("mean", Type::tensor(std::nullopt, std::nullopt, 1), {tensor_any(), Type::int_type()}, 1, 2), false, true, false, false, false),
        op(OpId::Sqrt, signature("sqrt", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::Rsqrt, signature("rsqrt", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::CausalMask, signature("causal_mask", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::FlattenHeads, signature("flatten_heads", tensor_any(), {tensor_any()}, 1, 1), false, true, false, true, false),
        op(OpId::RepeatKv, signature("repeat_kv", tensor_any(), {tensor_any(), Type::int_type()}, 2, 2), false, true, false, true, false),
        op(OpId::Print, signature("print", Type::void_type(), {}, 0, 1), false, false, false, false, false),
    };
}

} // namespace

const std::vector<OpDefinition>& all_ops() {
    static const std::vector<OpDefinition> ops = make_ops();
    return ops;
}

std::vector<BuiltinSignature> all_builtin_signatures() {
    std::vector<BuiltinSignature> signatures;
    for (const auto& item : all_ops()) {
        signatures.push_back(item.signature);
    }
    return signatures;
}

const OpDefinition* lookup_op(std::string_view name) {
    const auto& ops = all_ops();
    auto found = std::find_if(ops.begin(), ops.end(), [&](const OpDefinition& item) {
        return std::string_view(item.signature.name) == name;
    });
    return found == ops.end() ? nullptr : &*found;
}

std::optional<OpId> lookup_op_id(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    if (item == nullptr) {
        return std::nullopt;
    }
    return item->id;
}

const char* op_id_name(OpId id) {
    switch (id) {
        case OpId::Linear:
            return "Linear";
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

bool is_builtin_op(std::string_view name) {
    return lookup_op(name) != nullptr;
}

bool is_primitive_tensor_op(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    return item != nullptr && item->is_primitive_tensor_op;
}

bool is_library_op(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    return item != nullptr && item->is_library_op;
}

bool is_callable_library_op(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    return item != nullptr && item->is_callable_library_op;
}

bool preserves_first_tensor_arg(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    return item != nullptr && item->preserves_first_tensor_arg;
}

bool runtime_supports_library_op(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    return item != nullptr && item->runtime_supports_library_op;
}

RuntimePrimitiveKind runtime_primitive(std::string_view name) {
    const OpDefinition* item = lookup_op(name);
    return item == nullptr ? RuntimePrimitiveKind::Unsupported : item->runtime_primitive_kind;
}
