#pragma once

#include "parser.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

enum class OpId {
    Linear,
    Matmul,
    MatmulRelu,
    Relu,
    Scale,
    Silu,
    Gelu,
    Tanh,
    Sigmoid,
    Softmax,
    RmsNorm,
    CrossEntropy,
    Embedding,
    Dropout,
    Rope,
    Reshape,
    Transpose,
    Sum,
    Mean,
    Sqrt,
    Rsqrt,
    CausalMask,
    FlattenHeads,
    RepeatKv,
    Print,
    Tensor
};

enum class RuntimePrimitiveKind {
    Unsupported,
    Matmul,
    Relu,
    Scale,
};

struct BuiltinSignature {
    std::string name;
    Type return_type;
    std::vector<Type> arg_types;
    std::size_t min_arity = 0;
    std::size_t max_arity = 0;
};

struct OpDefinition {
    OpId id = OpId::Print;
    BuiltinSignature signature;
    bool is_primitive_tensor_op = false; // this field is used for the op lowers to a primitive graph or not.
    bool is_library_op = false; // this field is used for considered a builtin/library-level operations.
    bool is_callable_library_op = false;
    bool preserves_first_tensor_arg = false; //the result keeps the general tensor type/shape from the first tensor argument.
    bool runtime_supports_library_op = false; 
    RuntimePrimitiveKind runtime_primitive_kind = RuntimePrimitiveKind::Unsupported;
};

const std::vector<OpDefinition>& all_ops();
std::vector<BuiltinSignature> all_builtin_signatures();
const OpDefinition* lookup_op(std::string_view name);
std::optional<OpId> lookup_op_id(std::string_view name);
const char* op_id_name(OpId id);
bool is_builtin_op(std::string_view name);
bool is_primitive_tensor_op(std::string_view name);
bool is_library_op(std::string_view name);
bool is_callable_library_op(std::string_view name);
bool preserves_first_tensor_arg(std::string_view name);
bool runtime_supports_library_op(std::string_view name);
RuntimePrimitiveKind runtime_primitive(std::string_view name);
