#pragma once

#include "parser.h"

#include <optional>
#include <string>
#include <vector>

enum class OpId {
    Linear,
    Matmul,
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
    bool is_primitive_tensor_op = false;
    bool is_library_op = false;
    bool is_callable_library_op = false;
    bool preserves_first_tensor_arg = false;
    bool runtime_supports_library_op = false;
    RuntimePrimitiveKind runtime_primitive_kind = RuntimePrimitiveKind::Unsupported;
};

const std::vector<OpDefinition>& all_ops();
std::vector<BuiltinSignature> all_builtin_signatures();
const OpDefinition* lookup_op(const std::string& name);
std::optional<OpId> lookup_op_id(const std::string& name);
const char* op_id_name(OpId id);
bool is_builtin_op(const std::string& name);
bool is_primitive_tensor_op(const std::string& name);
bool is_library_op(const std::string& name);
bool is_callable_library_op(const std::string& name);
bool preserves_first_tensor_arg(const std::string& name);
bool runtime_supports_library_op(const std::string& name);
RuntimePrimitiveKind runtime_primitive(const std::string& name);
