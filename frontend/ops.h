#pragma once

#include "parser.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Unique identifiers for built-in operations (both layers and primitive ops).
 */
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

/**
 * @brief Identifies whether a backend runtime natively supports a specific primitive.
 */
enum class RuntimePrimitiveKind {
    Unsupported,
    Matmul,
    Relu,
    Scale,
};

/**
 * @brief Defines the type signature and arity requirements for a built-in operation.
 */
struct BuiltinSignature {
    std::string name;
    Type return_type;
    std::vector<Type> arg_types;
    std::size_t min_arity = 0;
    std::size_t max_arity = 0;
};

/**
 * @brief Defines a built-in operation, its signature, and its lowering properties.
 */
struct OpDefinition {
    OpId id = OpId::Print;
    BuiltinSignature signature;
    /// If true, this op lowers directly to a primitive graph node.
    bool is_primitive_tensor_op = false; 
    /// If true, this is considered a library-level operation (often implemented as a composite or runtime call).
    bool is_library_op = false; 
    /// If true, this library op returns a callable that must be invoked with an input tensor.
    bool is_callable_library_op = false;
    /// If true, the result keeps the general tensor type/shape from the first tensor argument.
    bool preserves_first_tensor_arg = false; 
    /// If true, the underlying runtime natively supports this library op directly.
    bool runtime_supports_library_op = false; 
    /// Specifies the primitive kind if natively supported.
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
