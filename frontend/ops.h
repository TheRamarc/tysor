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
    /**
     * @brief The name of the built-in operation.
     * 
     * Why it exists: Used to identify the operation during parsing and semantic analysis.
     * What it tracks: The string identifier for the operation (e.g., "linear", "relu").
     * What mutates/updates it: Static string populated during definition; immutable at runtime.
     */
    std::string name;

    /**
     * @brief The return type of the operation.
     * 
     * Why it exists: Enables type-checking of operation outputs.
     * What it tracks: The semantic Type returned by the operation.
     * What mutates/updates it: Statically populated during definition.
     */
    Type return_type;

    /**
     * @brief The types of arguments expected by the operation.
     * 
     * Why it exists: To validate function and layer call arguments.
     * What it tracks: A list of expected parameter Types.
     * What mutates/updates it: Statically populated during definition.
     */
    std::vector<Type> arg_types;

    /**
     * @brief The minimum number of arguments required.
     * 
     * Why it exists: Allows operations to have optional arguments.
     * What it tracks: The lowest valid count of arguments.
     * What mutates/updates it: Statically populated during definition.
     */
    std::size_t min_arity = 0;

    /**
     * @brief The maximum number of arguments permitted.
     * 
     * Why it exists: Restricts over-passing arguments to operations.
     * What it tracks: The highest valid count of arguments.
     * What mutates/updates it: Statically populated during definition.
     */
    std::size_t max_arity = 0;
};

/**
 * @brief Defines a built-in operation, its signature, and its lowering properties.
 */
struct OpDefinition {
    /**
     * @brief The unique enum identifier for this operation.
     * 
     * Why it exists: Provides a fast, integer-based way to switch over operations.
     * What it tracks: The OpId corresponding to this definition.
     * What mutates/updates it: Statically populated during definition.
     */
    OpId id = OpId::Print;

    /**
     * @brief The type signature of this operation.
     * 
     * Why it exists: Provides type information to the semantic analyzer.
     * What it tracks: Name, return type, and argument types/arities.
     * What mutates/updates it: Statically populated during definition.
     */
    BuiltinSignature signature;

    /**
     * @brief If true, this op lowers directly to a primitive graph node.
     * 
     * Why it exists: Differentiates foundational ops from higher-level library ops.
     * What it tracks: Whether this is a primitive tensor operation.
     * What mutates/updates it: Statically populated during definition.
     */
    bool is_primitive_tensor_op = false; 

    /**
     * @brief If true, this is considered a library-level operation.
     * 
     * Why it exists: Identifies ops that may be implemented as composites or runtime calls.
     * What it tracks: Library operation status.
     * What mutates/updates it: Statically populated during definition.
     */
    bool is_library_op = false; 

    /**
     * @brief If true, this library op returns a callable.
     * 
     * Why it exists: Useful for ops that act as higher-order functions or layer constructors.
     * What it tracks: Callable return status for library ops.
     * What mutates/updates it: Statically populated during definition.
     */
    bool is_callable_library_op = false;

    /**
     * @brief If true, the result keeps the general tensor type/shape from the first tensor argument.
     * 
     * Why it exists: Simplifies type inference for shape-preserving ops like activations.
     * What it tracks: Shape preservation semantics.
     * What mutates/updates it: Statically populated during definition.
     */
    bool preserves_first_tensor_arg = false; 

    /**
     * @brief If true, the underlying runtime natively supports this library op directly.
     * 
     * Why it exists: Allows emitting direct hardware/runtime calls instead of decomposing.
     * What it tracks: Native runtime support capability.
     * What mutates/updates it: Statically populated during definition.
     */
    bool runtime_supports_library_op = false; 

    /**
     * @brief Specifies the primitive kind if natively supported.
     * 
     * Why it exists: Maps the generic library op to a specific runtime primitive.
     * What it tracks: The equivalent RuntimePrimitiveKind enum value.
     * What mutates/updates it: Statically populated during definition.
     */
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
