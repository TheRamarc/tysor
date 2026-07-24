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
    Type returnType;
    std::vector<Type> argTypes;
    std::size_t minArity = 0;
    std::size_t maxArity = 0;
};

/**
 * @brief Defines a built-in operation, its signature, and its lowering properties.
 */
struct OpDefinition {
    OpId id = OpId::Print;
    BuiltinSignature signature;
    bool isPrimitiveTensorOp = false; 
    bool isLibraryOp = false; 
    bool isCallableLibraryOp = false;
    bool preservesFirstTensorArg = false; 
    bool runtimeSupportsLibraryOp = false; 
    RuntimePrimitiveKind runtimePrimitiveKind = RuntimePrimitiveKind::Unsupported;
};

const std::vector<OpDefinition>& allOps();
std::vector<BuiltinSignature> allBuiltinSignatures();
const OpDefinition* lookupOp(std::string_view name);
std::optional<OpId> lookupOpId(std::string_view name);
const char* opIdName(OpId id);
bool isBuiltinOp(std::string_view name);
bool isPrimitiveTensorOp(std::string_view name);
bool isLibraryOp(std::string_view name);
bool isCallableLibraryOp(std::string_view name);
bool preservesFirstTensorArg(std::string_view name);
bool runtimeSupportsLibraryOp(std::string_view name);
RuntimePrimitiveKind runtimePrimitive(std::string_view name);
