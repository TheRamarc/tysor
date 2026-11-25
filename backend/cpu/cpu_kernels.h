#pragma once

#include "runtime_tensor.h"

#include <cstdint>
#include <variant>

std::variant<SimpleTensor, Diagnostic> elementwise_binary(
    FeBinaryOp op,
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> tensor_scalar_binary(
    FeBinaryOp op,
    const SimpleTensor& lhs,
    double rhs,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> scalar_tensor_binary(
    FeBinaryOp op,
    double lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> matmul(
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> matmul_relu(
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> transpose_2d(
    const SimpleTensor& tensor,
    RuntimeTensorWorkspace* workspace = nullptr
);

SimpleTensor apply_relu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_silu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_gelu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_tanh(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_sigmoid(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
std::variant<SimpleTensor, Diagnostic> apply_softmax(
    const SimpleTensor& tensor,
    RuntimeTensorWorkspace* workspace = nullptr
);

std::variant<SimpleTensor, Diagnostic> apply_linear(
    const LinearClosure& closure,
    const SimpleTensor& input,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_embedding_with_parameters(
    const SimpleTensor& indices,
    const SimpleTensor& weight,
    std::int64_t num_embeddings,
    std::int64_t embedding_dim,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_dropout(
    const SimpleTensor& input,
    double probability,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_reshape(
    const SimpleTensor& input,
    ShapeView shape,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_transpose(
    const SimpleTensor& input,
    RuntimeTensorWorkspace* workspace = nullptr
);
SimpleTensor apply_sum(const SimpleTensor& input, RuntimeTensorWorkspace* workspace = nullptr);
std::variant<SimpleTensor, Diagnostic> apply_sum_axis(
    const SimpleTensor& input,
    std::int64_t axis,
    RuntimeTensorWorkspace* workspace = nullptr
);
SimpleTensor apply_mean(const SimpleTensor& input, RuntimeTensorWorkspace* workspace = nullptr);
std::variant<SimpleTensor, Diagnostic> apply_mean_axis(
    const SimpleTensor& input,
    std::int64_t axis,
    RuntimeTensorWorkspace* workspace = nullptr
);
SimpleTensor apply_sqrt(const SimpleTensor& input, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_rsqrt(const SimpleTensor& input, RuntimeTensorWorkspace* workspace = nullptr);
std::variant<SimpleTensor, Diagnostic> apply_repeat_kv(
    const SimpleTensor& input,
    std::int64_t repeats,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_flatten_heads(
    const SimpleTensor& input,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_causal_mask(
    const SimpleTensor& input,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_rope(
    const SimpleTensor& input,
    std::int64_t head_dim,
    double theta,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_rms_norm(
    const SimpleTensor& input,
    std::int64_t hidden_size,
    RuntimeTensorWorkspace* workspace = nullptr
);
std::variant<SimpleTensor, Diagnostic> apply_cross_entropy(
    const SimpleTensor& logits,
    const SimpleTensor& target,
    RuntimeTensorWorkspace* workspace = nullptr
);
