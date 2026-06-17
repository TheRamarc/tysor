#pragma once

#include "frontend_ir.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <variant>
#include <vector>

template <typename T, std::size_t Alignment>
class AlignedAllocator {
    static_assert(Alignment >= alignof(T), "Alignment must satisfy the allocated type");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        if (count == 0) {
            return nullptr;
        }
        return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* pointer, std::size_t) noexcept {
        if (pointer == nullptr) {
            return;
        }
        ::operator delete(pointer, std::align_val_t{Alignment});
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>& lhs, const AlignedAllocator<U, Alignment>& rhs) noexcept {
    return !(lhs == rhs);
}

constexpr std::size_t tensor_data_alignment = 64;
using TensorData = std::vector<float, AlignedAllocator<float, tensor_data_alignment>>;

struct SimpleTensor {
    std::vector<std::int64_t> shape;
    TensorData data;
    std::string dtype = "float32";

    SimpleTensor() = default;
    SimpleTensor(std::vector<std::int64_t> tensor_shape, TensorData tensor_data, std::string tensor_dtype = "float32");
    SimpleTensor(std::vector<std::int64_t> tensor_shape, std::vector<float> tensor_data, std::string tensor_dtype = "float32");
    SimpleTensor(
        std::vector<std::int64_t> tensor_shape,
        std::initializer_list<float> tensor_data,
        std::string tensor_dtype = "float32"
    );
};

struct LinearClosure {
    std::optional<std::int64_t> in_features;
    std::int64_t out_features = 0;
    bool with_bias = true;
    std::string dtype = "float32";
};

struct EmbeddingClosure {
    std::int64_t num_embeddings = 0;
    std::int64_t embedding_dim = 0;
    std::string dtype = "float32";
};

class ShapeView {
public:
    ShapeView() = default;
    ShapeView(const std::vector<std::int64_t>& shape);
    ShapeView(const std::int64_t* values, std::size_t count);

    const std::int64_t* begin() const;
    const std::int64_t* end() const;
    std::size_t size() const;
    bool empty() const;

private:
    const std::int64_t* values_ = nullptr;
    std::size_t size_ = 0;
};

std::size_t num_elements(ShapeView shape);
bool is_aligned_to(const void* pointer, std::size_t alignment);
bool tensor_data_is_aligned(const SimpleTensor& tensor);
SimpleTensor make_synthetic_tensor(ShapeView shape, std::string dtype);
std::string format_tensor(const SimpleTensor& tensor);
void print_tensor(const SimpleTensor& tensor);

std::variant<SimpleTensor, Diagnostic> elementwise_binary(FeBinaryOp op, const SimpleTensor& lhs, const SimpleTensor& rhs);
std::variant<SimpleTensor, Diagnostic> tensor_scalar_binary(FeBinaryOp op, const SimpleTensor& lhs, double rhs);
std::variant<SimpleTensor, Diagnostic> scalar_tensor_binary(FeBinaryOp op, double lhs, const SimpleTensor& rhs);
std::variant<SimpleTensor, Diagnostic> matmul(const SimpleTensor& lhs, const SimpleTensor& rhs);
std::variant<SimpleTensor, Diagnostic> transpose_2d(const SimpleTensor& tensor);

SimpleTensor apply_relu(const SimpleTensor& tensor);
SimpleTensor apply_silu(const SimpleTensor& tensor);
SimpleTensor apply_gelu(const SimpleTensor& tensor);
SimpleTensor apply_tanh(const SimpleTensor& tensor);
SimpleTensor apply_sigmoid(const SimpleTensor& tensor);
std::variant<SimpleTensor, Diagnostic> apply_softmax(const SimpleTensor& tensor);

SimpleTensor make_linear_weight(std::int64_t in_features, std::int64_t out_features, const std::string& dtype);
SimpleTensor make_linear_bias(std::int64_t out_features, const std::string& dtype);
SimpleTensor make_embedding_weight(std::int64_t num_embeddings, std::int64_t embedding_dim, const std::string& dtype);
std::variant<SimpleTensor, Diagnostic> apply_linear(const LinearClosure& closure, const SimpleTensor& input);
std::variant<SimpleTensor, Diagnostic> apply_embedding_with_parameters(
    const SimpleTensor& indices,
    const SimpleTensor& weight,
    std::int64_t num_embeddings,
    std::int64_t embedding_dim
);
std::variant<SimpleTensor, Diagnostic> apply_dropout(const SimpleTensor& input, double probability);
std::variant<SimpleTensor, Diagnostic> apply_reshape(const SimpleTensor& input, ShapeView shape);
std::variant<SimpleTensor, Diagnostic> apply_transpose(const SimpleTensor& input);
SimpleTensor apply_sum(const SimpleTensor& input);
std::variant<SimpleTensor, Diagnostic> apply_sum_axis(const SimpleTensor& input, std::int64_t axis);
SimpleTensor apply_mean(const SimpleTensor& input);
std::variant<SimpleTensor, Diagnostic> apply_mean_axis(const SimpleTensor& input, std::int64_t axis);
SimpleTensor apply_sqrt(const SimpleTensor& input);
SimpleTensor apply_rsqrt(const SimpleTensor& input);
std::variant<SimpleTensor, Diagnostic> apply_repeat_kv(const SimpleTensor& input, std::int64_t repeats);
std::variant<SimpleTensor, Diagnostic> apply_flatten_heads(const SimpleTensor& input);
std::variant<SimpleTensor, Diagnostic> apply_causal_mask(const SimpleTensor& input);
std::variant<SimpleTensor, Diagnostic> apply_rope(const SimpleTensor& input, std::int64_t head_dim, double theta);
std::variant<SimpleTensor, Diagnostic> apply_rms_norm(const SimpleTensor& input, std::int64_t hidden_size);
std::variant<SimpleTensor, Diagnostic> apply_cross_entropy(const SimpleTensor& logits, const SimpleTensor& target);
