#pragma once

#include "frontend_ir.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Runtime tensor buffers use an explicit allocator so CPU kernels and backend
// upload paths can rely on a stable alignment contract without requiring C++20.
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

// 64 bytes matches a common cache-line size and is a good baseline for future
// SIMD/vectorized tensor kernels.
constexpr std::size_t tensor_data_alignment = 64;

using TensorDataStorage = std::vector<float, AlignedAllocator<float, tensor_data_alignment>>;

// TensorData keeps vector-like ergonomics for existing kernels while allowing
// explicit metadata-only views. Ordinary copies are deep copies; shared views
// detach on mutation from either side so value-style tensor code remains safe.
class TensorData {
public:
    using value_type = TensorDataStorage::value_type;
    using size_type = TensorDataStorage::size_type;
    using difference_type = TensorDataStorage::difference_type;
    using reference = TensorDataStorage::reference;
    using const_reference = TensorDataStorage::const_reference;
    using pointer = TensorDataStorage::pointer;
    using const_pointer = TensorDataStorage::const_pointer;
    using iterator = TensorDataStorage::iterator;
    using const_iterator = TensorDataStorage::const_iterator;

    TensorData() : storage_(std::make_shared<TensorDataStorage>()) {}
    TensorData(const TensorData& other)
        : storage_(other.storage_ == nullptr
              ? std::make_shared<TensorDataStorage>()
              : std::make_shared<TensorDataStorage>(*other.storage_)) {}
    TensorData(TensorData&& other) noexcept : storage_(std::move(other.storage_)) {
        other.storage_ = std::make_shared<TensorDataStorage>();
    }
    TensorData(std::initializer_list<float> values)
        : storage_(std::make_shared<TensorDataStorage>(values.begin(), values.end())) {}

    template <typename InputIt>
    TensorData(InputIt first, InputIt last) : storage_(std::make_shared<TensorDataStorage>(first, last)) {}

    TensorData& operator=(const TensorData& other) {
        if (this != &other) {
            storage_ = other.storage_ == nullptr
                ? std::make_shared<TensorDataStorage>()
                : std::make_shared<TensorDataStorage>(*other.storage_);
        }
        return *this;
    }
    TensorData& operator=(TensorData&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            other.storage_ = std::make_shared<TensorDataStorage>();
        }
        return *this;
    }

    static TensorData shared_view(const TensorData& data) {
        return TensorData{data.storage_};
    }

    bool shares_storage_with(const TensorData& other) const {
        return storage_ == other.storage_;
    }

    bool unique_storage() const {
        return storage_ != nullptr && storage_.use_count() == 1;
    }

    bool empty() const { return storage_->empty(); }
    size_type size() const { return storage_->size(); }
    size_type capacity() const { return storage_->capacity(); }

    void reserve(size_type count) {
        detach_for_write();
        storage_->reserve(count);
    }

    void assign(size_type count, float value) {
        detach_for_write();
        storage_->assign(count, value);
    }

    void clear() {
        detach_for_write();
        storage_->clear();
    }

    void push_back(float value) {
        detach_for_write();
        storage_->push_back(value);
    }

    iterator begin() {
        detach_for_write();
        return storage_->begin();
    }
    const_iterator begin() const { return storage_->begin(); }
    const_iterator cbegin() const { return storage_->cbegin(); }

    iterator end() {
        detach_for_write();
        return storage_->end();
    }
    const_iterator end() const { return storage_->end(); }
    const_iterator cend() const { return storage_->cend(); }

    pointer data() {
        detach_for_write();
        return storage_->data();
    }
    const_pointer data() const { return storage_->data(); }

    TensorDataStorage& writable_storage() {
        detach_for_write();
        return *storage_;
    }
    const TensorDataStorage& storage() const { return *storage_; }

    reference operator[](size_type index) {
        detach_for_write();
        return (*storage_)[index];
    }
    const_reference operator[](size_type index) const { return (*storage_)[index]; }

    template <typename InputIt>
    iterator insert(const_iterator position, InputIt first, InputIt last) {
        const auto offset = static_cast<difference_type>(position - storage_->cbegin());
        detach_for_write();
        return storage_->insert(storage_->cbegin() + offset, first, last);
    }

private:
    explicit TensorData(std::shared_ptr<TensorDataStorage> storage) : storage_(std::move(storage)) {}

    void detach_for_write() {
        if (storage_ == nullptr) {
            storage_ = std::make_shared<TensorDataStorage>();
            return;
        }
        if (storage_.use_count() != 1) {
            storage_ = std::make_shared<TensorDataStorage>(*storage_);
        }
    }

    // Why it exists: To share underlying data safely across views without deep copies until mutated.
    // What it tracks: A reference-counted pointer to the heap-allocated, aligned float vector.
    // What mutates it: Allocated on creation, cloned on write if refcount > 1.
    std::shared_ptr<TensorDataStorage> storage_;
};

inline bool operator==(const TensorData& lhs, const TensorData& rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

inline bool operator!=(const TensorData& lhs, const TensorData& rhs) {
    return !(lhs == rhs);
}

struct SimpleTensor;

struct RuntimeTensorWorkspaceStats {
    // Why it exists: To track performance metrics of the pool.
    // What it tracks: The number of times the OS was asked for fresh memory.
    // What mutates it: Incremented when a suitably sized buffer wasn't found.
    std::size_t allocations = 0;
    // Why it exists: To measure the pool's effectiveness.
    // What it tracks: The number of times an existing buffer was recycled.
    // What mutates it: Incremented when acquire() successfully recycles a buffer.
    std::size_t reuses = 0;
    // Why it exists: To keep track of returns to the pool.
    // What it tracks: The total number of times tensors were handed back.
    // What mutates it: Incremented during release().
    std::size_t releases = 0;
    std::size_t cached_buffers = 0;
    std::size_t cached_bytes = 0;
};

// RuntimeTensorWorkspace is a small C++17 tensor buffer pool. Kernels still
// return owning SimpleTensor values, but temporary tensors can return their
// aligned buffers to this workspace once the executor knows they are dead.
// This significantly reduces memory allocations during inference.
class RuntimeTensorWorkspace {
public:
    TensorData acquire(std::size_t element_count);
    void release(SimpleTensor&& tensor);
    void release(TensorData&& data);
    void clear();
    RuntimeTensorWorkspaceStats stats() const;

private:
    // Why it exists: To prevent reallocation overhead during tight loops in inference.
    // What it tracks: An object pool of detached, reusable TensorData allocations.
    // What mutates it: Buffers pushed on release(), popped on acquire().
    std::vector<TensorData> free_buffers_;
    // Why it exists: To track performance metrics of the pool.
    // What it tracks: The number of times the OS was asked for fresh memory.
    // What mutates it: Incremented when a suitably sized buffer wasn't found.
    std::size_t allocations_ = 0;
    // Why it exists: To measure the pool's effectiveness.
    // What it tracks: The number of times an existing buffer was recycled.
    // What mutates it: Incremented when acquire() successfully recycles a buffer.
    std::size_t reuses_ = 0;
    // Why it exists: To keep track of returns to the pool.
    // What it tracks: The total number of times tensors were handed back.
    // What mutates it: Incremented during release().
    std::size_t releases_ = 0;
};

// Simple host tensor used by local execution, tests, and host/device transfer.
// Shape stays ordinary vector metadata; data uses TensorData for aligned floats.
// Provides a straightforward CPU-backed multi-dimensional array representation.
struct SimpleTensor {
    // Why it exists: To define the mathematical geometry of the data.
    // What it tracks: The sequence of dimension sizes for this tensor.
    // What mutates it: Assigned on creation or changed via reshape operations.
    std::vector<std::int64_t> shape;
    // Why it exists: To hold the actual numerical content of the tensor.
    // What it tracks: The aligned contiguous floating-point elements.
    // What mutates it: Modified by executor kernels during mathematical operations.
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
    // Why it exists: To define the input size expected by the linear layer.
    // What it tracks: The innermost dimension of the incoming tensor.
    // What mutates it: Parsed from model architecture definitions.
    std::optional<std::int64_t> in_features;
    // Why it exists: To define the output size produced by the linear layer.
    // What it tracks: The size of the newly projected dimension.
    // What mutates it: Parsed from model architecture definitions.
    std::int64_t out_features = 0;
    // Why it exists: To control whether an additive bias is applied.
    // What it tracks: True if a bias parameter should be added after matmul.
    // What mutates it: Configured by layer properties.
    bool with_bias = true;
    std::string dtype = "float32";
};

struct EmbeddingClosure {
    // Why it exists: To define the vocabulary size for the embedding table.
    // What it tracks: The number of discrete indices this embedding can map.
    // What mutates it: Defined at model instantiation.
    std::int64_t num_embeddings = 0;
    // Why it exists: To specify the size of the continuous vector space.
    // What it tracks: The size of the vector assigned to each index.
    // What mutates it: Defined at model instantiation.
    std::int64_t embedding_dim = 0;
    std::string dtype = "float32";
};

// C++17-friendly non-owning shape view. This keeps shape APIs lightweight
// without introducing std::span or raising the project language standard.
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
    // Why it exists: To point to an existing array of dimension sizes without owning it.
    // What it tracks: A raw pointer to the start of the shape array.
    // What mutates it: Bound on construction; immutable.
    const std::int64_t* values_ = nullptr;
    // Why it exists: To safely bound iterations over the shape pointer.
    // What it tracks: The number of dimensions present in the shape.
    // What mutates it: Bound on construction; immutable.
    std::size_t size_ = 0;
};

std::size_t num_elements(ShapeView shape);
bool is_aligned_to(const void* pointer, std::size_t alignment);
bool tensor_data_is_aligned(const SimpleTensor& tensor);
bool tensor_data_shares_storage(const SimpleTensor& lhs, const SimpleTensor& rhs);
SimpleTensor make_synthetic_tensor(ShapeView shape, std::string dtype, RuntimeTensorWorkspace* workspace = nullptr);
std::string format_tensor(const SimpleTensor& tensor);
void print_tensor(const SimpleTensor& tensor);

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
std::variant<SimpleTensor, Diagnostic> transpose_2d(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);

SimpleTensor apply_relu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_silu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_gelu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_tanh(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
SimpleTensor apply_sigmoid(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);
std::variant<SimpleTensor, Diagnostic> apply_softmax(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace = nullptr);

SimpleTensor make_linear_weight(
    std::int64_t in_features,
    std::int64_t out_features,
    const std::string& dtype,
    RuntimeTensorWorkspace* workspace = nullptr
);
SimpleTensor make_linear_bias(
    std::int64_t out_features,
    const std::string& dtype,
    RuntimeTensorWorkspace* workspace = nullptr
);
SimpleTensor make_embedding_weight(
    std::int64_t num_embeddings,
    std::int64_t embedding_dim,
    const std::string& dtype,
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
std::variant<SimpleTensor, Diagnostic> apply_transpose(const SimpleTensor& input, RuntimeTensorWorkspace* workspace = nullptr);
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
