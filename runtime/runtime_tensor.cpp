#include "runtime_tensor.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>

namespace {

std::vector<std::int64_t> copy_shape(ShapeView shape) {
    std::vector<std::int64_t> copied;
    copied.reserve(shape.size());
    for (const auto dim : shape) {
        copied.push_back(dim);
    }
    return copied;
}

TensorData data_with_capacity(std::size_t element_count, RuntimeTensorWorkspace* workspace) {
    TensorData data = workspace != nullptr ? workspace->acquire(element_count) : TensorData{};
    data.reserve(element_count);
    return data;
}

std::mt19937& runtime_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

} // namespace

TensorData RuntimeTensorWorkspace::acquire(std::size_t element_count) {
    // Locates the smallest available free buffer that satisfies `element_count`.
    // If found, detaches it for reuse, otherwise allocates a new TensorData buffer.
    auto best = free_buffers_.end();
    for (auto it = free_buffers_.begin(); it != free_buffers_.end(); ++it) {
        if (it->capacity() < element_count) {
            continue;
        }
        if (best == free_buffers_.end() || it->capacity() < best->capacity()) {
            best = it;
        }
    }

    if (best != free_buffers_.end()) {
        TensorData data = std::move(*best);
        free_buffers_.erase(best);
        data.clear();
        ++reuses_;
        return data;
    }

    TensorData data;
    data.reserve(element_count);
    ++allocations_;
    return data;
}

void RuntimeTensorWorkspace::release(SimpleTensor&& tensor) {
    release(std::move(tensor.data));
}

void RuntimeTensorWorkspace::release(TensorData&& data) {
    if (data.capacity() == 0 || !data.unique_storage()) {
        return;
    }
    data.clear();
    free_buffers_.push_back(std::move(data));
    ++releases_;
}

void RuntimeTensorWorkspace::clear() {
    free_buffers_.clear();
}

RuntimeTensorWorkspaceStats RuntimeTensorWorkspace::stats() const {
    RuntimeTensorWorkspaceStats result;
    result.allocations = allocations_;
    result.reuses = reuses_;
    result.releases = releases_;
    result.cached_buffers = free_buffers_.size();
    for (const auto& buffer : free_buffers_) {
        result.cached_bytes += buffer.capacity() * sizeof(float);
    }
    return result;
}

SimpleTensor::SimpleTensor(std::vector<std::int64_t> tensor_shape, TensorData tensor_data, std::string tensorDtype)
    : shape(std::move(tensor_shape)), data(std::move(tensor_data)), dtype(std::move(tensorDtype)) {}

// Keep existing vector-based construction sites working while normalizing the
// stored tensor data into TensorData's aligned allocation policy.
SimpleTensor::SimpleTensor(
    std::vector<std::int64_t> tensor_shape,
    std::vector<float> tensor_data,
    std::string tensorDtype
)
    : shape(std::move(tensor_shape)),
      data(tensor_data.begin(), tensor_data.end()),
      dtype(std::move(tensorDtype)) {}

SimpleTensor::SimpleTensor(
    std::vector<std::int64_t> tensor_shape,
    std::initializer_list<float> tensor_data,
    std::string tensorDtype
)
    : shape(std::move(tensor_shape)), data(tensor_data.begin(), tensor_data.end()), dtype(std::move(tensorDtype)) {}

ShapeView::ShapeView(const std::vector<std::int64_t>& shape)
    : values_(shape.data()), size_(shape.size()) {}

ShapeView::ShapeView(const std::int64_t* values, std::size_t count)
    : values_(values), size_(count) {}

const std::int64_t* ShapeView::begin() const {
    return values_;
}

const std::int64_t* ShapeView::end() const {
    return values_ == nullptr ? nullptr : values_ + size_;
}

std::size_t ShapeView::size() const {
    return size_;
}

bool ShapeView::empty() const {
    return size_ == 0;
}

std::size_t numElements(ShapeView shape) {
    std::int64_t product = 1;
    for (const auto dim : shape) {
        product *= dim;
    }
    return static_cast<std::size_t>(std::max<std::int64_t>(product, 1));
}

bool is_aligned_to(const void* pointer, std::size_t alignment) {
    if (alignment == 0) {
        return false;
    }
    if (pointer == nullptr) {
        return true;
    }
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

bool tensor_data_is_aligned(const SimpleTensor& tensor) {
    return is_aligned_to(tensor.data.data(), tensor_data_alignment);
}

bool tensor_data_shares_storage(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    return lhs.data.shares_storage_with(rhs.data);
}

SimpleTensor make_synthetic_tensor(ShapeView shape, std::string dtype, RuntimeTensorWorkspace* workspace) {
    const std::size_t element_count = numElements(shape);
    // Synthetic parameters are created through TensorData so executor inputs
    // get the same alignment guarantees as intermediate tensors.
    TensorData data = data_with_capacity(element_count, workspace);
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back(static_cast<float>((index % 17) + 1) / 8.0F);
    }
    return SimpleTensor{copy_shape(shape), std::move(data), std::move(dtype)};
}

SimpleTensor make_filled_tensor(ShapeView shape, float value, std::string dtype, RuntimeTensorWorkspace* workspace) {
    const std::size_t element_count = numElements(shape);
    TensorData data = data_with_capacity(element_count, workspace);
    data.assign(element_count, value);
    return SimpleTensor{copy_shape(shape), std::move(data), std::move(dtype)};
}

SimpleTensor make_randn_tensor(ShapeView shape, std::string dtype, RuntimeTensorWorkspace* workspace) {
    const std::size_t element_count = numElements(shape);
    TensorData data = data_with_capacity(element_count, workspace);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    auto& rng = runtime_rng();
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back(dist(rng));
    }
    return SimpleTensor{copy_shape(shape), std::move(data), std::move(dtype)};
}

std::string format_tensor(const SimpleTensor& tensor) {
    std::ostringstream out;
    out << "shape=[";
    for (std::size_t index = 0; index < tensor.shape.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << tensor.shape[index];
    }
    out << "] dtype=" << tensor.dtype << '\n';
    out << "values=[";
    out << std::fixed << std::setprecision(4);
    for (std::size_t index = 0; index < tensor.data.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << tensor.data[index];
    }
    out << ']';
    return out.str();
}

void print_tensor(const SimpleTensor& tensor) {
    std::cout << "\n--- Execution Output ---\n";
    std::cout << format_tensor(tensor) << '\n';
    std::cout << "------------------------\n";
}

SimpleTensor make_linear_weight(
    std::int64_t in_features,
    std::int64_t out_features,
    const std::string& dtype,
    RuntimeTensorWorkspace* workspace
) {
    const std::size_t element_count = static_cast<std::size_t>(in_features * out_features);
    TensorData data = data_with_capacity(element_count, workspace);
    const float stddev = 1.0F / std::sqrt(static_cast<float>(std::max<std::int64_t>(in_features, 1)));
    std::normal_distribution<float> dist(0.0F, stddev);
    auto& rng = runtime_rng();
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back(dist(rng));
    }
    return SimpleTensor{{in_features, out_features}, std::move(data), dtype};
}

SimpleTensor make_linear_bias(std::int64_t out_features, const std::string& dtype, RuntimeTensorWorkspace* workspace) {
    return make_filled_tensor(std::vector<std::int64_t>{out_features}, 0.0F, dtype, workspace);
}

SimpleTensor make_embedding_weight(
    std::int64_t num_embeddings,
    std::int64_t embedding_dim,
    const std::string& dtype,
    RuntimeTensorWorkspace* workspace
) {
    const std::size_t element_count = static_cast<std::size_t>(num_embeddings * embedding_dim);
    TensorData data = data_with_capacity(element_count, workspace);
    const float stddev = 1.0F / std::sqrt(static_cast<float>(std::max<std::int64_t>(embedding_dim, 1)));
    std::normal_distribution<float> dist(0.0F, stddev);
    auto& rng = runtime_rng();
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back(dist(rng));
    }
    return SimpleTensor{{num_embeddings, embedding_dim}, std::move(data), dtype};
}
