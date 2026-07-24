#include "runtime_tensor.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

namespace {

Diagnostic runtime_error(std::string message) {
    return Diagnostic::error(DiagnosticCode::RuntimeError, std::move(message));
}

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

TensorData zeroed_data(std::size_t element_count, RuntimeTensorWorkspace* workspace) {
    TensorData data = data_with_capacity(element_count, workspace);
    data.assign(element_count, 0.0F);
    return data;
}

SimpleTensor copy_tensor(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    TensorData data = data_with_capacity(tensor.data.size(), workspace);
    auto& out = data.writable_storage();
    out.insert(out.end(), tensor.data.begin(), tensor.data.end());
    return SimpleTensor{tensor.shape, std::move(data), tensor.dtype};
}

float apply_binary_scalar(FeBinaryOp op, float lhs, float rhs, std::optional<Diagnostic>& error) {
    switch (op) {
        case FeBinaryOp::Add:
            return lhs + rhs;
        case FeBinaryOp::Sub:
            return lhs - rhs;
        case FeBinaryOp::Mul:
            return lhs * rhs;
        case FeBinaryOp::Div:
            return lhs / rhs;
        case FeBinaryOp::FloorDiv:
            return std::floor(lhs / rhs);
        case FeBinaryOp::Eq:
        case FeBinaryOp::NotEq:
        case FeBinaryOp::Lt:
        case FeBinaryOp::Gt:
        case FeBinaryOp::LtEq:
        case FeBinaryOp::GtEq:
        case FeBinaryOp::And:
        case FeBinaryOp::Or:
        case FeBinaryOp::Not:
            error = runtime_error("unsupported tensor binary operation");
            return 0.0F;
    }
    error = runtime_error("unsupported tensor binary operation");
    return 0.0F;
}

bool is_trailing_vector_broadcast(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    return lhs.shape.size() == 2 &&
           rhs.shape.size() == 1 &&
           lhs.shape[1] == rhs.shape[0];
}

std::variant<SimpleTensor, Diagnostic> apply_linear_with_parameters(
    const SimpleTensor& input,
    const SimpleTensor& weight,
    const std::optional<SimpleTensor>& bias,
    RuntimeTensorWorkspace* workspace
) {
    auto multiplied = matmul(input, weight, workspace);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&multiplied)) {
        return *diagnostic;
    }
    SimpleTensor output = std::get<SimpleTensor>(std::move(multiplied));
    if (bias) {
        if (bias->shape.size() != 1 || output.shape.size() != 2 || bias->shape[0] != output.shape[1]) {
            return runtime_error("linear bias shape mismatch");
        }
        const std::size_t batch = static_cast<std::size_t>(output.shape[0]);
        const std::size_t width = static_cast<std::size_t>(output.shape[1]);
        float* out = output.data.data();
        const float* bias_data = bias->data.data();
        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t col = 0; col < width; ++col) {
                out[row * width + col] += bias_data[col];
            }
        }
    }
    return output;
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

SimpleTensor::SimpleTensor(std::vector<std::int64_t> tensor_shape, TensorData tensor_data, std::string tensor_dtype)
    : shape(std::move(tensor_shape)), data(std::move(tensor_data)), dtype(std::move(tensor_dtype)) {}

// Keep existing vector-based construction sites working while normalizing the
// stored tensor data into TensorData's aligned allocation policy.
SimpleTensor::SimpleTensor(
    std::vector<std::int64_t> tensor_shape,
    std::vector<float> tensor_data,
    std::string tensor_dtype
)
    : shape(std::move(tensor_shape)),
      data(tensor_data.begin(), tensor_data.end()),
      dtype(std::move(tensor_dtype)) {}

SimpleTensor::SimpleTensor(
    std::vector<std::int64_t> tensor_shape,
    std::initializer_list<float> tensor_data,
    std::string tensor_dtype
)
    : shape(std::move(tensor_shape)), data(tensor_data.begin(), tensor_data.end()), dtype(std::move(tensor_dtype)) {}

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

std::size_t num_elements(ShapeView shape) {
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
    const std::size_t element_count = num_elements(shape);
    // Synthetic parameters are created through TensorData so executor inputs
    // get the same alignment guarantees as intermediate tensors.
    TensorData data = data_with_capacity(element_count, workspace);
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back(static_cast<float>((index % 17) + 1) / 8.0F);
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

std::variant<SimpleTensor, Diagnostic> elementwise_binary(
    FeBinaryOp op,
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace
) {
    if (lhs.shape != rhs.shape && !is_trailing_vector_broadcast(lhs, rhs)) {
        return runtime_error("tensor shape mismatch");
    }
    if (lhs.shape == rhs.shape && lhs.data.size() != rhs.data.size()) {
        return runtime_error("tensor data length mismatch");
    }
    TensorData data = data_with_capacity(lhs.data.size(), workspace);
    std::optional<Diagnostic> error;
    auto& out = data.writable_storage();
    const float* lhs_data = lhs.data.data();
    const float* rhs_data = rhs.data.data();
    if (is_trailing_vector_broadcast(lhs, rhs)) {
        const auto rows = static_cast<std::size_t>(lhs.shape[0]);
        const auto cols = static_cast<std::size_t>(lhs.shape[1]);
        if (lhs.data.size() != rows * cols || rhs.data.size() != cols) {
            return runtime_error("tensor broadcast data length mismatch");
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                out.push_back(apply_binary_scalar(op, lhs_data[row * cols + col], rhs_data[col], error));
                if (error) {
                    return *error;
                }
            }
        }
        return SimpleTensor{lhs.shape, std::move(data), lhs.dtype};
    }
    for (std::size_t index = 0; index < lhs.data.size(); ++index) {
        out.push_back(apply_binary_scalar(op, lhs_data[index], rhs_data[index], error));
        if (error) {
            return *error;
        }
    }
    return SimpleTensor{lhs.shape, std::move(data), lhs.dtype};
}

std::variant<SimpleTensor, Diagnostic> tensor_scalar_binary(
    FeBinaryOp op,
    const SimpleTensor& lhs,
    double rhs,
    RuntimeTensorWorkspace* workspace
) {
    TensorData data = data_with_capacity(lhs.data.size(), workspace);
    std::optional<Diagnostic> error;
    const auto scalar = static_cast<float>(rhs);
    auto& out = data.writable_storage();
    for (const auto value : lhs.data) {
        out.push_back(apply_binary_scalar(op, value, scalar, error));
        if (error) {
            return *error;
        }
    }
    return SimpleTensor{lhs.shape, std::move(data), lhs.dtype};
}

std::variant<SimpleTensor, Diagnostic> scalar_tensor_binary(
    FeBinaryOp op,
    double lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace
) {
    TensorData data = data_with_capacity(rhs.data.size(), workspace);
    std::optional<Diagnostic> error;
    const auto scalar = static_cast<float>(lhs);
    auto& out = data.writable_storage();
    for (const auto value : rhs.data) {
        out.push_back(apply_binary_scalar(op, scalar, value, error));
        if (error) {
            return *error;
        }
    }
    return SimpleTensor{rhs.shape, std::move(data), rhs.dtype};
}

std::variant<SimpleTensor, Diagnostic> matmul_impl(
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace,
    bool apply_relu_to_output
) {
    // Core CPU implementation of matrix multiplication. 
    // Supports an optional fused ReLU activation to avoid an extra memory pass.
    if (lhs.shape.size() != 2 || rhs.shape.size() != 2) {
        return runtime_error("matmul currently requires rank-2 tensors");
    }
    const auto m = static_cast<std::size_t>(lhs.shape[0]);
    const auto k = static_cast<std::size_t>(lhs.shape[1]);
    const auto rhs_k = static_cast<std::size_t>(rhs.shape[0]);
    const auto n = static_cast<std::size_t>(rhs.shape[1]);
    if (k != rhs_k) {
        return runtime_error("matmul inner dimension mismatch");
    }
    TensorData output = zeroed_data(m * n, workspace);
    const float* lhs_data = lhs.data.data();
    const float* rhs_data = rhs.data.data();
    float* output_data = output.data();
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float acc = 0.0F;
            for (std::size_t inner = 0; inner < k; ++inner) {
                acc += lhs_data[row * k + inner] * rhs_data[inner * n + col];
            }
            output_data[row * n + col] = apply_relu_to_output ? std::max(acc, 0.0F) : acc;
        }
    }
    return SimpleTensor{{static_cast<std::int64_t>(m), static_cast<std::int64_t>(n)}, std::move(output), lhs.dtype};
}

std::variant<SimpleTensor, Diagnostic> matmul(
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace
) {
    return matmul_impl(lhs, rhs, workspace, false);
}

std::variant<SimpleTensor, Diagnostic> matmul_relu(
    const SimpleTensor& lhs,
    const SimpleTensor& rhs,
    RuntimeTensorWorkspace* workspace
) {
    return matmul_impl(lhs, rhs, workspace, true);
}

std::variant<SimpleTensor, Diagnostic> transpose_2d(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    if (tensor.shape.size() != 2) {
        return runtime_error("transpose_2d currently requires rank-2 tensors");
    }
    const auto rows = static_cast<std::size_t>(tensor.shape[0]);
    const auto cols = static_cast<std::size_t>(tensor.shape[1]);
    TensorData output = zeroed_data(tensor.data.size(), workspace);
    const float* input_data = tensor.data.data();
    float* output_data = output.data();
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            output_data[col * rows + row] = input_data[row * cols + col];
        }
    }
    return SimpleTensor{{static_cast<std::int64_t>(cols), static_cast<std::int64_t>(rows)}, std::move(output), tensor.dtype};
}

SimpleTensor apply_relu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(tensor, workspace);
    for (auto& value : output.data) {
        value = std::max(value, 0.0F);
    }
    return output;
}

SimpleTensor apply_silu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(tensor, workspace);
    for (auto& value : output.data) {
        value = value / (1.0F + std::exp(-value));
    }
    return output;
}

SimpleTensor apply_gelu(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(tensor, workspace);
    for (auto& value : output.data) {
        const float x = value;
        value = 0.5F * x * (1.0F + std::tanh(0.7978846F * (x + 0.044715F * x * x * x)));
    }
    return output;
}

SimpleTensor apply_tanh(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(tensor, workspace);
    for (auto& value : output.data) {
        value = std::tanh(value);
    }
    return output;
}

SimpleTensor apply_sigmoid(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(tensor, workspace);
    for (auto& value : output.data) {
        value = 1.0F / (1.0F + std::exp(-value));
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> apply_softmax(const SimpleTensor& tensor, RuntimeTensorWorkspace* workspace) {
    // Computes softmax along the last dimension.
    // Uses max-value subtraction for numerical stability before exponentiating.
    if (tensor.shape.empty()) {
        return runtime_error("softmax requires a tensor with at least one dimension");
    }
    const auto axis = static_cast<std::size_t>(tensor.shape.back());
    if (axis == 0 || tensor.data.size() % axis != 0) {
        return runtime_error("softmax requires the last dimension to divide the data length");
    }
    SimpleTensor output = copy_tensor(tensor, workspace);
    const float* input_data = tensor.data.data();
    float* output_data = output.data.data();
    const std::size_t rows = tensor.data.size() / axis;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t start = row * axis;
        const float* begin = input_data + start;
        const float* end = begin + axis;
        const float max_value = *std::max_element(begin, end);
        float sum = 0.0F;
        for (const float* it = begin; it != end; ++it) {
            const std::size_t index = static_cast<std::size_t>(it - input_data);
            output_data[index] = std::exp(*it - max_value);
            sum += output_data[index];
        }
        for (std::size_t index = 0; index < axis; ++index) {
            output_data[start + index] /= std::max(sum, std::numeric_limits<float>::epsilon());
        }
    }
    return output;
}

SimpleTensor make_linear_weight(
    std::int64_t in_features,
    std::int64_t out_features,
    const std::string& dtype,
    RuntimeTensorWorkspace* workspace
) {
    const std::size_t element_count = static_cast<std::size_t>(in_features * out_features);
    TensorData data = data_with_capacity(element_count, workspace);
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back((static_cast<float>(index % 13) + 1.0F) / 16.0F);
    }
    return SimpleTensor{{in_features, out_features}, std::move(data), dtype};
}

SimpleTensor make_linear_bias(std::int64_t out_features, const std::string& dtype, RuntimeTensorWorkspace* workspace) {
    TensorData data = data_with_capacity(static_cast<std::size_t>(out_features), workspace);
    for (std::size_t index = 0; index < static_cast<std::size_t>(out_features); ++index) {
        data.push_back((static_cast<float>(index % 7) + 1.0F) / 32.0F);
    }
    return SimpleTensor{{out_features}, std::move(data), dtype};
}

SimpleTensor make_embedding_weight(
    std::int64_t num_embeddings,
    std::int64_t embedding_dim,
    const std::string& dtype,
    RuntimeTensorWorkspace* workspace
) {
    const std::size_t element_count = static_cast<std::size_t>(num_embeddings * embedding_dim);
    TensorData data = data_with_capacity(element_count, workspace);
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back((static_cast<float>(index % 17) + 1.0F) / 8.0F);
    }
    return SimpleTensor{{num_embeddings, embedding_dim}, std::move(data), dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_linear(
    const LinearClosure& closure,
    const SimpleTensor& input,
    RuntimeTensorWorkspace* workspace
) {
    if (input.shape.size() != 2) {
        return runtime_error("linear currently requires rank-2 input tensors");
    }
    const std::int64_t inferred_in_features = input.shape[1];
    const std::int64_t in_features = closure.in_features.value_or(inferred_in_features);
    if (inferred_in_features != in_features) {
        return runtime_error("linear input feature size mismatch");
    }
    SimpleTensor weight = make_linear_weight(in_features, closure.out_features, closure.dtype, workspace);
    if (!closure.with_bias) {
        auto result = matmul(input, weight, workspace);
        if (workspace != nullptr) {
            workspace->release(std::move(weight));
        }
        return result;
    }
    SimpleTensor bias = make_linear_bias(closure.out_features, closure.dtype, workspace);
    auto result = apply_linear_with_parameters(input, weight, bias, workspace);
    if (workspace != nullptr) {
        workspace->release(std::move(weight));
        workspace->release(std::move(bias));
    }
    return result;
}

std::variant<SimpleTensor, Diagnostic> apply_embedding_with_parameters(
    const SimpleTensor& indices,
    const SimpleTensor& weight,
    std::int64_t num_embeddings,
    std::int64_t embedding_dim,
    RuntimeTensorWorkspace* workspace
) {
    if (weight.shape != std::vector<std::int64_t>{num_embeddings, embedding_dim}) {
        return runtime_error("embedding weight shape mismatch");
    }
    std::vector<std::int64_t> shape = indices.shape;
    shape.push_back(embedding_dim);
    TensorData data = zeroed_data(indices.data.size() * static_cast<std::size_t>(embedding_dim), workspace);
    for (std::size_t index = 0; index < indices.data.size(); ++index) {
        const auto embedding_index = static_cast<std::int64_t>(indices.data[index]);
        if (embedding_index < 0 || embedding_index >= num_embeddings) {
            return runtime_error("Embedding index out of range");
        }
        for (std::size_t dim = 0; dim < static_cast<std::size_t>(embedding_dim); ++dim) {
            data[index * static_cast<std::size_t>(embedding_dim) + dim] =
                weight.data[static_cast<std::size_t>(embedding_index) * static_cast<std::size_t>(embedding_dim) + dim];
        }
    }
    return SimpleTensor{std::move(shape), std::move(data), weight.dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_dropout(
    const SimpleTensor& input,
    double probability,
    RuntimeTensorWorkspace* workspace
) {
    if (probability < 0.0 || probability >= 1.0) {
        return runtime_error("dropout probability must be in [0, 1)");
    }
    SimpleTensor output = copy_tensor(input, workspace);
    const float keep_scale = static_cast<float>(1.0 - probability);
    for (auto& value : output.data) {
        value *= keep_scale;
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> apply_reshape(
    const SimpleTensor& input,
    ShapeView shape,
    RuntimeTensorWorkspace* workspace
) {
    (void)workspace;
    if (input.data.size() != num_elements(shape)) {
        return runtime_error("reshape requires matching element counts");
    }
    return SimpleTensor{copy_shape(shape), TensorData::shared_view(input.data), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_transpose(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    return transpose_2d(input, workspace);
}

SimpleTensor apply_sum(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    TensorData data = data_with_capacity(1, workspace);
    data.push_back(std::accumulate(input.data.begin(), input.data.end(), 0.0F));
    return SimpleTensor{{1}, std::move(data), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_sum_axis(
    const SimpleTensor& input,
    std::int64_t axis,
    RuntimeTensorWorkspace* workspace
) {
    if (input.shape.size() != 2) {
        return runtime_error("sum axis currently requires rank-2 tensors");
    }
    const auto rows = static_cast<std::size_t>(input.shape[0]);
    const auto cols = static_cast<std::size_t>(input.shape[1]);
    if (axis == 0) {
        TensorData data = zeroed_data(cols, workspace);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                data[col] += input.data[row * cols + col];
            }
        }
        return SimpleTensor{{input.shape[1]}, std::move(data), input.dtype};
    }
    if (axis == 1) {
        TensorData data = zeroed_data(rows, workspace);
        for (std::size_t row = 0; row < rows; ++row) {
            data[row] = std::accumulate(
                input.data.begin() + static_cast<std::ptrdiff_t>(row * cols),
                input.data.begin() + static_cast<std::ptrdiff_t>((row + 1) * cols),
                0.0F
            );
        }
        return SimpleTensor{{input.shape[0]}, std::move(data), input.dtype};
    }
    return runtime_error("sum axis currently supports axis 0 or 1");
}

SimpleTensor apply_mean(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    const float sum = std::accumulate(input.data.begin(), input.data.end(), 0.0F);
    TensorData data = data_with_capacity(1, workspace);
    data.push_back(sum / static_cast<float>(input.data.size()));
    return SimpleTensor{{1}, std::move(data), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_mean_axis(
    const SimpleTensor& input,
    std::int64_t axis,
    RuntimeTensorWorkspace* workspace
) {
    auto summed = apply_sum_axis(input, axis, workspace);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&summed)) {
        return *diagnostic;
    }
    SimpleTensor output = std::get<SimpleTensor>(std::move(summed));
    const float divisor = static_cast<float>(axis == 0 ? input.shape[0] : input.shape[1]);
    for (auto& value : output.data) {
        value /= divisor;
    }
    return output;
}

SimpleTensor apply_sqrt(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(input, workspace);
    for (auto& value : output.data) {
        value = std::sqrt(value);
    }
    return output;
}

SimpleTensor apply_rsqrt(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    SimpleTensor output = copy_tensor(input, workspace);
    for (auto& value : output.data) {
        value = 1.0F / std::sqrt(value);
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> apply_repeat_kv(
    const SimpleTensor& input,
    std::int64_t repeats,
    RuntimeTensorWorkspace* workspace
) {
    if (repeats <= 0) {
        return runtime_error("repeat_kv repeats must be positive");
    }
    if (input.shape.size() < 2) {
        return runtime_error("repeat_kv expects rank >= 2");
    }
    std::vector<std::int64_t> output_shape = input.shape;
    output_shape[1] *= repeats;
    const auto inner = static_cast<std::size_t>(
        std::accumulate(input.shape.begin() + 2, input.shape.end(), std::int64_t{1}, std::multiplies<>())
    );
    const auto outer = static_cast<std::size_t>(input.shape[0]);
    const auto heads = static_cast<std::size_t>(input.shape[1]);
    TensorData data = zeroed_data(num_elements(output_shape), workspace);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t src_base = (outer_index * heads + head) * inner;
            for (std::size_t rep = 0; rep < static_cast<std::size_t>(repeats); ++rep) {
                const std::size_t dst_head = head * static_cast<std::size_t>(repeats) + rep;
                const std::size_t dst_base = (outer_index * static_cast<std::size_t>(output_shape[1]) + dst_head) * inner;
                std::copy(input.data.begin() + static_cast<std::ptrdiff_t>(src_base),
                          input.data.begin() + static_cast<std::ptrdiff_t>(src_base + inner),
                          data.begin() + static_cast<std::ptrdiff_t>(dst_base));
            }
        }
    }
    return SimpleTensor{std::move(output_shape), std::move(data), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_flatten_heads(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    (void)workspace;
    if (input.shape.size() < 3) {
        return SimpleTensor{input.shape, TensorData::shared_view(input.data), input.dtype};
    }
    std::vector<std::int64_t> shape(input.shape.begin(), input.shape.end() - 2);
    shape.push_back(input.shape[input.shape.size() - 2] * input.shape[input.shape.size() - 1]);
    return SimpleTensor{std::move(shape), TensorData::shared_view(input.data), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> apply_causal_mask(const SimpleTensor& input, RuntimeTensorWorkspace* workspace) {
    if (input.shape.size() < 2) {
        return runtime_error("causal_mask expects rank >= 2");
    }
    SimpleTensor output = copy_tensor(input, workspace);
    const auto q = static_cast<std::size_t>(input.shape[input.shape.size() - 2]);
    const auto k = static_cast<std::size_t>(input.shape[input.shape.size() - 1]);
    const std::size_t inner_stride = q * k;
    const std::size_t outer = input.data.size() / std::max<std::size_t>(inner_stride, 1);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
        const std::size_t base = outer_index * inner_stride;
        for (std::size_t row = 0; row < q; ++row) {
            for (std::size_t col = row + 1; col < k; ++col) {
                output.data[base + row * k + col] = -1.0e4F;
            }
        }
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> apply_rope(
    const SimpleTensor& input,
    std::int64_t head_dim,
    double theta,
    RuntimeTensorWorkspace* workspace
) {
    if (input.shape.empty() || input.shape.back() != head_dim) {
        return runtime_error("rope head_dim mismatch");
    }
    if (head_dim % 2 != 0) {
        return runtime_error("rope requires an even head_dim");
    }
    const auto half = static_cast<std::size_t>(head_dim / 2);
    const auto seq_len = static_cast<std::size_t>(input.shape.size() >= 2 ? input.shape[input.shape.size() - 2] : input.shape[0]);
    const std::size_t width = static_cast<std::size_t>(head_dim);
    const std::size_t outer = input.data.size() / (seq_len * width);
    SimpleTensor output = copy_tensor(input, workspace);
    std::vector<float> inv_freq(half, 0.0F);
    for (std::size_t index = 0; index < half; ++index) {
        inv_freq[index] = std::pow(static_cast<float>(theta), -static_cast<float>(index) / static_cast<float>(half));
    }
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
        const std::size_t outer_base = outer_index * seq_len * width;
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            const std::size_t pos_base = outer_base + pos * width;
            for (std::size_t index = 0; index < half; ++index) {
                const float angle = static_cast<float>(pos) * inv_freq[index];
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float x1 = input.data[pos_base + index];
                const float x2 = input.data[pos_base + half + index];
                output.data[pos_base + index] = x1 * c - x2 * s;
                output.data[pos_base + half + index] = x1 * s + x2 * c;
            }
        }
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> apply_rms_norm(
    const SimpleTensor& input,
    std::int64_t hidden_size,
    RuntimeTensorWorkspace* workspace
) {
    if (input.shape.empty() || input.shape.back() != hidden_size) {
        return runtime_error("rms_norm hidden size mismatch");
    }
    const auto width = static_cast<std::size_t>(hidden_size);
    const std::size_t rows = input.data.size() / width;
    SimpleTensor output = copy_tensor(input, workspace);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t start = row * width;
        float mean_square = 0.0F;
        for (std::size_t index = 0; index < width; ++index) {
            mean_square += input.data[start + index] * input.data[start + index];
        }
        mean_square /= static_cast<float>(width);
        const float scale = 1.0F / std::sqrt(mean_square + 1e-5F);
        for (std::size_t index = 0; index < width; ++index) {
            output.data[start + index] = input.data[start + index] * scale;
        }
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> apply_cross_entropy(
    const SimpleTensor& logits,
    const SimpleTensor& target,
    RuntimeTensorWorkspace* workspace
) {
    if (logits.shape != target.shape) {
        return runtime_error("cross_entropy requires logits and target to have identical shapes");
    }
    auto probabilities_result = apply_softmax(logits, workspace);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&probabilities_result)) {
        return *diagnostic;
    }
    SimpleTensor probabilities = std::get<SimpleTensor>(std::move(probabilities_result));
    const auto width = static_cast<std::size_t>(logits.shape.empty() ? 1 : logits.shape.back());
    const std::size_t rows = logits.data.size() / std::max<std::size_t>(width, 1);
    TensorData losses = zeroed_data(rows, workspace);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t start = row * width;
        for (std::size_t col = 0; col < width; ++col) {
            losses[row] -= target.data[start + col] * std::log(std::max(probabilities.data[start + col], 1e-6F));
        }
    }
    if (workspace != nullptr) {
        workspace->release(std::move(probabilities));
    }
    return SimpleTensor{{static_cast<std::int64_t>(rows), 1}, std::move(losses), logits.dtype};
}
