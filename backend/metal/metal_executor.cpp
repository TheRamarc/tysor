#include "metal_executor.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#if defined(__APPLE__)
extern "C" {
const char* tysor_metal_last_error(void);
const char* tysor_metal_device_report(void);
bool tysor_metal_probe_device(void);
void* tysor_metal_context_new(const char* source);
void tysor_metal_context_free(void* context);
void* tysor_metal_buffer_new_with_data(void* context, const float* data, std::size_t count);
void* tysor_metal_buffer_new_zeroed(void* context, std::size_t count);
void tysor_metal_buffer_free(void* buffer);
bool tysor_metal_buffer_read(void* buffer, float* out_data, std::size_t count);
bool tysor_metal_dispatch_matmul(
    void* context,
    const char* kernel_name,
    void* lhs,
    void* rhs,
    void* out,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k
);
bool tysor_metal_dispatch_unary(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t count
);
bool tysor_metal_dispatch_reduction(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t rows,
    std::uint32_t cols,
    std::int32_t axis,
    std::uint32_t output_count
);
bool tysor_metal_dispatch_binary_tt(
    void* context,
    const char* kernel_name,
    void* lhs,
    void* rhs,
    void* out,
    std::uint32_t count
);
bool tysor_metal_dispatch_binary_ts_scalar(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t count,
    float scalar
);
bool tysor_metal_dispatch_binary_st_scalar(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t count,
    float scalar
);
bool tysor_metal_dispatch_transpose(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t rows,
    std::uint32_t cols
);
bool tysor_metal_dispatch_linear(
    void* context,
    const char* kernel_name,
    void* input,
    void* weight,
    void* bias,
    void* out,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    bool with_bias
);
bool tysor_metal_dispatch_embedding(
    void* context,
    const char* kernel_name,
    void* indices,
    void* weight,
    void* out,
    std::uint32_t index_count,
    std::uint32_t embedding_dim
);
bool tysor_metal_dispatch_softmax(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t rows,
    std::uint32_t width
);
bool tysor_metal_dispatch_repeat_kv(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t outer,
    std::uint32_t out_heads,
    std::uint32_t inner,
    std::uint32_t repeats
);
bool tysor_metal_dispatch_causal_mask(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t outer,
    std::uint32_t q,
    std::uint32_t k
);
bool tysor_metal_dispatch_rope_runtime(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t outer,
    std::uint32_t seq_len,
    std::uint32_t half_dim,
    float theta
);
bool tysor_metal_dispatch_rms_norm(
    void* context,
    const char* kernel_name,
    void* input,
    void* out,
    std::uint32_t rows,
    std::uint32_t width
);
bool tysor_metal_dispatch_cross_entropy(
    void* context,
    const char* kernel_name,
    void* logits,
    void* target,
    void* out,
    std::uint32_t rows,
    std::uint32_t width
);
}
#endif

namespace {

// Helper to construct a unified diagnostic error for Metal backend failures
Diagnostic metal_error(std::string message) {
    return Diagnostic::error(DiagnosticCode::RuntimeExecutionError, std::move(message));
}

// Locates a specific execution plan by its entry function name
const ExecutionPlan* find_plan(const PlanModule& module, const std::string& entry) {
    auto found = std::find_if(module.plans.begin(), module.plans.end(), [&](const ExecutionPlan& plan) {
        return plan.name == entry;
    });
    return found == module.plans.end() ? nullptr : &*found;
}

// Synthesizes a tensor with specific shape and type for testing/execution from CLI shapes
std::variant<SimpleTensor, Diagnostic> make_tensor_argument(
    const PlanValue& value,
    const GraphExecutorOptions& options
) {
    auto shape = options.tensor_shapes.find(value.name);
    if (shape == options.tensor_shapes.end()) {
        return metal_error("Missing --shape for tensor parameter '" + value.name + "'");
    }
    return make_synthetic_tensor(shape->second, value.type.tensor_dtype.value_or("float32"));
}

#if defined(__APPLE__)
const char* metal_source() {
    return R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        acc += lhs[row * k + inner] * rhs[inner * n + col];
    }
    out[row * n + col] = acc;
}

kernel void linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        acc += input[row * k + inner] * weight[inner * n + col];
    }
    acc += bias[col];
    out[row * n + col] = acc;
}

kernel void linear_no_bias(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        acc += input[row * k + inner] * weight[inner * n + col];
    }
    out[row * n + col] = acc;
}

kernel void embedding(
    device const float* indices [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& index_count [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= embedding_dim || gid.y >= index_count) {
        return;
    }
    uint index = uint(max(indices[gid.y], 0.0f));
    out[gid.y * embedding_dim + gid.x] = weight[index * embedding_dim + gid.x];
}

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = max(input[gid], 0.0f);
}

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float x = input[gid];
    out[gid] = x / (1.0f + exp(-x));
}

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float x = input[gid];
    out[gid] = 0.5f * x * (1.0f + tanh(0.7978846f * (x + 0.044715f * x * x * x)));
}

kernel void tanh_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = tanh(input[gid]);
}

kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float x = input[gid];
    out[gid] = 1.0f / (1.0f + exp(-x));
}

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float max_value = input[row * width];
    for (uint col = 1; col < width; ++col) {
        max_value = max(max_value, input[row * width + col]);
    }
    float sum = 0.0f;
    for (uint col = 0; col < width; ++col) {
        float e = exp(input[row * width + col] - max_value);
        out[row * width + col] = e;
        sum += e;
    }
    for (uint col = 0; col < width; ++col) {
        out[row * width + col] = out[row * width + col] / sum;
    }
}

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float mean_sq = 0.0f;
    for (uint col = 0; col < width; ++col) {
        float x = input[row * width + col];
        mean_sq += x * x;
    }
    mean_sq /= float(width);
    float inv_rms = rsqrt(mean_sq + 1.0e-5f);
    for (uint col = 0; col < width; ++col) {
        out[row * width + col] = input[row * width + col] * inv_rms;
    }
}

kernel void cross_entropy(
    device const float* logits [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float max_value = logits[row * width];
    for (uint col = 1; col < width; ++col) {
        max_value = max(max_value, logits[row * width + col]);
    }
    float sum = 0.0f;
    for (uint col = 0; col < width; ++col) {
        sum += exp(logits[row * width + col] - max_value);
    }
    float loss = 0.0f;
    for (uint col = 0; col < width; ++col) {
        float probability = exp(logits[row * width + col] - max_value) / sum;
        loss -= target[row * width + col] * log(max(probability, 1.0e-6f));
    }
    out[row] = loss;
}

kernel void sqrt_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = sqrt(input[gid]);
}

kernel void rsqrt_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = rsqrt(input[gid]);
}

kernel void copy_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = input[gid];
}

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    out[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}

kernel void repeat_kv(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& out_heads [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& repeats [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inner || gid.y >= outer * out_heads) {
        return;
    }
    uint outer_idx = gid.y / out_heads;
    uint dst_head = gid.y % out_heads;
    uint src_head = dst_head / repeats;
    uint in_heads = out_heads / repeats;
    uint src = (outer_idx * in_heads + src_head) * inner + gid.x;
    uint dst = gid.y * inner + gid.x;
    out[dst] = input[src];
}

kernel void causal_mask(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& q [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= k || gid.y >= outer * q) {
        return;
    }
    uint outer_idx = gid.y / q;
    uint row = gid.y % q;
    uint index = outer_idx * q * k + row * k + gid.x;
    float value = input[index];
    if (gid.x > row) {
        value = -10000.0f;
    }
    out[index] = value;
}

kernel void rope(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& half_dim [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= half_dim || gid.y >= outer * seq_len) {
        return;
    }
    uint outer_idx = gid.y / seq_len;
    uint pos = gid.y % seq_len;
    uint head_dim = half_dim * 2;
    uint base = outer_idx * seq_len * head_dim + pos * head_dim;
    float inv_freq = pow(theta, -(float(gid.x) / float(half_dim)));
    float angle = float(pos) * inv_freq;
    float c = cos(angle);
    float s = sin(angle);
    float x1 = input[base + gid.x];
    float x2 = input[base + half_dim + gid.x];
    out[base + gid.x] = x1 * c - x2 * s;
    out[base + half_dim + gid.x] = x1 * s + x2 * c;
}

kernel void sum_reduce(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    if (axis == 0) {
        if (gid >= cols) return;
        for (uint row = 0; row < rows; ++row) {
            acc += input[row * cols + gid];
        }
        out[gid] = acc;
        return;
    }
    if (axis == 1) {
        if (gid >= rows) return;
        uint base = gid * cols;
        for (uint col = 0; col < cols; ++col) {
            acc += input[base + col];
        }
        out[gid] = acc;
        return;
    }
    if (gid > 0) return;
    uint count = rows * cols;
    for (uint index = 0; index < count; ++index) {
        acc += input[index];
    }
    out[0] = acc;
}

kernel void mean_reduce(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    if (axis == 0) {
        if (gid >= cols) return;
        for (uint row = 0; row < rows; ++row) {
            acc += input[row * cols + gid];
        }
        out[gid] = acc / float(rows);
        return;
    }
    if (axis == 1) {
        if (gid >= rows) return;
        uint base = gid * cols;
        for (uint col = 0; col < cols; ++col) {
            acc += input[base + col];
        }
        out[gid] = acc / float(cols);
        return;
    }
    if (gid > 0) return;
    uint count = rows * cols;
    for (uint index = 0; index < count; ++index) {
        acc += input[index];
    }
    out[0] = acc / float(count);
}

#define TYSOR_BINARY_TT(name, expr) \
kernel void name(device const float* lhs [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { \
    if (gid >= count) return; \
    float l = lhs[gid]; \
    float r = rhs[gid]; \
    out[gid] = (expr); \
}

#define TYSOR_BINARY_TS(name, expr) \
kernel void name(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], constant float& scalar [[buffer(3)]], uint gid [[thread_position_in_grid]]) { \
    if (gid >= count) return; \
    float t = input[gid]; \
    float s = scalar; \
    out[gid] = (expr); \
}

TYSOR_BINARY_TT(add_tt, l + r)
TYSOR_BINARY_TT(sub_tt, l - r)
TYSOR_BINARY_TT(mul_tt, l * r)
TYSOR_BINARY_TT(div_tt, l / r)
TYSOR_BINARY_TT(floordiv_tt, floor(l / r))
TYSOR_BINARY_TS(add_ts, t + s)
TYSOR_BINARY_TS(sub_ts, t - s)
TYSOR_BINARY_TS(mul_ts, t * s)
TYSOR_BINARY_TS(div_ts, t / s)
TYSOR_BINARY_TS(floordiv_ts, floor(t / s))
TYSOR_BINARY_TS(add_st, s + t)
TYSOR_BINARY_TS(sub_st, s - t)
TYSOR_BINARY_TS(mul_st, s * t)
TYSOR_BINARY_TS(div_st, s / t)
TYSOR_BINARY_TS(floordiv_st, floor(s / t))

#undef TYSOR_BINARY_TT
#undef TYSOR_BINARY_TS
)";
}

// Retrieves the last error string reported by the Objective-C Metal bridge
std::string last_metal_error() {
    const char* error = tysor_metal_last_error();
    return error == nullptr || std::string(error).empty() ? "unknown Metal bridge error" : std::string(error);
}

// Custom deleter for bridging C++ unique_ptr with Objective-C ARC buffer deallocation
struct MetalBufferDeleter {
    void operator()(void* buffer) const {
        if (buffer != nullptr) {
            tysor_metal_buffer_free(buffer);
        }
    }
};

using MetalBufferPtr = std::unique_ptr<void, MetalBufferDeleter>;

// Structs to hold metadata for delayed activation function dispatch
struct ActivationClosure {
    std::string op;
    double probability = 0.0;
};

// Represents an allocated memory buffer on the Metal GPU device
struct DeviceTensor {
    SimpleTensor tensor;
    MetalBufferPtr buffer;
};

using MetalHostValue = std::variant<std::int64_t, double, bool, SimpleTensor, LinearClosure, EmbeddingClosure, ActivationClosure>;

struct HostValue {
    MetalHostValue value;
};

// Represents a value living either on the CPU (HostValue) or GPU (DeviceTensor)
using MetalValue = std::variant<HostValue, DeviceTensor>;

// Allocates a Metal buffer and copies the host tensor data to the device
std::variant<DeviceTensor, Diagnostic> upload_tensor(void* context, SimpleTensor tensor) {
    void* buffer = tysor_metal_buffer_new_with_data(context, tensor.data.data(), tensor.data.size());
    if (buffer == nullptr) {
        return metal_error("Metal upload failed: " + last_metal_error());
    }
    return DeviceTensor{std::move(tensor), MetalBufferPtr(buffer)};
}

// Allocates an empty/zero-initialized Metal buffer for device results
std::variant<DeviceTensor, Diagnostic> allocate_device_tensor(
    void* context,
    std::vector<std::int64_t> shape,
    std::string dtype
) {
    void* buffer = tysor_metal_buffer_new_zeroed(context, num_elements(shape));
    if (buffer == nullptr) {
        return metal_error("Metal allocation failed: " + last_metal_error());
    }
    return DeviceTensor{SimpleTensor{std::move(shape), std::vector<float>{}, std::move(dtype)}, MetalBufferPtr(buffer)};
}

DeviceTensor* require_device_tensor(std::map<std::size_t, MetalValue>& values, std::size_t id, const std::string& label) {
    (void)label;
    auto found = values.find(id);
    if (found == values.end()) {
        return nullptr;
    }
    return std::get_if<DeviceTensor>(&found->second);
}

HostValue* require_host_value(std::map<std::size_t, MetalValue>& values, std::size_t id) {
    auto found = values.find(id);
    if (found == values.end()) {
        return nullptr;
    }
    return std::get_if<HostValue>(&found->second);
}

std::variant<double, Diagnostic> require_host_number(std::map<std::size_t, MetalValue>& values, std::size_t id, const std::string& label) {
    HostValue* host = require_host_value(values, id);
    if (host == nullptr) {
        return metal_error("Missing " + label);
    }
    if (const auto* value = std::get_if<std::int64_t>(&host->value)) {
        return static_cast<double>(*value);
    }
    if (const auto* value = std::get_if<double>(&host->value)) {
        return *value;
    }
    if (const auto* value = std::get_if<bool>(&host->value)) {
        return *value ? 1.0 : 0.0;
    }
    return metal_error("Metal executor expected numeric host value for " + label);
}

std::variant<std::int64_t, Diagnostic> require_host_int(std::map<std::size_t, MetalValue>& values, std::size_t id, const std::string& label) {
    HostValue* host = require_host_value(values, id);
    if (host == nullptr) {
        return metal_error("Missing " + label);
    }
    if (const auto* value = std::get_if<std::int64_t>(&host->value)) {
        return *value;
    }
    return metal_error("Metal executor expected integer host value for " + label);
}

std::variant<HostValue, Diagnostic> constant_to_host_value(const FeValue& value) {
    return std::visit(
        [](const auto& inner) -> std::variant<HostValue, Diagnostic> {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                return HostValue{MetalHostValue{inner}};
            } else if constexpr (std::is_same_v<T, double>) {
                return HostValue{MetalHostValue{inner}};
            } else if constexpr (std::is_same_v<T, bool>) {
                return HostValue{MetalHostValue{inner}};
            } else {
                return metal_error("Unsupported Metal constant");
            }
        },
        value.value
    );
}

std::variant<GraphRuntimeValue, Diagnostic> host_to_graph_value(const HostValue& host) {
    if (const auto* value = std::get_if<std::int64_t>(&host.value)) {
        return GraphRuntimeValue{*value};
    }
    if (const auto* value = std::get_if<double>(&host.value)) {
        return GraphRuntimeValue{*value};
    }
    if (const auto* value = std::get_if<bool>(&host.value)) {
        return GraphRuntimeValue{*value};
    }
    if (const auto* value = std::get_if<SimpleTensor>(&host.value)) {
        return GraphRuntimeValue{*value};
    }
    return metal_error("Metal runtime cannot materialize callable host values");
}

std::variant<LinearClosure, Diagnostic> build_linear_closure(
    const PlanOp& op,
    const PlanValue& output,
    std::map<std::size_t, MetalValue>& values
) {
    LinearClosure closure;
    if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
        closure.dtype = *output.type.callable_return->tensor_dtype;
    }
    if (op.inputs.size() == 1) {
        auto out_features = require_host_int(values, op.inputs[0], "out_features");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
            return *diagnostic;
        }
        closure.out_features = std::get<std::int64_t>(out_features);
        return closure;
    }
    if (op.inputs.size() < 2 || op.inputs.size() > 3) {
        return metal_error("Metal linear constructor expected 1 to 3 arguments");
    }
    auto in_features = require_host_int(values, op.inputs[0], "in_features");
    auto out_features = require_host_int(values, op.inputs[1], "out_features");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&in_features)) {
        return *diagnostic;
    }
    if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
        return *diagnostic;
    }
    closure.in_features = std::get<std::int64_t>(in_features);
    closure.out_features = std::get<std::int64_t>(out_features);
    if (op.inputs.size() == 3) {
        HostValue* host = require_host_value(values, op.inputs[2]);
        if (host == nullptr) {
            return metal_error("Missing with_bias");
        }
        if (const auto* with_bias = std::get_if<bool>(&host->value)) {
            closure.with_bias = *with_bias;
        } else {
            return metal_error("Metal linear constructor expected bool with_bias");
        }
    }
    return closure;
}

std::variant<EmbeddingClosure, Diagnostic> build_embedding_closure(
    const PlanOp& op,
    const PlanValue& output,
    std::map<std::size_t, MetalValue>& values
) {
    if (op.inputs.size() != 2) {
        return metal_error("Metal Embedding constructor expected num_embeddings and embedding_dim");
    }
    auto num_embeddings = require_host_int(values, op.inputs[0], "num_embeddings");
    auto embedding_dim = require_host_int(values, op.inputs[1], "embedding_dim");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&num_embeddings)) {
        return *diagnostic;
    }
    if (const auto* diagnostic = std::get_if<Diagnostic>(&embedding_dim)) {
        return *diagnostic;
    }
    EmbeddingClosure closure;
    closure.num_embeddings = std::get<std::int64_t>(num_embeddings);
    closure.embedding_dim = std::get<std::int64_t>(embedding_dim);
    if (closure.num_embeddings <= 0 || closure.embedding_dim <= 0) {
        return metal_error("Metal Embedding dimensions must be positive");
    }
    if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
        closure.dtype = *output.type.callable_return->tensor_dtype;
    }
    return closure;
}

bool is_activation_constructor(const std::string& op) {
    return op == "SiLU" || op == "GELU" || op == "Tanh" || op == "Sigmoid" || op == "Softmax" || op == "Dropout";
}

std::variant<ActivationClosure, Diagnostic> build_activation_closure(
    const PlanOp& op,
    std::map<std::size_t, MetalValue>& values
) {
    if (!is_activation_constructor(op.op)) {
        return metal_error("Metal activation constructor expected an activation op");
    }
    ActivationClosure closure;
    closure.op = op.op;
    if (op.op == "Dropout") {
        if (op.inputs.empty()) {
            return metal_error("Metal Dropout constructor expected probability");
        }
        auto probability = require_host_number(values, op.inputs[0], "dropout probability");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&probability)) {
            return *diagnostic;
        }
        closure.probability = std::get<double>(probability);
        if (closure.probability < 0.0 || closure.probability >= 1.0) {
            return metal_error("dropout probability must be in [0, 1)");
        }
    }
    return closure;
}

// Predicts the output shape of various operations prior to their dispatch
std::variant<std::vector<std::int64_t>, Diagnostic> infer_output_shape(
    const PlanOp& op,
    std::map<std::size_t, MetalValue>& values
) {
    // Matmul output dimensions are determined by [lhs_rows, rhs_cols]
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "matmul") {
        DeviceTensor* lhs = require_device_tensor(values, op.inputs[0], "matmul lhs");
        DeviceTensor* rhs = require_device_tensor(values, op.inputs[1], "matmul rhs");
        if (lhs == nullptr || rhs == nullptr) {
            return metal_error("Metal matmul requires device tensor inputs");
        }
        return std::vector<std::int64_t>{lhs->tensor.shape[0], rhs->tensor.shape[1]};
    }
    if ((op.kind == PlanOpKind::PrimitiveCall && (op.op == "relu" || op.op == "scale")) ||
        (op.kind == PlanOpKind::LibraryCall && (op.op == "sqrt" || op.op == "rsqrt"))) {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], op.op + " input");
        if (input == nullptr) {
            return metal_error("Metal " + op.op + " requires a device tensor input");
        }
        return input->tensor.shape;
    }
    if (op.kind == PlanOpKind::Binary) {
        if (DeviceTensor* lhs = require_device_tensor(values, op.inputs[0], "binary lhs")) {
            return lhs->tensor.shape;
        }
        if (DeviceTensor* rhs = require_device_tensor(values, op.inputs[1], "binary rhs")) {
            return rhs->tensor.shape;
        }
        return metal_error("Metal does not support scalar-scalar binary ops");
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "reshape") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "reshape input");
        if (input == nullptr) {
            return metal_error("Metal reshape requires a device tensor input");
        }
        std::vector<std::int64_t> shape;
        for (std::size_t index = 1; index < op.inputs.size(); ++index) {
            auto dim = require_host_int(values, op.inputs[index], "reshape dim");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&dim)) {
                return *diagnostic;
            }
            shape.push_back(std::get<std::int64_t>(dim));
        }
        if (num_elements(shape) != num_elements(input->tensor.shape)) {
            return metal_error("Metal reshape requires matching element counts");
        }
        return shape;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "transpose") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "transpose input");
        if (input == nullptr) {
            return metal_error("Metal transpose requires a device tensor input");
        }
        if (input->tensor.shape.size() != 2) {
            return metal_error("Metal transpose currently requires rank-2 tensors");
        }
        return std::vector<std::int64_t>{input->tensor.shape[1], input->tensor.shape[0]};
    }
    if (op.kind == PlanOpKind::LibraryCall && (op.op == "sum" || op.op == "mean")) {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], op.op + " input");
        if (input == nullptr) {
            return metal_error("Metal " + op.op + " requires a device tensor input");
        }
        if (op.inputs.size() == 1) {
            return std::vector<std::int64_t>{1};
        }
        if (input->tensor.shape.size() != 2) {
            return metal_error("Metal " + op.op + " axis currently requires rank-2 tensors");
        }
        auto axis = require_host_int(values, op.inputs[1], op.op + " axis");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) {
            return *diagnostic;
        }
        switch (std::get<std::int64_t>(axis)) {
            case 0:
                return std::vector<std::int64_t>{input->tensor.shape[1]};
            case 1:
                return std::vector<std::int64_t>{input->tensor.shape[0]};
            default:
                return metal_error("Metal " + op.op + " axis currently supports axis 0 or 1");
        }
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rms_norm") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "rms_norm input");
        if (input == nullptr) {
            return metal_error("Metal rms_norm requires a device tensor input");
        }
        if (input->tensor.shape.empty()) {
            return metal_error("Metal rms_norm requires a tensor with at least one dimension");
        }
        auto hidden = require_host_int(values, op.inputs[1], "rms_norm hidden size");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&hidden)) {
            return *diagnostic;
        }
        if (input->tensor.shape.back() != std::get<std::int64_t>(hidden)) {
            return metal_error("Metal rms_norm hidden size mismatch");
        }
        return input->tensor.shape;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "cross_entropy") {
        DeviceTensor* logits = require_device_tensor(values, op.inputs[0], "cross_entropy logits");
        DeviceTensor* target = require_device_tensor(values, op.inputs[1], "cross_entropy target");
        if (logits == nullptr || target == nullptr) {
            return metal_error("Metal cross_entropy requires device tensor inputs");
        }
        if (logits->tensor.shape.size() != 2 || target->tensor.shape.size() != 2) {
            return metal_error("Metal cross_entropy requires rank-2 tensors");
        }
        if (logits->tensor.shape != target->tensor.shape) {
            return metal_error("Metal cross_entropy requires logits and target to have identical shapes");
        }
        return std::vector<std::int64_t>{logits->tensor.shape[0], 1};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "repeat_kv") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "repeat_kv input");
        if (input == nullptr) {
            return metal_error("Metal repeat_kv requires a device tensor input");
        }
        if (input->tensor.shape.size() < 2) {
            return metal_error("Metal repeat_kv expects rank >= 2");
        }
        auto repeats = require_host_int(values, op.inputs[1], "repeat_kv repeats");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&repeats)) {
            return *diagnostic;
        }
        if (std::get<std::int64_t>(repeats) <= 0) {
            return metal_error("Metal repeat_kv repeats must be positive");
        }
        std::vector<std::int64_t> output_shape = input->tensor.shape;
        output_shape[1] *= std::get<std::int64_t>(repeats);
        return output_shape;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "flatten_heads") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "flatten_heads input");
        if (input == nullptr) {
            return metal_error("Metal flatten_heads requires a device tensor input");
        }
        if (input->tensor.shape.size() < 3) {
            return input->tensor.shape;
        }
        std::vector<std::int64_t> output_shape(input->tensor.shape.begin(), input->tensor.shape.end() - 2);
        output_shape.push_back(input->tensor.shape[input->tensor.shape.size() - 2] * input->tensor.shape.back());
        return output_shape;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "causal_mask") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "causal_mask input");
        if (input == nullptr) {
            return metal_error("Metal causal_mask requires a device tensor input");
        }
        if (input->tensor.shape.size() < 2) {
            return metal_error("Metal causal_mask expects rank >= 2");
        }
        return input->tensor.shape;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rope") {
        DeviceTensor* input = require_device_tensor(values, op.inputs[0], "rope input");
        if (input == nullptr) {
            return metal_error("Metal rope requires a device tensor input");
        }
        if (input->tensor.shape.size() < 2) {
            return metal_error("Metal rope expects rank >= 2");
        }
        auto head_dim = require_host_int(values, op.inputs[1], "rope head_dim");
        auto theta = require_host_number(values, op.inputs[2], "rope theta");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&head_dim)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&theta)) {
            return *diagnostic;
        }
        (void)theta;
        const std::int64_t dim = std::get<std::int64_t>(head_dim);
        if (dim % 2 != 0) {
            return metal_error("Metal rope requires an even head_dim");
        }
        if (input->tensor.shape.back() != dim) {
            return metal_error("Metal rope head_dim mismatch");
        }
        return input->tensor.shape;
    }
    if (op.kind == PlanOpKind::Apply) {
        HostValue* callee = require_host_value(values, op.inputs[0]);
        DeviceTensor* input = require_device_tensor(values, op.inputs[1], "apply input");
        if (callee == nullptr || input == nullptr) {
            return metal_error("Metal apply requires a host closure and device tensor input");
        }
        const auto* closure = std::get_if<LinearClosure>(&callee->value);
        if (closure != nullptr) {
            if (input->tensor.shape.size() != 2) {
                return metal_error("Metal linear currently requires rank-2 input tensors");
            }
            const std::int64_t inferred_in = input->tensor.shape[1];
            const std::int64_t expected_in = closure->in_features.value_or(inferred_in);
            if (inferred_in != expected_in) {
                return metal_error("Metal linear input feature size mismatch");
            }
            return std::vector<std::int64_t>{input->tensor.shape[0], closure->out_features};
        }
        const auto* embedding = std::get_if<EmbeddingClosure>(&callee->value);
        if (embedding != nullptr) {
            std::vector<std::int64_t> output_shape = input->tensor.shape;
            output_shape.push_back(embedding->embedding_dim);
            return output_shape;
        }
        const auto* activation = std::get_if<ActivationClosure>(&callee->value);
        if (activation != nullptr) {
            if (activation->op == "Softmax" && input->tensor.shape.empty()) {
                return metal_error("Metal softmax requires a tensor with at least one dimension");
            }
            return input->tensor.shape;
        }
        return metal_error("Metal Apply currently supports linear, embedding, and activation closures only");
    }
    return metal_error("Metal cannot infer output shape for planned op '" + op.op + "'");
}

// Computes matrix multiplication using the Metal bridged dispatch
std::variant<DeviceTensor, Diagnostic> dispatch_matmul(
    void* context,
    const DeviceTensor& lhs,
    const DeviceTensor& rhs
) {
    if (lhs.tensor.shape.size() != 2 || rhs.tensor.shape.size() != 2) {
        return metal_error("Metal matmul currently requires rank-2 tensors");
    }
    const auto m = static_cast<std::uint32_t>(lhs.tensor.shape[0]);
    const auto k = static_cast<std::uint32_t>(lhs.tensor.shape[1]);
    const auto rhs_k = static_cast<std::uint32_t>(rhs.tensor.shape[0]);
    const auto n = static_cast<std::uint32_t>(rhs.tensor.shape[1]);
    
    // Ensure inner dimensions match
    if (k != rhs_k) {
        return metal_error("Metal matmul inner dimension mismatch");
    }
    
    // Allocate the output matrix C
    auto output = allocate_device_tensor(context, {m, n}, lhs.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    
    // Dispatch to the Objective-C Metal runner
    if (!tysor_metal_dispatch_matmul(context, "matmul", lhs.buffer.get(), rhs.buffer.get(), out.buffer.get(), m, n, k)) {
        return metal_error("Metal matmul dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_unary(
    void* context,
    const DeviceTensor& input,
    const char* kernel_name
) {
    auto output = allocate_device_tensor(context, input.tensor.shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    if (!tysor_metal_dispatch_unary(
            context,
            kernel_name,
            input.buffer.get(),
            out.buffer.get(),
            static_cast<std::uint32_t>(num_elements(input.tensor.shape))
        )) {
        return metal_error(std::string("Metal unary dispatch failed: ") + last_metal_error());
    }
    return out;
}

std::variant<std::string, Diagnostic> binary_kernel_name(FeBinaryOp op, const char* suffix) {
    switch (op) {
        case FeBinaryOp::Add:
            return std::string("add_") + suffix;
        case FeBinaryOp::Sub:
            return std::string("sub_") + suffix;
        case FeBinaryOp::Mul:
            return std::string("mul_") + suffix;
        case FeBinaryOp::Div:
            return std::string("div_") + suffix;
        case FeBinaryOp::FloorDiv:
            return std::string("floordiv_") + suffix;
        case FeBinaryOp::Eq:
        case FeBinaryOp::NotEq:
        case FeBinaryOp::Lt:
        case FeBinaryOp::Gt:
        case FeBinaryOp::LtEq:
        case FeBinaryOp::GtEq:
        case FeBinaryOp::And:
        case FeBinaryOp::Or:
        case FeBinaryOp::Not:
            return metal_error("Metal binary executor supports arithmetic ops only");
    }
    return metal_error("Metal binary executor supports arithmetic ops only");
}

std::variant<DeviceTensor, Diagnostic> dispatch_tensor_tensor_binary(
    void* context,
    const DeviceTensor& lhs,
    const DeviceTensor& rhs,
    FeBinaryOp binary_op
) {
    if (lhs.tensor.shape != rhs.tensor.shape) {
        return metal_error("Metal binary tensor shape mismatch");
    }
    auto output = allocate_device_tensor(context, lhs.tensor.shape, lhs.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    auto kernel = binary_kernel_name(binary_op, "tt");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&kernel)) {
        return *diagnostic;
    }
    if (!tysor_metal_dispatch_binary_tt(
            context,
            std::get<std::string>(kernel).c_str(),
            lhs.buffer.get(),
            rhs.buffer.get(),
            out.buffer.get(),
            static_cast<std::uint32_t>(num_elements(lhs.tensor.shape))
        )) {
        return metal_error("Metal tensor/tensor binary dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_tensor_scalar_binary(
    void* context,
    const DeviceTensor& tensor,
    double scalar,
    FeBinaryOp binary_op,
    bool scalar_first
) {
    auto output = allocate_device_tensor(context, tensor.tensor.shape, tensor.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    auto kernel = binary_kernel_name(binary_op, scalar_first ? "st" : "ts");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&kernel)) {
        return *diagnostic;
    }
    const bool ok = scalar_first
        ? tysor_metal_dispatch_binary_st_scalar(
              context,
              std::get<std::string>(kernel).c_str(),
              tensor.buffer.get(),
              out.buffer.get(),
              static_cast<std::uint32_t>(num_elements(tensor.tensor.shape)),
              static_cast<float>(scalar)
          )
        : tysor_metal_dispatch_binary_ts_scalar(
              context,
              std::get<std::string>(kernel).c_str(),
              tensor.buffer.get(),
              out.buffer.get(),
              static_cast<std::uint32_t>(num_elements(tensor.tensor.shape)),
              static_cast<float>(scalar)
          );
    if (!ok) {
        return metal_error("Metal tensor/scalar binary dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_copy(
    void* context,
    const DeviceTensor& input,
    std::vector<std::int64_t> output_shape
) {
    if (num_elements(input.tensor.shape) != num_elements(output_shape)) {
        return metal_error("Metal copy requires matching element counts");
    }
    auto output = allocate_device_tensor(context, std::move(output_shape), input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    if (!tysor_metal_dispatch_unary(
            context,
            "copy_op",
            input.buffer.get(),
            out.buffer.get(),
            static_cast<std::uint32_t>(num_elements(out.tensor.shape))
        )) {
        return metal_error("Metal copy dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_transpose(
    void* context,
    const DeviceTensor& input
) {
    if (input.tensor.shape.size() != 2) {
        return metal_error("Metal transpose currently requires rank-2 tensors");
    }
    auto output = allocate_device_tensor(context, {input.tensor.shape[1], input.tensor.shape[0]}, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    if (!tysor_metal_dispatch_transpose(
            context,
            "transpose_2d",
            input.buffer.get(),
            out.buffer.get(),
            static_cast<std::uint32_t>(input.tensor.shape[0]),
            static_cast<std::uint32_t>(input.tensor.shape[1])
        )) {
        return metal_error("Metal transpose dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_reduction(
    void* context,
    const DeviceTensor& input,
    const std::string& op_name,
    std::optional<std::int64_t> axis
) {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::int32_t axis_value = -1;
    std::uint32_t output_count = 1;
    std::vector<std::int64_t> output_shape{1};
    if (axis) {
        if (input.tensor.shape.size() != 2) {
            return metal_error("Metal " + op_name + " axis currently requires rank-2 tensors");
        }
        rows = static_cast<std::uint32_t>(input.tensor.shape[0]);
        cols = static_cast<std::uint32_t>(input.tensor.shape[1]);
        axis_value = static_cast<std::int32_t>(*axis);
        if (*axis == 0) {
            output_count = cols;
            output_shape = {input.tensor.shape[1]};
        } else if (*axis == 1) {
            output_count = rows;
            output_shape = {input.tensor.shape[0]};
        } else {
            return metal_error("Metal " + op_name + " axis currently supports axis 0 or 1");
        }
    } else {
        rows = static_cast<std::uint32_t>(num_elements(input.tensor.shape));
        cols = 1;
    }
    auto output = allocate_device_tensor(context, output_shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    if (!tysor_metal_dispatch_reduction(
            context,
            op_name == "sum" ? "sum_reduce" : "mean_reduce",
            input.buffer.get(),
            out.buffer.get(),
            rows,
            cols,
            axis_value,
            output_count
        )) {
        return metal_error("Metal reduction dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_linear(
    void* context,
    const LinearClosure& closure,
    const DeviceTensor& input
) {
    if (input.tensor.shape.size() != 2) {
        return metal_error("Metal linear currently requires rank-2 input tensors");
    }
    const std::int64_t inferred_in = input.tensor.shape[1];
    const std::int64_t in_features = closure.in_features.value_or(inferred_in);
    if (inferred_in != in_features) {
        return metal_error("Metal linear input feature size mismatch");
    }
    const std::vector<std::int64_t> output_shape{input.tensor.shape[0], closure.out_features};
    auto output = allocate_device_tensor(context, output_shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));

    SimpleTensor weight = make_linear_weight(in_features, closure.out_features, closure.dtype);
    auto uploaded_weight = upload_tensor(context, std::move(weight));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&uploaded_weight)) {
        return *diagnostic;
    }
    DeviceTensor weight_buffer = std::get<DeviceTensor>(std::move(uploaded_weight));

    std::optional<DeviceTensor> bias_buffer;
    if (closure.with_bias) {
        auto uploaded_bias = upload_tensor(context, make_linear_bias(closure.out_features, closure.dtype));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&uploaded_bias)) {
            return *diagnostic;
        }
        bias_buffer = std::get<DeviceTensor>(std::move(uploaded_bias));
    }

    if (!tysor_metal_dispatch_linear(
            context,
            closure.with_bias ? "linear" : "linear_no_bias",
            input.buffer.get(),
            weight_buffer.buffer.get(),
            bias_buffer ? bias_buffer->buffer.get() : nullptr,
            out.buffer.get(),
            static_cast<std::uint32_t>(input.tensor.shape[0]),
            static_cast<std::uint32_t>(closure.out_features),
            static_cast<std::uint32_t>(in_features),
            closure.with_bias
        )) {
        return metal_error("Metal linear dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_embedding(
    void* context,
    const EmbeddingClosure& closure,
    const DeviceTensor& indices
) {
    std::vector<std::int64_t> output_shape = indices.tensor.shape;
    output_shape.push_back(closure.embedding_dim);
    auto output = allocate_device_tensor(context, output_shape, closure.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));

    auto uploaded_weight = upload_tensor(
        context,
        make_embedding_weight(closure.num_embeddings, closure.embedding_dim, closure.dtype)
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&uploaded_weight)) {
        return *diagnostic;
    }
    DeviceTensor weight = std::get<DeviceTensor>(std::move(uploaded_weight));

    if (!tysor_metal_dispatch_embedding(
            context,
            "embedding",
            indices.buffer.get(),
            weight.buffer.get(),
            out.buffer.get(),
            static_cast<std::uint32_t>(num_elements(indices.tensor.shape)),
            static_cast<std::uint32_t>(closure.embedding_dim)
        )) {
        return metal_error("Metal embedding dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_softmax(
    void* context,
    const DeviceTensor& input
) {
    if (input.tensor.shape.empty()) {
        return metal_error("Metal softmax requires a tensor with at least one dimension");
    }
    const std::int64_t width = input.tensor.shape.back();
    if (width <= 0 || num_elements(input.tensor.shape) % static_cast<std::size_t>(width) != 0) {
        return metal_error("Metal softmax requires the last dimension to divide the data length");
    }
    auto output = allocate_device_tensor(context, input.tensor.shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    const auto rows = static_cast<std::uint32_t>(num_elements(input.tensor.shape) / static_cast<std::size_t>(width));
    if (!tysor_metal_dispatch_softmax(
            context,
            "softmax",
            input.buffer.get(),
            out.buffer.get(),
            rows,
            static_cast<std::uint32_t>(width)
        )) {
        return metal_error("Metal softmax dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_repeat_kv(
    void* context,
    const DeviceTensor& input,
    std::int64_t repeats
) {
    if (repeats <= 0) {
        return metal_error("Metal repeat_kv repeats must be positive");
    }
    if (input.tensor.shape.size() < 2) {
        return metal_error("Metal repeat_kv expects rank >= 2");
    }
    std::vector<std::int64_t> output_shape = input.tensor.shape;
    output_shape[1] *= repeats;
    auto output = allocate_device_tensor(context, output_shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    const auto inner = static_cast<std::uint32_t>(
        std::accumulate(input.tensor.shape.begin() + 2, input.tensor.shape.end(), std::int64_t{1}, std::multiplies<>())
    );
    if (!tysor_metal_dispatch_repeat_kv(
            context,
            "repeat_kv",
            input.buffer.get(),
            out.buffer.get(),
            static_cast<std::uint32_t>(input.tensor.shape[0]),
            static_cast<std::uint32_t>(output_shape[1]),
            inner,
            static_cast<std::uint32_t>(repeats)
        )) {
        return metal_error("Metal repeat_kv dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_causal_mask(
    void* context,
    const DeviceTensor& input
) {
    if (input.tensor.shape.size() < 2) {
        return metal_error("Metal causal_mask expects rank >= 2");
    }
    auto output = allocate_device_tensor(context, input.tensor.shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    const auto q = static_cast<std::uint32_t>(input.tensor.shape[input.tensor.shape.size() - 2]);
    const auto k = static_cast<std::uint32_t>(input.tensor.shape.back());
    const auto outer = static_cast<std::uint32_t>(
        std::accumulate(input.tensor.shape.begin(), input.tensor.shape.end() - 2, std::int64_t{1}, std::multiplies<>())
    );
    if (!tysor_metal_dispatch_causal_mask(
            context,
            "causal_mask",
            input.buffer.get(),
            out.buffer.get(),
            outer,
            q,
            k
        )) {
        return metal_error("Metal causal_mask dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_rope(
    void* context,
    const DeviceTensor& input,
    std::int64_t head_dim,
    double theta
) {
    if (input.tensor.shape.size() < 2) {
        return metal_error("Metal rope expects rank >= 2");
    }
    if (head_dim % 2 != 0) {
        return metal_error("Metal rope requires an even head_dim");
    }
    if (input.tensor.shape.back() != head_dim) {
        return metal_error("Metal rope head_dim mismatch");
    }
    auto output = allocate_device_tensor(context, input.tensor.shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    const auto seq_len = static_cast<std::uint32_t>(input.tensor.shape[input.tensor.shape.size() - 2]);
    const auto outer = static_cast<std::uint32_t>(
        std::accumulate(input.tensor.shape.begin(), input.tensor.shape.end() - 2, std::int64_t{1}, std::multiplies<>())
    );
    if (!tysor_metal_dispatch_rope_runtime(
            context,
            "rope",
            input.buffer.get(),
            out.buffer.get(),
            outer,
            seq_len,
            static_cast<std::uint32_t>(head_dim / 2),
            static_cast<float>(theta)
        )) {
        return metal_error("Metal rope dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_rms_norm(
    void* context,
    const DeviceTensor& input,
    std::int64_t hidden_size
) {
    if (input.tensor.shape.empty() || input.tensor.shape.back() != hidden_size) {
        return metal_error("Metal rms_norm hidden size mismatch");
    }
    if (hidden_size <= 0) {
        return metal_error("Metal rms_norm hidden size must be positive");
    }
    const auto width = static_cast<std::uint32_t>(hidden_size);
    const auto rows = static_cast<std::uint32_t>(num_elements(input.tensor.shape) / static_cast<std::size_t>(hidden_size));
    auto output = allocate_device_tensor(context, input.tensor.shape, input.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    if (!tysor_metal_dispatch_rms_norm(
            context,
            "rms_norm",
            input.buffer.get(),
            out.buffer.get(),
            rows,
            width
        )) {
        return metal_error("Metal rms_norm dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_cross_entropy(
    void* context,
    const DeviceTensor& logits,
    const DeviceTensor& target
) {
    if (logits.tensor.shape.size() != 2 || target.tensor.shape.size() != 2) {
        return metal_error("Metal cross_entropy requires rank-2 tensors");
    }
    if (logits.tensor.shape != target.tensor.shape) {
        return metal_error("Metal cross_entropy requires logits and target to have identical shapes");
    }
    const auto rows = static_cast<std::uint32_t>(logits.tensor.shape[0]);
    const auto width = static_cast<std::uint32_t>(logits.tensor.shape[1]);
    auto output = allocate_device_tensor(context, {logits.tensor.shape[0], 1}, logits.tensor.dtype);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
        return *diagnostic;
    }
    DeviceTensor out = std::get<DeviceTensor>(std::move(output));
    if (!tysor_metal_dispatch_cross_entropy(
            context,
            "cross_entropy",
            logits.buffer.get(),
            target.buffer.get(),
            out.buffer.get(),
            rows,
            width
        )) {
        return metal_error("Metal cross_entropy dispatch failed: " + last_metal_error());
    }
    return out;
}

std::variant<DeviceTensor, Diagnostic> dispatch_activation(
    void* context,
    const ActivationClosure& closure,
    const DeviceTensor& input
) {
    if (closure.op == "SiLU") {
        return dispatch_unary(context, input, "silu");
    }
    if (closure.op == "GELU") {
        return dispatch_unary(context, input, "gelu");
    }
    if (closure.op == "Tanh") {
        return dispatch_unary(context, input, "tanh_op");
    }
    if (closure.op == "Sigmoid") {
        return dispatch_unary(context, input, "sigmoid");
    }
    if (closure.op == "Softmax") {
        return dispatch_softmax(context, input);
    }
    if (closure.op == "Dropout") {
        if (closure.probability < 0.0 || closure.probability >= 1.0) {
            return metal_error("dropout probability must be in [0, 1)");
        }
        return dispatch_tensor_scalar_binary(context, input, 1.0 - closure.probability, FeBinaryOp::Mul, false);
    }
    return metal_error("Unsupported Metal activation '" + closure.op + "'");
}

// Checks if the planned ops are supported by the current Metal backend
std::optional<Diagnostic> validate_supported_plan(const ExecutionPlan& plan) {
    auto producer_for = [&](std::size_t value_id) -> const PlanOp* {
        auto found = std::find_if(plan.ops.begin(), plan.ops.end(), [&](const PlanOp& item) {
            return item.output == value_id;
        });
        return found == plan.ops.end() ? nullptr : &*found;
    };

    for (std::size_t index = 0; index < plan.ops.size(); ++index) {
        const PlanOp& op = plan.ops[index];
        bool supported =
            op.kind == PlanOpKind::Constant ||
            op.kind == PlanOpKind::Binary ||
            (op.kind == PlanOpKind::PrimitiveCall && (op.op == "matmul" || op.op == "relu" || op.op == "scale")) ||
            (op.kind == PlanOpKind::LibraryCtor && (op.op == "linear" || op.op == "Embedding" || is_activation_constructor(op.op))) ||
            (op.kind == PlanOpKind::LibraryCall &&
             (op.op == "sqrt" || op.op == "rsqrt" || op.op == "reshape" || op.op == "transpose" ||
              op.op == "sum" || op.op == "mean" || op.op == "rms_norm" || op.op == "cross_entropy" ||
              op.op == "repeat_kv" || op.op == "flatten_heads" || op.op == "causal_mask" || op.op == "rope"));
        if (op.kind == PlanOpKind::Apply && !op.inputs.empty()) {
            const PlanOp* callee = producer_for(op.inputs[0]);
            supported = callee != nullptr &&
                callee->kind == PlanOpKind::LibraryCtor &&
                (callee->op == "linear" || callee->op == "Embedding" || is_activation_constructor(callee->op));
        }
        if (!supported) {
            return metal_error(
                "Metal executor does not support planned op #" + std::to_string(index) +
                " '" + op.op + "' yet"
            );
        }
    }
    return std::nullopt;
}

// Central interpreter loop for the execution plan using the Metal backend
std::variant<GraphExecutionResult, Diagnostic> execute_plan_native(
    const ExecutionPlan& plan,
    const GraphExecutorOptions& options
) {
    if (plan.backend != BackendKind::Metal) {
        return metal_error("Metal executor received a non-Metal execution plan");
    }
    if (auto diagnostic = validate_supported_plan(plan)) {
        return *diagnostic;
    }

    // Initialize the shared Metal state via Objective-C bridge, loading kernel shaders
    std::unique_ptr<void, decltype(&tysor_metal_context_free)> context(
        tysor_metal_context_new(metal_source()),
        tysor_metal_context_free
    );
    if (!context) {
        return metal_error("Metal context creation failed: " + last_metal_error());
    }

    // Storage for intermediate graph variables mapping value ID -> data (host or device)
    std::map<std::size_t, MetalValue> values;
    std::map<std::size_t, GraphRuntimeValue> outputs;

    // Process the linear sequence of plan steps
    for (const auto& step : plan.steps) {
        switch (step.kind) {
            case PlanStepKind::AllocateHostValue: {
                // Initialize synthetic tensor inputs based on CLI flags
                const PlanValue& value = plan.values[step.value_id];
                if (value.is_parameter) {
                    if (value.type.kind != FeTypeKind::Tensor) {
                        return metal_error("Metal executor currently supports tensor parameters only");
                    }
                    auto tensor = make_tensor_argument(value, options);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&tensor)) {
                        return *diagnostic;
                    }
                    values[value.id] = HostValue{std::get<SimpleTensor>(std::move(tensor))};
                }
                break;
            }
            case PlanStepKind::UploadToDevice: {
                // Transfer a tensor resident in host memory to GPU VRAM
                auto found = values.find(step.value_id);
                if (found == values.end()) {
                    return metal_error("Metal upload step references an uninitialized host value");
                }
                auto* host = std::get_if<HostValue>(&found->second);
                if (host == nullptr) {
                    break;
                }
                auto* tensor = std::get_if<SimpleTensor>(&host->value);
                if (tensor == nullptr) {
                    return metal_error("Metal upload currently supports tensor values only");
                }
                auto uploaded = upload_tensor(context.get(), *tensor);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&uploaded)) {
                    return *diagnostic;
                }
                values[step.value_id] = std::get<DeviceTensor>(std::move(uploaded));
                break;
            }
            case PlanStepKind::AllocateDeviceValue:
                break;
            case PlanStepKind::DispatchDeviceOp: {
                if (!step.op_index) {
                    return metal_error("Metal dispatch step is missing an op index");
                }
                const PlanOp& op = plan.ops[*step.op_index];
                if (op.kind == PlanOpKind::Constant) {
                    auto host = constant_to_host_value(op.constant);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&host)) {
                        return *diagnostic;
                    }
                    values[op.output] = std::get<HostValue>(std::move(host));
                    break;
                }
                if (op.kind == PlanOpKind::LibraryCtor && op.op == "linear") {
                    auto closure = build_linear_closure(op, plan.values[op.output], values);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&closure)) {
                        return *diagnostic;
                    }
                    values[op.output] = HostValue{MetalHostValue{std::get<LinearClosure>(std::move(closure))}};
                    break;
                }
                if (op.kind == PlanOpKind::LibraryCtor && op.op == "Embedding") {
                    auto closure = build_embedding_closure(op, plan.values[op.output], values);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&closure)) {
                        return *diagnostic;
                    }
                    values[op.output] = HostValue{MetalHostValue{std::get<EmbeddingClosure>(std::move(closure))}};
                    break;
                }
                if (op.kind == PlanOpKind::LibraryCtor && is_activation_constructor(op.op)) {
                    auto closure = build_activation_closure(op, values);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&closure)) {
                        return *diagnostic;
                    }
                    values[op.output] = HostValue{MetalHostValue{std::get<ActivationClosure>(std::move(closure))}};
                    break;
                }
                std::variant<DeviceTensor, Diagnostic> result;
                if (op.kind == PlanOpKind::PrimitiveCall && op.op == "matmul") {
                    DeviceTensor* lhs = require_device_tensor(values, op.inputs[0], "matmul lhs");
                    DeviceTensor* rhs = require_device_tensor(values, op.inputs[1], "matmul rhs");
                    if (lhs == nullptr || rhs == nullptr) {
                        return metal_error("Metal matmul requires device tensor inputs");
                    }
                    result = dispatch_matmul(context.get(), *lhs, *rhs);
                } else if (op.kind == PlanOpKind::Binary) {
                    DeviceTensor* lhs_tensor = require_device_tensor(values, op.inputs[0], "binary lhs");
                    DeviceTensor* rhs_tensor = require_device_tensor(values, op.inputs[1], "binary rhs");
                    if (lhs_tensor != nullptr && rhs_tensor != nullptr) {
                        result = dispatch_tensor_tensor_binary(context.get(), *lhs_tensor, *rhs_tensor, op.binary_op);
                    } else if (lhs_tensor != nullptr) {
                        auto scalar = require_host_number(values, op.inputs[1], "binary rhs");
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) {
                            return *diagnostic;
                        }
                        result = dispatch_tensor_scalar_binary(context.get(), *lhs_tensor, std::get<double>(scalar), op.binary_op, false);
                    } else if (rhs_tensor != nullptr) {
                        auto scalar = require_host_number(values, op.inputs[0], "binary lhs");
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) {
                            return *diagnostic;
                        }
                        result = dispatch_tensor_scalar_binary(context.get(), *rhs_tensor, std::get<double>(scalar), op.binary_op, true);
                    } else {
                        return metal_error("Metal does not support scalar-scalar binary ops");
                    }
                } else if (op.kind == PlanOpKind::PrimitiveCall && op.op == "relu") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "relu input");
                    if (input == nullptr) {
                        return metal_error("Metal relu requires a device tensor input");
                    }
                    result = dispatch_unary(context.get(), *input, "relu");
                } else if (op.kind == PlanOpKind::PrimitiveCall && op.op == "scale") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "scale input");
                    if (input == nullptr) {
                        return metal_error("Metal scale requires a device tensor input");
                    }
                    auto scalar = require_host_number(values, op.inputs[1], "scale");
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) {
                        return *diagnostic;
                    }
                    result = dispatch_tensor_scalar_binary(context.get(), *input, std::get<double>(scalar), FeBinaryOp::Mul, false);
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "sqrt") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "sqrt input");
                    if (input == nullptr) {
                        return metal_error("Metal sqrt requires a device tensor input");
                    }
                    result = dispatch_unary(context.get(), *input, "sqrt_op");
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "rsqrt") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "rsqrt input");
                    if (input == nullptr) {
                        return metal_error("Metal rsqrt requires a device tensor input");
                    }
                    result = dispatch_unary(context.get(), *input, "rsqrt_op");
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "reshape") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "reshape input");
                    if (input == nullptr) {
                        return metal_error("Metal reshape requires a device tensor input");
                    }
                    auto shape = infer_output_shape(op, values);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&shape)) {
                        return *diagnostic;
                    }
                    result = dispatch_copy(context.get(), *input, std::get<std::vector<std::int64_t>>(std::move(shape)));
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "transpose") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "transpose input");
                    if (input == nullptr) {
                        return metal_error("Metal transpose requires a device tensor input");
                    }
                    result = dispatch_transpose(context.get(), *input);
                } else if (op.kind == PlanOpKind::LibraryCall && (op.op == "sum" || op.op == "mean")) {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], op.op + " input");
                    if (input == nullptr) {
                        return metal_error("Metal " + op.op + " requires a device tensor input");
                    }
                    std::optional<std::int64_t> axis;
                    if (op.inputs.size() > 1) {
                        auto axis_value = require_host_int(values, op.inputs[1], op.op + " axis");
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&axis_value)) {
                            return *diagnostic;
                        }
                        axis = std::get<std::int64_t>(axis_value);
                    }
                    result = dispatch_reduction(context.get(), *input, op.op, axis);
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "rms_norm") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "rms_norm input");
                    if (input == nullptr) {
                        return metal_error("Metal rms_norm requires a device tensor input");
                    }
                    auto hidden_size = require_host_int(values, op.inputs[1], "rms_norm hidden size");
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&hidden_size)) {
                        return *diagnostic;
                    }
                    result = dispatch_rms_norm(context.get(), *input, std::get<std::int64_t>(hidden_size));
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "cross_entropy") {
                    DeviceTensor* logits = require_device_tensor(values, op.inputs[0], "cross_entropy logits");
                    DeviceTensor* target = require_device_tensor(values, op.inputs[1], "cross_entropy target");
                    if (logits == nullptr || target == nullptr) {
                        return metal_error("Metal cross_entropy requires device tensor inputs");
                    }
                    result = dispatch_cross_entropy(context.get(), *logits, *target);
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "repeat_kv") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "repeat_kv input");
                    if (input == nullptr) {
                        return metal_error("Metal repeat_kv requires a device tensor input");
                    }
                    auto repeats = require_host_int(values, op.inputs[1], "repeat_kv repeats");
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&repeats)) {
                        return *diagnostic;
                    }
                    result = dispatch_repeat_kv(context.get(), *input, std::get<std::int64_t>(repeats));
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "flatten_heads") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "flatten_heads input");
                    if (input == nullptr) {
                        return metal_error("Metal flatten_heads requires a device tensor input");
                    }
                    auto shape = infer_output_shape(op, values);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&shape)) {
                        return *diagnostic;
                    }
                    result = dispatch_copy(context.get(), *input, std::get<std::vector<std::int64_t>>(std::move(shape)));
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "causal_mask") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "causal_mask input");
                    if (input == nullptr) {
                        return metal_error("Metal causal_mask requires a device tensor input");
                    }
                    result = dispatch_causal_mask(context.get(), *input);
                } else if (op.kind == PlanOpKind::LibraryCall && op.op == "rope") {
                    DeviceTensor* input = require_device_tensor(values, op.inputs[0], "rope input");
                    if (input == nullptr) {
                        return metal_error("Metal rope requires a device tensor input");
                    }
                    auto head_dim = require_host_int(values, op.inputs[1], "rope head_dim");
                    auto theta = require_host_number(values, op.inputs[2], "rope theta");
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&head_dim)) {
                        return *diagnostic;
                    }
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&theta)) {
                        return *diagnostic;
                    }
                    result = dispatch_rope(context.get(), *input, std::get<std::int64_t>(head_dim), std::get<double>(theta));
                } else if (op.kind == PlanOpKind::Apply) {
                    HostValue* callee = require_host_value(values, op.inputs[0]);
                    DeviceTensor* input = require_device_tensor(values, op.inputs[1], "apply input");
                    if (callee == nullptr || input == nullptr) {
                        return metal_error("Metal apply requires a host closure and device tensor input");
                    }
                    if (const auto* closure = std::get_if<LinearClosure>(&callee->value)) {
                        result = dispatch_linear(context.get(), *closure, *input);
                    } else if (const auto* embedding = std::get_if<EmbeddingClosure>(&callee->value)) {
                        result = dispatch_embedding(context.get(), *embedding, *input);
                    } else if (const auto* activation = std::get_if<ActivationClosure>(&callee->value)) {
                        result = dispatch_activation(context.get(), *activation, *input);
                    } else {
                        return metal_error("Metal Apply currently supports linear, embedding, and activation closures only");
                    }
                } else {
                    return metal_error("Metal executor does not support planned op '" + op.op + "' yet");
                }
                if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                    return *diagnostic;
                }
                values[op.output] = std::get<DeviceTensor>(std::move(result));
                break;
            }
            case PlanStepKind::DownloadToHost: {
                // Read a buffer back from GPU memory into standard CPU memory
                DeviceTensor* device = require_device_tensor(values, step.value_id, "download value");
                if (device == nullptr) {
                    return metal_error("Metal download requires a device tensor value");
                }
                SimpleTensor host = device->tensor;
                host.data.assign(num_elements(host.shape), 0.0F);
                if (!tysor_metal_buffer_read(device->buffer.get(), host.data.data(), host.data.size())) {
                    return metal_error("Metal readback failed: " + last_metal_error());
                }
                values[step.value_id] = HostValue{std::move(host)};
                break;
            }
            case PlanStepKind::MaterializeOutput: {
                auto found = values.find(step.value_id);
                if (found == values.end()) {
                    return metal_error("Metal output step references an uninitialized value");
                }
                auto* host = std::get_if<HostValue>(&found->second);
                if (host == nullptr) {
                    return metal_error("Metal output was not downloaded to host");
                }
                auto output = host_to_graph_value(*host);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) {
                    return *diagnostic;
                }
                outputs[step.value_id] = std::get<GraphRuntimeValue>(std::move(output));
                break;
            }
            case PlanStepKind::ExecuteOp:
                return metal_error("Metal executor cannot run host ExecuteOp steps");
        }
    }

    return GraphExecutionResult{{}, std::move(outputs)};
}
#endif

} // namespace

GraphExecutionResultVariant execute_metal_plan_module(
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
) {
    const ExecutionPlan* plan = find_plan(module, entry);
    if (plan == nullptr) {
        return metal_error("Entry function '" + entry + "' not found in execution plan module");
    }
#if defined(__APPLE__)
    return execute_plan_native(*plan, options);
#else
    (void)options;
    return metal_error("Native Metal runtime is only available on macOS");
#endif
}

std::variant<std::string, Diagnostic> probe_native_metal_device() {
#if defined(__APPLE__)
    if (!tysor_metal_probe_device()) {
        return metal_error("Native Metal device probe failed: " + last_metal_error());
    }
    const char* report = tysor_metal_device_report();
    return std::string(report == nullptr || std::string(report).empty() ? "default_device=yes" : report);
#else
    return metal_error("Native Metal runtime is only available on macOS");
#endif
}
