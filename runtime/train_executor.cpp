#include "train_executor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

Diagnostic train_error(std::string message) {
    return Diagnostic::error("runtime", "R0004", std::move(message));
}

struct LinearSpec {
    LinearClosure closure;
    std::string weight_name;
    std::optional<std::string> bias_name;
    std::optional<GraphTensorType> weight_type;
    std::optional<GraphTensorType> bias_type;
};

struct EmbeddingSpec {
    EmbeddingClosure closure;
    std::string weight_name;
    std::optional<GraphTensorType> weight_type;
};

struct ActivationSpec {
    std::string op;
    double probability = 0.0;
};

using TrainValue = std::variant<std::int64_t, double, bool, SimpleTensor, LinearSpec, EmbeddingSpec, ActivationSpec>;

struct TrainExecutionResult {
    std::map<std::size_t, TrainValue> values;
};

const ExecutionPlan* find_plan(const PlanModule& module, const std::string& entry) {
    auto found = std::find_if(module.plans.begin(), module.plans.end(), [&](const ExecutionPlan& plan) {
        return plan.name == entry;
    });
    return found == module.plans.end() ? nullptr : &*found;
}

std::optional<std::string> find_skipped_plan_reason(const PlanModule& module, const std::string& entry) {
    auto found = std::find_if(module.skipped.begin(), module.skipped.end(), [&](const PlanBuildSkipped& skipped) {
        return skipped.function_name == entry;
    });
    if (found == module.skipped.end()) {
        return std::nullopt;
    }
    return found->reason;
}

const FeExecutionRun* find_train_run(const LoweredModule& lowered, const std::string& entry) {
    if (!lowered.execution_plan) {
        return nullptr;
    }
    auto found = std::find_if(lowered.execution_plan->runs.begin(), lowered.execution_plan->runs.end(), [&](const FeExecutionRun& run) {
        return run.model_name == entry;
    });
    return found == lowered.execution_plan->runs.end() ? nullptr : &*found;
}

std::string linear_weight_name(std::size_t output_id) {
    return "linear_" + std::to_string(output_id) + "_weight";
}

std::string linear_bias_name(std::size_t output_id) {
    return "linear_" + std::to_string(output_id) + "_bias";
}

std::string embedding_weight_name(std::size_t output_id) {
    return "embedding_" + std::to_string(output_id) + "_weight";
}

const PlanParameter* find_plan_parameter(
    const ExecutionPlan& plan,
    std::size_t owner_value,
    const std::string& role
) {
    auto found = std::find_if(plan.parameters.begin(), plan.parameters.end(), [&](const PlanParameter& parameter) {
        return parameter.owner_value == owner_value && parameter.role == role;
    });
    return found == plan.parameters.end() ? nullptr : &*found;
}

const PlanParameter* find_plan_parameter_by_value(const ExecutionPlan& plan, std::size_t value_id) {
    auto found = std::find_if(plan.parameters.begin(), plan.parameters.end(), [&](const PlanParameter& parameter) {
        return parameter.value_id == value_id;
    });
    return found == plan.parameters.end() ? nullptr : &*found;
}

std::optional<std::int64_t> known_graph_dim(const GraphTensorType& type, std::size_t index) {
    if (!type.has_known_rank || index >= type.shape.size()) {
        return std::nullopt;
    }
    const GraphDim& dim = type.shape[index];
    if (dim.kind != GraphDimKind::Known) {
        return std::nullopt;
    }
    return dim.value;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool is_trailing_vector_broadcast(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    return lhs.shape.size() == 2 &&
           rhs.shape.size() == 1 &&
           lhs.shape[1] == rhs.shape[0];
}

std::variant<SimpleTensor, Diagnostic> synthesize_model_parameter(
    const PlanValue& value,
    const PlanParameter& parameter
) {
    const std::string dtype = parameter.tensor_type.dtype.value_or(value.type.tensor_dtype.value_or("float32"));
    if (starts_with(parameter.name, "linear_") && parameter.role == "weight") {
        auto in_features = known_graph_dim(parameter.tensor_type, 0);
        auto out_features = known_graph_dim(parameter.tensor_type, 1);
        if (!in_features || !out_features) {
            return train_error("Linear weight parameter '" + parameter.name + "' requires known rank-2 shape");
        }
        return make_linear_weight(*in_features, *out_features, dtype);
    }
    if (starts_with(parameter.name, "linear_") && parameter.role == "bias") {
        auto out_features = known_graph_dim(parameter.tensor_type, 0);
        if (!out_features) {
            return train_error("Linear bias parameter '" + parameter.name + "' requires known rank-1 shape");
        }
        return make_linear_bias(*out_features, dtype);
    }
    if (starts_with(parameter.name, "embedding_") && parameter.role == "weight") {
        auto num_embeddings = known_graph_dim(parameter.tensor_type, 0);
        auto embedding_dim = known_graph_dim(parameter.tensor_type, 1);
        if (!num_embeddings || !embedding_dim) {
            return train_error("Embedding weight parameter '" + parameter.name + "' requires known rank-2 shape");
        }
        return make_embedding_weight(*num_embeddings, *embedding_dim, dtype);
    }
    return train_error("Unsupported model parameter '" + parameter.name + "'");
}

std::variant<SimpleTensor, Diagnostic> materialize_model_parameter(
    const PlanValue& value,
    const PlanParameter& parameter,
    std::map<std::string, SimpleTensor>& parameters
) {
    auto existing = parameters.find(parameter.name);
    if (existing != parameters.end()) {
        return existing->second;
    }
    auto synthesized = synthesize_model_parameter(value, parameter);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&synthesized)) {
        return *diagnostic;
    }
    auto inserted = parameters.emplace(parameter.name, std::get<SimpleTensor>(std::move(synthesized)));
    return inserted.first->second;
}

std::variant<double, Diagnostic> require_number(const TrainValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return static_cast<double>(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item ? 1.0 : 0.0;
    }
    return train_error("Train executor expected scalar value");
}

std::variant<std::int64_t, Diagnostic> require_int(const TrainValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return *item;
    }
    return train_error("Train executor expected integer value");
}

std::variant<const SimpleTensor*, Diagnostic> require_tensor(const TrainValue& value) {
    if (const auto* item = std::get_if<SimpleTensor>(&value)) {
        return item;
    }
    return train_error("Train executor expected tensor value");
}

std::variant<const SimpleTensor*, Diagnostic> require_tensor(
    const TrainExecutionResult& execution,
    std::size_t id,
    const std::string& label
) {
    auto found = execution.values.find(id);
    if (found == execution.values.end()) {
        return train_error("Missing " + label);
    }
    return require_tensor(found->second);
}

std::variant<double, Diagnostic> require_number(
    const TrainExecutionResult& execution,
    std::size_t id,
    const std::string& label
) {
    auto found = execution.values.find(id);
    if (found == execution.values.end()) {
        return train_error("Missing " + label);
    }
    return require_number(found->second);
}

std::variant<std::int64_t, Diagnostic> require_int(
    const TrainExecutionResult& execution,
    std::size_t id,
    const std::string& label
) {
    auto found = execution.values.find(id);
    if (found == execution.values.end()) {
        return train_error("Missing " + label);
    }
    return require_int(found->second);
}

std::variant<TrainValue, Diagnostic> constant_to_train_value(const FeValue& value) {
    return std::visit(
        [](const auto& inner) -> std::variant<TrainValue, Diagnostic> {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double> || std::is_same_v<T, bool>) {
                return TrainValue{inner};
            } else {
                return train_error("Unsupported train graph constant");
            }
        },
        value.value
    );
}

std::variant<SimpleTensor, Diagnostic> apply_linear_with_parameters(
    const SimpleTensor& input,
    const SimpleTensor& weight,
    const SimpleTensor* bias
) {
    auto multiplied = matmul(input, weight);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&multiplied)) {
        return *diagnostic;
    }
    SimpleTensor output = std::get<SimpleTensor>(std::move(multiplied));
    if (bias != nullptr) {
        if (bias->shape.size() != 1 || output.shape.size() != 2 || bias->shape[0] != output.shape[1]) {
            return train_error("linear bias shape mismatch");
        }
        const auto rows = static_cast<std::size_t>(output.shape[0]);
        const auto cols = static_cast<std::size_t>(output.shape[1]);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                output.data[row * cols + col] += bias->data[col];
            }
        }
    }
    return output;
}

SimpleTensor zeros_like(const SimpleTensor& tensor) {
    return SimpleTensor{tensor.shape, std::vector<float>(tensor.data.size(), 0.0F), tensor.dtype};
}

SimpleTensor ones_like(const SimpleTensor& tensor) {
    return SimpleTensor{tensor.shape, std::vector<float>(tensor.data.size(), 1.0F), tensor.dtype};
}

SimpleTensor negate(const SimpleTensor& tensor) {
    SimpleTensor output = tensor;
    for (auto& value : output.data) {
        value = -value;
    }
    return output;
}

std::optional<Diagnostic> add_in_place(SimpleTensor& dst, const SimpleTensor& src) {
    if (dst.shape != src.shape || dst.data.size() != src.data.size()) {
        return train_error("tensor shape mismatch");
    }
    for (std::size_t index = 0; index < dst.data.size(); ++index) {
        dst.data[index] += src.data[index];
    }
    return std::nullopt;
}

std::optional<Diagnostic> accumulate_value(
    std::map<std::size_t, SimpleTensor>& gradients,
    std::size_t id,
    const SimpleTensor& grad
) {
    auto found = gradients.find(id);
    if (found == gradients.end()) {
        gradients.emplace(id, grad);
        return std::nullopt;
    }
    return add_in_place(found->second, grad);
}

std::optional<Diagnostic> accumulate_parameter(
    std::map<std::string, SimpleTensor>& gradients,
    const std::string& name,
    const SimpleTensor& grad
) {
    auto found = gradients.find(name);
    if (found == gradients.end()) {
        gradients.emplace(name, grad);
        return std::nullopt;
    }
    return add_in_place(found->second, grad);
}

std::variant<SimpleTensor, Diagnostic> transpose_checked(const SimpleTensor& tensor) {
    auto result = transpose_2d(tensor);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        return *diagnostic;
    }
    return std::get<SimpleTensor>(std::move(result));
}

std::variant<SimpleTensor, Diagnostic> matmul_checked(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    auto result = matmul(lhs, rhs);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        return *diagnostic;
    }
    return std::get<SimpleTensor>(std::move(result));
}

std::variant<SimpleTensor, Diagnostic> backward_linear_input(const SimpleTensor& grad, const SimpleTensor& weight) {
    auto transposed = transpose_checked(weight);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&transposed)) {
        return *diagnostic;
    }
    return matmul_checked(grad, std::get<SimpleTensor>(transposed));
}

std::variant<SimpleTensor, Diagnostic> backward_linear_weight(const SimpleTensor& grad, const SimpleTensor& input) {
    auto transposed = transpose_checked(input);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&transposed)) {
        return *diagnostic;
    }
    return matmul_checked(std::get<SimpleTensor>(transposed), grad);
}

std::variant<SimpleTensor, Diagnostic> backward_linear_bias(const SimpleTensor& grad) {
    if (grad.shape.size() != 2) {
        return train_error("backward_linear_bias currently requires rank-2 gradients");
    }
    const auto rows = static_cast<std::size_t>(grad.shape[0]);
    const auto cols = static_cast<std::size_t>(grad.shape[1]);
    SimpleTensor bias{{static_cast<std::int64_t>(cols)}, std::vector<float>(cols, 0.0F), grad.dtype};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            bias.data[col] += grad.data[row * cols + col];
        }
    }
    return bias;
}

std::variant<SimpleTensor, Diagnostic> backward_cross_entropy_logits(const SimpleTensor& logits, const SimpleTensor& target) {
    if (logits.shape != target.shape) {
        return train_error("backward_cross_entropy_logits shape mismatch");
    }
    auto probabilities = apply_softmax(logits);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&probabilities)) {
        return *diagnostic;
    }
    SimpleTensor grad = std::get<SimpleTensor>(std::move(probabilities));
    for (std::size_t index = 0; index < grad.data.size(); ++index) {
        grad.data[index] -= target.data[index];
    }
    return grad;
}

std::variant<SimpleTensor, Diagnostic> backward_silu(const SimpleTensor& input, const SimpleTensor& grad) {
    if (input.shape != grad.shape) {
        return train_error("backward_silu shape mismatch");
    }
    SimpleTensor output = input;
    for (std::size_t index = 0; index < output.data.size(); ++index) {
        const float x = input.data[index];
        const float sigmoid = 1.0F / (1.0F + std::exp(-x));
        output.data[index] = grad.data[index] * (sigmoid + x * sigmoid * (1.0F - sigmoid));
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> backward_gelu(const SimpleTensor& input, const SimpleTensor& grad) {
    if (input.shape != grad.shape) {
        return train_error("backward_gelu shape mismatch");
    }
    SimpleTensor output = input;
    for (std::size_t index = 0; index < output.data.size(); ++index) {
        const float x = input.data[index];
        const float inner = 0.7978846F * (x + 0.044715F * x * x * x);
        const float tanh_inner = std::tanh(inner);
        const float sech_sq = 1.0F - tanh_inner * tanh_inner;
        const float inner_grad = 0.7978846F * (1.0F + 3.0F * 0.044715F * x * x);
        output.data[index] = grad.data[index] * (0.5F * (1.0F + tanh_inner) + 0.5F * x * sech_sq * inner_grad);
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> backward_tanh(const SimpleTensor& input, const SimpleTensor& grad) {
    if (input.shape != grad.shape) {
        return train_error("backward_tanh shape mismatch");
    }
    SimpleTensor output = input;
    for (std::size_t index = 0; index < output.data.size(); ++index) {
        const float y = std::tanh(input.data[index]);
        output.data[index] = grad.data[index] * (1.0F - y * y);
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> backward_sigmoid(const SimpleTensor& input, const SimpleTensor& grad) {
    if (input.shape != grad.shape) {
        return train_error("backward_sigmoid shape mismatch");
    }
    SimpleTensor output = input;
    for (std::size_t index = 0; index < output.data.size(); ++index) {
        const float y = 1.0F / (1.0F + std::exp(-input.data[index]));
        output.data[index] = grad.data[index] * y * (1.0F - y);
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> backward_softmax(const SimpleTensor& output, const SimpleTensor& grad) {
    if (output.shape != grad.shape) {
        return train_error("backward_softmax shape mismatch");
    }
    SimpleTensor result = output;
    for (std::size_t index = 0; index < result.data.size(); ++index) {
        result.data[index] = output.data[index] * (1.0F - output.data[index]) * grad.data[index];
    }
    return result;
}

std::variant<SimpleTensor, Diagnostic> backward_rms_norm_input(
    const SimpleTensor& input,
    const SimpleTensor& grad
) {
    if (input.shape != grad.shape) {
        return train_error("backward_rms_norm_input shape mismatch");
    }
    return grad;
}

std::variant<SimpleTensor, Diagnostic> backward_sum_input(const SimpleTensor& grad, const SimpleTensor& input) {
    if (grad.data.size() != 1) {
        return train_error("backward_sum_input expects a single gradient value");
    }
    return SimpleTensor{input.shape, std::vector<float>(input.data.size(), grad.data[0]), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> backward_mean_input(const SimpleTensor& grad, const SimpleTensor& input) {
    auto summed = backward_sum_input(grad, input);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&summed)) {
        return *diagnostic;
    }
    SimpleTensor output = std::get<SimpleTensor>(std::move(summed));
    for (auto& value : output.data) {
        value /= static_cast<float>(input.data.size());
    }
    return output;
}

std::variant<SimpleTensor, Diagnostic> backward_sum_axis_input(const SimpleTensor& grad, const SimpleTensor& input, std::int64_t axis) {
    if (input.shape.size() != 2) {
        return train_error("backward_sum_axis_input expects rank-2 input");
    }
    const auto rows = static_cast<std::size_t>(input.shape[0]);
    const auto cols = static_cast<std::size_t>(input.shape[1]);
    SimpleTensor output = zeros_like(input);
    if (axis == 0) {
        if (grad.data.size() != cols) return train_error("backward_sum_axis_input axis 0 gradient shape mismatch");
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                output.data[row * cols + col] = grad.data[col];
            }
        }
        return output;
    }
    if (axis == 1) {
        if (grad.data.size() != rows) return train_error("backward_sum_axis_input axis 1 gradient shape mismatch");
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                output.data[row * cols + col] = grad.data[row];
            }
        }
        return output;
    }
    return train_error("backward_sum_axis_input supports axis 0 or 1");
}

std::variant<SimpleTensor, Diagnostic> backward_mean_axis_input(const SimpleTensor& grad, const SimpleTensor& input, std::int64_t axis) {
    auto summed = backward_sum_axis_input(grad, input, axis);
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

std::variant<SimpleTensor, Diagnostic> backward_reshape_input(const SimpleTensor& grad, const SimpleTensor& input) {
    if (grad.data.size() != input.data.size()) {
        return train_error("reshape backward requires matching element counts");
    }
    return SimpleTensor{input.shape, grad.data, grad.dtype};
}

std::variant<SimpleTensor, Diagnostic> backward_repeat_kv_input(const SimpleTensor& grad, const SimpleTensor& input, std::int64_t repeats) {
    if (repeats <= 0) {
        return train_error("repeat_kv backward requires repeats > 0");
    }
    if (input.shape.size() < 2 || grad.shape.size() != input.shape.size()) {
        return train_error("repeat_kv backward expects matching ranks >= 2");
    }
    const auto inner = static_cast<std::size_t>(
        std::accumulate(input.shape.begin() + 2, input.shape.end(), std::int64_t{1}, std::multiplies<>())
    );
    const auto outer = static_cast<std::size_t>(input.shape[0]);
    const auto heads = static_cast<std::size_t>(input.shape[1]);
    SimpleTensor result = zeros_like(input);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t out_base = (outer_index * heads + head) * inner;
            for (std::size_t rep = 0; rep < static_cast<std::size_t>(repeats); ++rep) {
                const std::size_t grad_head = head * static_cast<std::size_t>(repeats) + rep;
                const std::size_t grad_base = (outer_index * static_cast<std::size_t>(grad.shape[1]) + grad_head) * inner;
                for (std::size_t index = 0; index < inner; ++index) {
                    result.data[out_base + index] += grad.data[grad_base + index];
                }
            }
        }
    }
    return result;
}

std::variant<SimpleTensor, Diagnostic> backward_causal_mask_input(const SimpleTensor& grad) {
    if (grad.shape.size() < 2) {
        return train_error("causal_mask backward expects rank >= 2");
    }
    SimpleTensor result = grad;
    const auto q = static_cast<std::size_t>(grad.shape[grad.shape.size() - 2]);
    const auto k = static_cast<std::size_t>(grad.shape.back());
    const std::size_t inner_stride = q * k;
    const std::size_t outer = grad.data.size() / std::max<std::size_t>(inner_stride, 1);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
        const std::size_t base = outer_index * inner_stride;
        for (std::size_t row = 0; row < q; ++row) {
            for (std::size_t col = row + 1; col < k; ++col) {
                result.data[base + row * k + col] = 0.0F;
            }
        }
    }
    return result;
}

std::variant<SimpleTensor, Diagnostic> backward_rope_input(const SimpleTensor& grad, std::int64_t head_dim, double theta) {
    if (grad.shape.empty() || grad.shape.back() != head_dim) {
        return train_error("rope backward head_dim mismatch");
    }
    if (head_dim % 2 != 0) {
        return train_error("rope backward requires even head_dim");
    }
    const auto half = static_cast<std::size_t>(head_dim / 2);
    const auto seq_len = static_cast<std::size_t>(grad.shape.size() >= 2 ? grad.shape[grad.shape.size() - 2] : grad.shape[0]);
    const std::size_t outer = grad.data.size() / (seq_len * static_cast<std::size_t>(head_dim));
    SimpleTensor result = zeros_like(grad);
    std::vector<float> inv_freq(half, 0.0F);
    for (std::size_t index = 0; index < half; ++index) {
        inv_freq[index] = std::pow(static_cast<float>(theta), -static_cast<float>(index) / static_cast<float>(half));
    }
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
        const std::size_t outer_base = outer_index * seq_len * static_cast<std::size_t>(head_dim);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            const std::size_t pos_base = outer_base + pos * static_cast<std::size_t>(head_dim);
            for (std::size_t index = 0; index < half; ++index) {
                const float angle = static_cast<float>(pos) * inv_freq[index];
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float gy1 = grad.data[pos_base + index];
                const float gy2 = grad.data[pos_base + half + index];
                result.data[pos_base + index] = gy1 * c + gy2 * s;
                result.data[pos_base + half + index] = -gy1 * s + gy2 * c;
            }
        }
    }
    return result;
}

std::variant<SimpleTensor, Diagnostic> backward_embedding_weight(
    const SimpleTensor& grad,
    const SimpleTensor& indices,
    std::int64_t num_embeddings,
    std::int64_t embedding_dim
) {
    SimpleTensor weight_grad{{num_embeddings, embedding_dim}, std::vector<float>(static_cast<std::size_t>(num_embeddings * embedding_dim), 0.0F), grad.dtype};
    for (std::size_t index = 0; index < indices.data.size(); ++index) {
        const auto embedding_index = static_cast<std::int64_t>(indices.data[index]);
        if (embedding_index < 0 || embedding_index >= num_embeddings) {
            continue;
        }
        for (std::size_t dim = 0; dim < static_cast<std::size_t>(embedding_dim); ++dim) {
            weight_grad.data[static_cast<std::size_t>(embedding_index) * static_cast<std::size_t>(embedding_dim) + dim] +=
                grad.data[index * static_cast<std::size_t>(embedding_dim) + dim];
        }
    }
    return weight_grad;
}

std::variant<LinearSpec, Diagnostic> build_linear_spec(
    const PlanOp& op,
    const ExecutionPlan& plan,
    const PlanValue& output,
    const TrainExecutionResult& execution
) {
    LinearSpec spec;
    if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
        spec.closure.dtype = *output.type.callable_return->tensor_dtype;
    }
    if (op.inputs.size() == 1) {
        auto out_features = require_int(execution, op.inputs[0], "linear out_features");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) return *diagnostic;
        spec.closure.out_features = std::get<std::int64_t>(out_features);
    } else if (op.inputs.size() == 2) {
        auto second = execution.values.find(op.inputs[1]);
        if (second != execution.values.end() && std::holds_alternative<bool>(second->second)) {
            auto out_features = require_int(execution, op.inputs[0], "linear out_features");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) return *diagnostic;
            spec.closure.out_features = std::get<std::int64_t>(out_features);
            spec.closure.with_bias = std::get<bool>(second->second);
        } else {
            auto in_features = require_int(execution, op.inputs[0], "linear in_features");
            auto out_features = require_int(execution, op.inputs[1], "linear out_features");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&in_features)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) return *diagnostic;
            spec.closure.in_features = std::get<std::int64_t>(in_features);
            spec.closure.out_features = std::get<std::int64_t>(out_features);
        }
    } else {
        auto in_features = require_int(execution, op.inputs[0], "linear in_features");
        auto out_features = require_int(execution, op.inputs[1], "linear out_features");
        auto with_bias = execution.values.find(op.inputs[2]);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&in_features)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) return *diagnostic;
        if (with_bias == execution.values.end() || !std::holds_alternative<bool>(with_bias->second)) {
            return train_error("Missing linear with_bias");
        }
        spec.closure.in_features = std::get<std::int64_t>(in_features);
        spec.closure.out_features = std::get<std::int64_t>(out_features);
        spec.closure.with_bias = std::get<bool>(with_bias->second);
    }
    spec.weight_name = linear_weight_name(op.output);
    if (const PlanParameter* weight = find_plan_parameter(plan, op.output, "weight")) {
        spec.weight_name = weight->name;
        spec.weight_type = weight->tensor_type;
        if (auto in_features = known_graph_dim(weight->tensor_type, 0)) {
            spec.closure.in_features = *in_features;
        }
        if (auto out_features = known_graph_dim(weight->tensor_type, 1)) {
            spec.closure.out_features = *out_features;
        }
        if (weight->tensor_type.dtype) {
            spec.closure.dtype = *weight->tensor_type.dtype;
        }
    }
    if (spec.closure.with_bias) {
        spec.bias_name = linear_bias_name(op.output);
        if (const PlanParameter* bias = find_plan_parameter(plan, op.output, "bias")) {
            spec.bias_name = bias->name;
            spec.bias_type = bias->tensor_type;
            if (auto out_features = known_graph_dim(bias->tensor_type, 0)) {
                spec.closure.out_features = *out_features;
            }
            if (bias->tensor_type.dtype) {
                spec.closure.dtype = *bias->tensor_type.dtype;
            }
        }
    }
    return spec;
}

std::variant<EmbeddingSpec, Diagnostic> build_embedding_spec(
    const PlanOp& op,
    const ExecutionPlan& plan,
    const PlanValue& output,
    const TrainExecutionResult& execution
) {
    auto num_embeddings = require_int(execution, op.inputs[0], "num_embeddings");
    auto embedding_dim = require_int(execution, op.inputs[1], "embedding_dim");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&num_embeddings)) return *diagnostic;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&embedding_dim)) return *diagnostic;
    EmbeddingSpec spec;
    spec.closure.num_embeddings = std::get<std::int64_t>(num_embeddings);
    spec.closure.embedding_dim = std::get<std::int64_t>(embedding_dim);
    if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
        spec.closure.dtype = *output.type.callable_return->tensor_dtype;
    }
    spec.weight_name = embedding_weight_name(op.output);
    if (const PlanParameter* weight = find_plan_parameter(plan, op.output, "weight")) {
        spec.weight_name = weight->name;
        spec.weight_type = weight->tensor_type;
        if (auto num_embeddings = known_graph_dim(weight->tensor_type, 0)) {
            spec.closure.num_embeddings = *num_embeddings;
        }
        if (auto embedding_dim = known_graph_dim(weight->tensor_type, 1)) {
            spec.closure.embedding_dim = *embedding_dim;
        }
        if (weight->tensor_type.dtype) {
            spec.closure.dtype = *weight->tensor_type.dtype;
        }
    }
    return spec;
}

void ensure_linear_parameters(
    const LinearSpec& spec,
    const SimpleTensor& input,
    std::map<std::string, SimpleTensor>& parameters
) {
    std::int64_t in_features = spec.closure.in_features.value_or(input.shape[1]);
    if (spec.weight_type) {
        in_features = known_graph_dim(*spec.weight_type, 0).value_or(in_features);
    }
    parameters.emplace(spec.weight_name, make_linear_weight(in_features, spec.closure.out_features, spec.closure.dtype));
    if (spec.bias_name) {
        parameters.emplace(*spec.bias_name, make_linear_bias(spec.closure.out_features, spec.closure.dtype));
    }
}

void ensure_embedding_parameters(const EmbeddingSpec& spec, std::map<std::string, SimpleTensor>& parameters) {
    parameters.emplace(
        spec.weight_name,
        make_embedding_weight(spec.closure.num_embeddings, spec.closure.embedding_dim, spec.closure.dtype)
    );
}

std::variant<TrainValue, Diagnostic> execute_binary(const PlanOp& op, const TrainExecutionResult& execution) {
    const TrainValue& lhs = execution.values.at(op.inputs[0]);
    const TrainValue& rhs = execution.values.at(op.inputs[1]);
    if (const auto* lhs_tensor = std::get_if<SimpleTensor>(&lhs)) {
        if (const auto* rhs_tensor = std::get_if<SimpleTensor>(&rhs)) {
            auto result = elementwise_binary(op.binary_op, *lhs_tensor, *rhs_tensor);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
            return TrainValue{std::get<SimpleTensor>(std::move(result))};
        }
        auto scalar = require_number(rhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
        auto result = tensor_scalar_binary(op.binary_op, *lhs_tensor, std::get<double>(scalar));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (const auto* rhs_tensor = std::get_if<SimpleTensor>(&rhs)) {
        auto scalar = require_number(lhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
        auto result = scalar_tensor_binary(op.binary_op, std::get<double>(scalar), *rhs_tensor);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    auto left = require_number(lhs);
    auto right = require_number(rhs);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) return *diagnostic;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) return *diagnostic;
    switch (op.binary_op) {
        case FeBinaryOp::Add:
            return TrainValue{std::get<double>(left) + std::get<double>(right)};
        case FeBinaryOp::Sub:
            return TrainValue{std::get<double>(left) - std::get<double>(right)};
        case FeBinaryOp::Mul:
            return TrainValue{std::get<double>(left) * std::get<double>(right)};
        case FeBinaryOp::Div:
            return TrainValue{std::get<double>(left) / std::get<double>(right)};
        case FeBinaryOp::FloorDiv:
            return TrainValue{std::floor(std::get<double>(left) / std::get<double>(right))};
        default:
            return train_error("Unsupported scalar train binary op");
    }
}

std::variant<TrainValue, Diagnostic> execute_op(
    const PlanOp& op,
    const ExecutionPlan& plan,
    TrainExecutionResult& execution,
    std::map<std::string, SimpleTensor>& parameters
) {
    if (op.kind == PlanOpKind::Constant) {
        return constant_to_train_value(op.constant);
    }
    if (op.kind == PlanOpKind::Binary) {
        return execute_binary(op, execution);
    }
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "matmul") {
        auto lhs = require_tensor(execution, op.inputs[0], "matmul lhs");
        auto rhs = require_tensor(execution, op.inputs[1], "matmul rhs");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) return *diagnostic;
        auto result = matmul(**std::get_if<const SimpleTensor*>(&lhs), **std::get_if<const SimpleTensor*>(&rhs));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "relu") {
        auto input = require_tensor(execution, op.inputs[0], "relu input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        return TrainValue{apply_relu(**std::get_if<const SimpleTensor*>(&input))};
    }
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "scale") {
        auto input = require_tensor(execution, op.inputs[0], "scale input");
        auto scalar = require_number(execution, op.inputs[1], "scale factor");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
        auto result = tensor_scalar_binary(FeBinaryOp::Mul, **std::get_if<const SimpleTensor*>(&input), std::get<double>(scalar));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCtor && op.op == "linear") {
        auto spec = build_linear_spec(op, plan, plan.values[op.output], execution);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&spec)) return *diagnostic;
        return TrainValue{std::get<LinearSpec>(std::move(spec))};
    }
    if (op.kind == PlanOpKind::LibraryCtor && op.op == "Embedding") {
        auto spec = build_embedding_spec(op, plan, plan.values[op.output], execution);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&spec)) return *diagnostic;
        return TrainValue{std::get<EmbeddingSpec>(std::move(spec))};
    }
    if (op.kind == PlanOpKind::LibraryCtor &&
        (op.op == "SiLU" || op.op == "GELU" || op.op == "Tanh" || op.op == "Sigmoid" || op.op == "Softmax")) {
        return TrainValue{ActivationSpec{op.op, 0.0}};
    }
    if (op.kind == PlanOpKind::LibraryCtor && op.op == "Dropout") {
        auto probability = require_number(execution, op.inputs[0], "dropout probability");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&probability)) return *diagnostic;
        return TrainValue{ActivationSpec{"Dropout", std::get<double>(probability)}};
    }
    if (op.kind == PlanOpKind::Apply) {
        const TrainValue& callee = execution.values.at(op.inputs[0]);
        auto input = require_tensor(execution, op.inputs[1], "apply input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        const SimpleTensor& tensor = **std::get_if<const SimpleTensor*>(&input);
        if (const auto* spec = std::get_if<LinearSpec>(&callee)) {
            ensure_linear_parameters(*spec, tensor, parameters);
            const SimpleTensor& weight = parameters.at(spec->weight_name);
            const SimpleTensor* bias = spec->bias_name ? &parameters.at(*spec->bias_name) : nullptr;
            auto result = apply_linear_with_parameters(tensor, weight, bias);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
            return TrainValue{std::get<SimpleTensor>(std::move(result))};
        }
        if (const auto* spec = std::get_if<EmbeddingSpec>(&callee)) {
            ensure_embedding_parameters(*spec, parameters);
            auto result = apply_embedding_with_parameters(tensor, parameters.at(spec->weight_name), spec->closure.num_embeddings, spec->closure.embedding_dim);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
            return TrainValue{std::get<SimpleTensor>(std::move(result))};
        }
        if (const auto* activation = std::get_if<ActivationSpec>(&callee)) {
            if (activation->op == "SiLU") return TrainValue{apply_silu(tensor)};
            if (activation->op == "GELU") return TrainValue{apply_gelu(tensor)};
            if (activation->op == "Tanh") return TrainValue{apply_tanh(tensor)};
            if (activation->op == "Sigmoid") return TrainValue{apply_sigmoid(tensor)};
            if (activation->op == "Softmax") {
                auto result = apply_softmax(tensor);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
                return TrainValue{std::get<SimpleTensor>(std::move(result))};
            }
            if (activation->op == "Dropout") {
                auto result = apply_dropout(tensor, activation->probability);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
                return TrainValue{std::get<SimpleTensor>(std::move(result))};
            }
        }
        return train_error("Unsupported apply in train graph");
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "cross_entropy") {
        auto logits = require_tensor(execution, op.inputs[0], "logits");
        auto target = require_tensor(execution, op.inputs[1], "target");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&logits)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&target)) return *diagnostic;
        auto result = apply_cross_entropy(**std::get_if<const SimpleTensor*>(&logits), **std::get_if<const SimpleTensor*>(&target));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rms_norm") {
        auto input = require_tensor(execution, op.inputs[0], "rms_norm input");
        auto hidden = require_int(execution, op.inputs[1], "hidden size");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&hidden)) return *diagnostic;
        auto result = apply_rms_norm(**std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(hidden));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "reshape") {
        auto input = require_tensor(execution, op.inputs[0], "reshape input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        std::vector<std::int64_t> shape;
        for (std::size_t index = 1; index < op.inputs.size(); ++index) {
            auto dim = require_int(execution, op.inputs[index], "reshape dim");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&dim)) return *diagnostic;
            shape.push_back(std::get<std::int64_t>(dim));
        }
        auto result = apply_reshape(**std::get_if<const SimpleTensor*>(&input), shape);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "sum") {
        auto input = require_tensor(execution, op.inputs[0], "sum input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        if (op.inputs.size() == 1) return TrainValue{apply_sum(**std::get_if<const SimpleTensor*>(&input))};
        auto axis = require_int(execution, op.inputs[1], "sum axis");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) return *diagnostic;
        auto result = apply_sum_axis(**std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(axis));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "mean") {
        auto input = require_tensor(execution, op.inputs[0], "mean input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        if (op.inputs.size() == 1) return TrainValue{apply_mean(**std::get_if<const SimpleTensor*>(&input))};
        auto axis = require_int(execution, op.inputs[1], "mean axis");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) return *diagnostic;
        auto result = apply_mean_axis(**std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(axis));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "sqrt") {
        auto input = require_tensor(execution, op.inputs[0], "sqrt input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        return TrainValue{apply_sqrt(**std::get_if<const SimpleTensor*>(&input))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rsqrt") {
        auto input = require_tensor(execution, op.inputs[0], "rsqrt input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        return TrainValue{apply_rsqrt(**std::get_if<const SimpleTensor*>(&input))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "transpose") {
        auto input = require_tensor(execution, op.inputs[0], "transpose input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        auto result = apply_transpose(**std::get_if<const SimpleTensor*>(&input));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "repeat_kv") {
        auto input = require_tensor(execution, op.inputs[0], "repeat_kv input");
        auto repeats = require_int(execution, op.inputs[1], "repeats");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&repeats)) return *diagnostic;
        auto result = apply_repeat_kv(**std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(repeats));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "flatten_heads") {
        auto input = require_tensor(execution, op.inputs[0], "flatten_heads input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        auto result = apply_flatten_heads(**std::get_if<const SimpleTensor*>(&input));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "causal_mask") {
        auto input = require_tensor(execution, op.inputs[0], "causal_mask input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        auto result = apply_causal_mask(**std::get_if<const SimpleTensor*>(&input));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rope") {
        auto input = require_tensor(execution, op.inputs[0], "rope input");
        auto head_dim = require_int(execution, op.inputs[1], "rope head_dim");
        auto theta = require_number(execution, op.inputs[2], "rope theta");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&head_dim)) return *diagnostic;
        if (const auto* diagnostic = std::get_if<Diagnostic>(&theta)) return *diagnostic;
        auto result = apply_rope(**std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(head_dim), std::get<double>(theta));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) return *diagnostic;
        return TrainValue{std::get<SimpleTensor>(std::move(result))};
    }
    return train_error("Unsupported op '" + op.op + "' in train graph");
}

std::variant<TrainExecutionResult, Diagnostic> execute_train_plan(
    const ExecutionPlan& plan,
    const GraphExecutorOptions& options,
    std::map<std::string, SimpleTensor>& parameters
) {
    TrainExecutionResult execution;
    std::vector<std::size_t> use_counts(plan.values.size(), 0);
    for (const auto& op : plan.ops) {
        for (const auto input_id : op.inputs) {
            if (input_id < use_counts.size()) {
                ++use_counts[input_id];
            }
        }
    }
    for (const auto& step : plan.steps) {
        if (step.kind == PlanStepKind::AllocateHostValue) {
            const PlanValue& value = plan.values[step.value_id];
            if (value.is_parameter) {
                if (value.type.kind != FeTypeKind::Tensor) {
                    return train_error("Train executor currently supports tensor parameters only");
                }
                auto tensor = options.tensor_shapes.find(value.name);
                if (tensor == options.tensor_shapes.end()) {
                    return train_error("Missing --shape for tensor parameter '" + value.name + "'");
                }
                execution.values[value.id] = make_synthetic_tensor(tensor->second, value.type.tensor_dtype.value_or("float32"));
            }
            if (value.is_model_parameter) {
                if (value.id >= use_counts.size() || use_counts[value.id] == 0) {
                    continue;
                }
                const PlanParameter* parameter = find_plan_parameter_by_value(plan, value.id);
                if (parameter == nullptr) {
                    return train_error("Model parameter value '" + value.name + "' is missing plan parameter metadata");
                }
                auto tensor = materialize_model_parameter(value, *parameter, parameters);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&tensor)) {
                    return *diagnostic;
                }
                execution.values[value.id] = std::get<SimpleTensor>(std::move(tensor));
            }
            continue;
        }
        if (step.kind == PlanStepKind::ExecuteOp) {
            if (!step.op_index) {
                return train_error("ExecuteOp step is missing an op index");
            }
            const PlanOp& op = plan.ops[*step.op_index];
            auto value = execute_op(op, plan, execution, parameters);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
                return *diagnostic;
            }
            execution.values[op.output] = std::get<TrainValue>(std::move(value));
            continue;
        }
        if (step.kind == PlanStepKind::MaterializeOutput) {
            continue;
        }
        return train_error("Train executor currently supports local host execution only");
    }
    return execution;
}

std::variant<std::size_t, Diagnostic> resolve_objective_id(
    const FeExecutionRun& run,
    const ExecutionPlan& plan
) {
    if (run.objective_symbol) {
        auto found = plan.named_values.find(*run.objective_symbol);
        if (found == plan.named_values.end()) {
            return train_error("Could not resolve objective '" + *run.objective_symbol + "' in execution plan");
        }
        return found->second;
    }
    if (plan.outputs.empty()) {
        return train_error("Graph does not have a return value");
    }
    return plan.outputs.front();
}

std::variant<std::map<std::string, SimpleTensor>, Diagnostic> compute_parameter_gradients(
    const ExecutionPlan& plan,
    const TrainExecutionResult& execution,
    std::size_t objective_id,
    const std::map<std::string, SimpleTensor>& parameters
) {
    auto objective = require_tensor(execution, objective_id, "objective runtime value");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&objective)) return *diagnostic;
    std::map<std::size_t, SimpleTensor> value_grads;
    std::map<std::string, SimpleTensor> param_grads;
    value_grads[objective_id] = ones_like(**std::get_if<const SimpleTensor*>(&objective));

    for (auto it = plan.ops.rbegin(); it != plan.ops.rend(); ++it) {
        const PlanOp& op = *it;
        auto grad_found = value_grads.find(op.output);
        if (grad_found == value_grads.end()) {
            continue;
        }
        const SimpleTensor grad = grad_found->second;
        if (op.kind == PlanOpKind::Constant || op.kind == PlanOpKind::LibraryCtor) {
            continue;
        }
        if (op.kind == PlanOpKind::Apply) {
            const TrainValue& callee = execution.values.at(op.inputs[0]);
            auto input = require_tensor(execution, op.inputs[1], "apply input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            const SimpleTensor& tensor = **std::get_if<const SimpleTensor*>(&input);
            if (const auto* spec = std::get_if<LinearSpec>(&callee)) {
                auto weight = parameters.find(spec->weight_name);
                if (weight == parameters.end()) return train_error("Missing linear weight parameter '" + spec->weight_name + "'");
                auto input_grad = backward_linear_input(grad, weight->second);
                auto weight_grad = backward_linear_weight(grad, tensor);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
                if (const auto* diagnostic = std::get_if<Diagnostic>(&weight_grad)) return *diagnostic;
                if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], std::get<SimpleTensor>(input_grad))) return *diagnostic;
                if (auto diagnostic = accumulate_parameter(param_grads, spec->weight_name, std::get<SimpleTensor>(weight_grad))) return *diagnostic;
                if (spec->bias_name) {
                    auto bias_grad = backward_linear_bias(grad);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&bias_grad)) return *diagnostic;
                    if (auto diagnostic = accumulate_parameter(param_grads, *spec->bias_name, std::get<SimpleTensor>(bias_grad))) return *diagnostic;
                }
            } else if (const auto* spec = std::get_if<EmbeddingSpec>(&callee)) {
                auto weight_grad = backward_embedding_weight(grad, tensor, spec->closure.num_embeddings, spec->closure.embedding_dim);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&weight_grad)) return *diagnostic;
                if (auto diagnostic = accumulate_parameter(param_grads, spec->weight_name, std::get<SimpleTensor>(weight_grad))) return *diagnostic;
            } else if (const auto* activation = std::get_if<ActivationSpec>(&callee)) {
                std::variant<SimpleTensor, Diagnostic> input_grad;
                if (activation->op == "SiLU") input_grad = backward_silu(tensor, grad);
                else if (activation->op == "GELU") input_grad = backward_gelu(tensor, grad);
                else if (activation->op == "Tanh") input_grad = backward_tanh(tensor, grad);
                else if (activation->op == "Sigmoid") input_grad = backward_sigmoid(tensor, grad);
                else if (activation->op == "Softmax") {
                    auto output = require_tensor(execution, op.output, "softmax output");
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) return *diagnostic;
                    input_grad = backward_softmax(**std::get_if<const SimpleTensor*>(&output), grad);
                } else if (activation->op == "Dropout") {
                    input_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad, 1.0 - activation->probability);
                } else {
                    return train_error("Unsupported activation in train backward");
                }
                if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
                if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            }
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "cross_entropy") {
            auto logits = require_tensor(execution, op.inputs[0], "logits");
            auto target = require_tensor(execution, op.inputs[1], "target");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&logits)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&target)) return *diagnostic;
            auto logits_grad = backward_cross_entropy_logits(**std::get_if<const SimpleTensor*>(&logits), **std::get_if<const SimpleTensor*>(&target));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&logits_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(logits_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "rms_norm") {
            auto input = require_tensor(execution, op.inputs[0], "rms_norm input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            auto input_grad = backward_rms_norm_input(**std::get_if<const SimpleTensor*>(&input), grad);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "reshape") {
            auto input = require_tensor(execution, op.inputs[0], "reshape input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            auto input_grad = backward_reshape_input(grad, **std::get_if<const SimpleTensor*>(&input));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "sum") {
            auto input = require_tensor(execution, op.inputs[0], "sum input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            std::variant<SimpleTensor, Diagnostic> input_grad;
            if (op.inputs.size() > 1) {
                auto axis = require_int(execution, op.inputs[1], "sum axis");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) return *diagnostic;
                input_grad = backward_sum_axis_input(grad, **std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(axis));
            } else {
                input_grad = backward_sum_input(grad, **std::get_if<const SimpleTensor*>(&input));
            }
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "mean") {
            auto input = require_tensor(execution, op.inputs[0], "mean input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            std::variant<SimpleTensor, Diagnostic> input_grad;
            if (op.inputs.size() > 1) {
                auto axis = require_int(execution, op.inputs[1], "mean axis");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) return *diagnostic;
                input_grad = backward_mean_axis_input(grad, **std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(axis));
            } else {
                input_grad = backward_mean_input(grad, **std::get_if<const SimpleTensor*>(&input));
            }
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "transpose") {
            auto input_grad = apply_transpose(grad);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "flatten_heads") {
            auto input = require_tensor(execution, op.inputs[0], "flatten_heads input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            auto input_grad = backward_reshape_input(grad, **std::get_if<const SimpleTensor*>(&input));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "repeat_kv") {
            auto input = require_tensor(execution, op.inputs[0], "repeat_kv input");
            auto repeats = require_int(execution, op.inputs[1], "repeats");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&repeats)) return *diagnostic;
            auto input_grad = backward_repeat_kv_input(grad, **std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(repeats));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "causal_mask") {
            auto input_grad = backward_causal_mask_input(grad);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "rope") {
            auto head_dim = require_int(execution, op.inputs[1], "rope head_dim");
            auto theta = require_number(execution, op.inputs[2], "rope theta");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&head_dim)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&theta)) return *diagnostic;
            auto input_grad = backward_rope_input(grad, std::get<std::int64_t>(head_dim), std::get<double>(theta));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::PrimitiveCall && op.op == "matmul") {
            auto lhs = require_tensor(execution, op.inputs[0], "matmul lhs");
            auto rhs = require_tensor(execution, op.inputs[1], "matmul rhs");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) return *diagnostic;
            auto lhs_grad = backward_linear_input(grad, **std::get_if<const SimpleTensor*>(&rhs));
            auto rhs_grad = backward_linear_weight(grad, **std::get_if<const SimpleTensor*>(&lhs));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], std::get<SimpleTensor>(rhs_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::PrimitiveCall && op.op == "relu") {
            auto output = require_tensor(execution, op.output, "relu output");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) return *diagnostic;
            SimpleTensor mask = zeros_like(**std::get_if<const SimpleTensor*>(&output));
            for (std::size_t index = 0; index < mask.data.size(); ++index) {
                mask.data[index] = (**std::get_if<const SimpleTensor*>(&output)).data[index] > 0.0F ? 1.0F : 0.0F;
            }
            auto input_grad = elementwise_binary(FeBinaryOp::Mul, grad, mask);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::PrimitiveCall && op.op == "scale") {
            auto scale = require_number(execution, op.inputs[1], "scale");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&scale)) return *diagnostic;
            auto input_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad, std::get<double>(scale));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::Binary) {
            const TrainValue& lhs = execution.values.at(op.inputs[0]);
            const TrainValue& rhs = execution.values.at(op.inputs[1]);
            const auto* lhs_tensor = std::get_if<SimpleTensor>(&lhs);
            const auto* rhs_tensor = std::get_if<SimpleTensor>(&rhs);
            if (lhs_tensor != nullptr && rhs_tensor != nullptr) {
                const bool rhs_broadcast = is_trailing_vector_broadcast(*lhs_tensor, *rhs_tensor);
                if (op.binary_op == FeBinaryOp::Add) {
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], grad)) return *diagnostic;
                    SimpleTensor rhs_grad = grad;
                    if (rhs_broadcast) {
                        auto reduced = backward_linear_bias(grad);
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&reduced)) return *diagnostic;
                        rhs_grad = std::get<SimpleTensor>(std::move(reduced));
                    }
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], rhs_grad)) return *diagnostic;
                } else if (op.binary_op == FeBinaryOp::Sub) {
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], grad)) return *diagnostic;
                    SimpleTensor rhs_grad = negate(grad);
                    if (rhs_broadcast) {
                        auto reduced = backward_linear_bias(rhs_grad);
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&reduced)) return *diagnostic;
                        rhs_grad = std::get<SimpleTensor>(std::move(reduced));
                    }
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], rhs_grad)) return *diagnostic;
                } else if (op.binary_op == FeBinaryOp::Mul) {
                    auto lhs_grad = elementwise_binary(FeBinaryOp::Mul, grad, *rhs_tensor);
                    auto rhs_grad = elementwise_binary(FeBinaryOp::Mul, grad, *lhs_tensor);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_grad)) return *diagnostic;
                    SimpleTensor rhs_value_grad = std::get<SimpleTensor>(std::move(rhs_grad));
                    if (rhs_broadcast) {
                        auto reduced = backward_linear_bias(rhs_value_grad);
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&reduced)) return *diagnostic;
                        rhs_value_grad = std::get<SimpleTensor>(std::move(reduced));
                    }
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], rhs_value_grad)) return *diagnostic;
                } else if (op.binary_op == FeBinaryOp::Div) {
                    auto lhs_grad = elementwise_binary(FeBinaryOp::Div, grad, *rhs_tensor);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
                }
            } else if (lhs_tensor != nullptr) {
                if (op.binary_op == FeBinaryOp::Add || op.binary_op == FeBinaryOp::Sub) {
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], grad)) return *diagnostic;
                } else if (op.binary_op == FeBinaryOp::Mul || op.binary_op == FeBinaryOp::Div) {
                    auto scalar = require_number(rhs);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
                    auto input_grad = tensor_scalar_binary(op.binary_op, grad, std::get<double>(scalar));
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
                }
            } else if (rhs_tensor != nullptr) {
                if (op.binary_op == FeBinaryOp::Add) {
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], grad)) return *diagnostic;
                } else if (op.binary_op == FeBinaryOp::Sub) {
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], negate(grad))) return *diagnostic;
                } else if (op.binary_op == FeBinaryOp::Mul) {
                    auto scalar = require_number(lhs);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
                    auto input_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad, std::get<double>(scalar));
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
                    if (auto diagnostic = accumulate_value(value_grads, op.inputs[1], std::get<SimpleTensor>(input_grad))) return *diagnostic;
                }
            }
            continue;
        }
    }
    for (const auto& parameter : plan.parameters) {
        if (!parameter.trainable) {
            continue;
        }
        auto found = value_grads.find(parameter.value_id);
        if (found == value_grads.end()) {
            continue;
        }
        if (auto diagnostic = accumulate_parameter(param_grads, parameter.name, found->second)) {
            return *diagnostic;
        }
    }
    return param_grads;
}

float tensor_sum(const SimpleTensor& tensor) {
    return std::accumulate(tensor.data.begin(), tensor.data.end(), 0.0F);
}

std::optional<Diagnostic> apply_sgd(
    std::map<std::string, SimpleTensor>& parameters,
    const std::map<std::string, SimpleTensor>& gradients,
    double learning_rate
) {
    for (const auto& [name, grad] : gradients) {
        auto parameter = parameters.find(name);
        if (parameter == parameters.end()) {
            continue;
        }
        if (parameter->second.shape != grad.shape || parameter->second.data.size() != grad.data.size()) {
            return train_error("Parameter gradient shape mismatch for '" + name + "'");
        }
        for (std::size_t index = 0; index < parameter->second.data.size(); ++index) {
            parameter->second.data[index] -= static_cast<float>(learning_rate) * grad.data[index];
        }
    }
    return std::nullopt;
}

std::variant<std::string, Diagnostic> train_optimizer(const FeExecutionRun& run) {
    if (!run.optimizer) {
        return std::string("sgd");
    }
    if (const auto* value = std::get_if<std::string>(&run.optimizer->value)) {
        return *value;
    }
    return train_error("Train optimizer must be a string");
}

double train_learning_rate(const FeExecutionRun& run) {
    if (!run.learning_rate) {
        return 0.01;
    }
    if (const auto* value = std::get_if<double>(&run.learning_rate->value)) {
        return *value;
    }
    if (const auto* value = std::get_if<std::int64_t>(&run.learning_rate->value)) {
        return static_cast<double>(*value);
    }
    return 0.01;
}

std::int64_t train_iterations(const FeExecutionRun& run) {
    if (!run.iteration) {
        return 1;
    }
    if (const auto* value = std::get_if<std::int64_t>(&run.iteration->value)) {
        return *value;
    }
    return 1;
}

} // namespace

std::optional<Diagnostic> run_train_plan_module(
    const LoweredModule& lowered,
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
) {
    if (module.backend != BackendKind::Local) {
        return train_error("C++ train currently supports the local backend only");
    }
    const ExecutionPlan* plan = find_plan(module, entry);
    if (plan == nullptr) {
        if (auto skipped_reason = find_skipped_plan_reason(module, entry)) {
            return train_error("Entry function '" + entry + "' was skipped during graph planning: " + *skipped_reason);
        }
        return train_error("Entry function '" + entry + "' not found in execution plan module");
    }
    const FeExecutionRun* run = find_train_run(lowered, entry);
    if (run == nullptr) {
        return train_error("Train executor requires a resolved 'config model' execution run");
    }
    auto optimizer = train_optimizer(*run);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&optimizer)) {
        return *diagnostic;
    }
    if (std::get<std::string>(optimizer) != "sgd") {
        return train_error("Train executor currently supports only optimizer='sgd'");
    }
    auto objective_id = resolve_objective_id(*run, *plan);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&objective_id)) {
        return *diagnostic;
    }

    std::map<std::string, SimpleTensor> parameters;
    const double learning_rate = train_learning_rate(*run);
    const std::int64_t iterations = train_iterations(*run);

    std::cout << "\n--- Training Output ---\n";
    std::cout << "backend=local\n";
    for (std::int64_t step = 0; step < iterations; ++step) {
        auto execution = execute_train_plan(*plan, options, parameters);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
            return *diagnostic;
        }
        const TrainExecutionResult& result = std::get<TrainExecutionResult>(execution);
        auto objective = require_tensor(result, std::get<std::size_t>(objective_id), "objective runtime value");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&objective)) {
            return *diagnostic;
        }
        auto gradients = compute_parameter_gradients(*plan, result, std::get<std::size_t>(objective_id), parameters);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&gradients)) {
            return *diagnostic;
        }
        const float loss = tensor_sum(**std::get_if<const SimpleTensor*>(&objective));
        if (auto diagnostic = apply_sgd(parameters, std::get<std::map<std::string, SimpleTensor>>(gradients), learning_rate)) {
            return *diagnostic;
        }
        std::cout << "step=" << (step + 1)
                  << " loss=" << std::fixed << std::setprecision(6) << loss
                  << " params=" << parameters.size() << '\n';
    }
    std::cout << "-----------------------\n";
    return std::nullopt;
}
