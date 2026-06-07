#include "backward_executor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <variant>

namespace {

Diagnostic backward_error(std::string message) {
    return Diagnostic::error("runtime", "R0003", std::move(message));
}

const ExecutionPlan* find_plan(const PlanModule& module, const std::string& entry) {
    auto found = std::find_if(module.plans.begin(), module.plans.end(), [&](const ExecutionPlan& plan) {
        return plan.name == entry;
    });
    return found == module.plans.end() ? nullptr : &*found;
}

std::variant<const SimpleTensor*, Diagnostic> require_tensor(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return backward_error("Missing " + label);
    }
    const auto* tensor = std::get_if<SimpleTensor>(&found->second);
    if (tensor == nullptr) {
        return backward_error("Backward expected tensor value for " + label);
    }
    return tensor;
}

std::variant<double, Diagnostic> require_scalar(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return backward_error("Missing " + label);
    }
    if (const auto* value = std::get_if<std::int64_t>(&found->second)) {
        return static_cast<double>(*value);
    }
    if (const auto* value = std::get_if<double>(&found->second)) {
        return *value;
    }
    if (const auto* value = std::get_if<bool>(&found->second)) {
        return *value ? 1.0 : 0.0;
    }
    return backward_error("Backward expected scalar value for " + label);
}

std::variant<std::int64_t, Diagnostic> require_int(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return backward_error("Missing " + label);
    }
    if (const auto* value = std::get_if<std::int64_t>(&found->second)) {
        return *value;
    }
    return backward_error("Backward expected integer value for " + label);
}

std::variant<bool, Diagnostic> require_bool(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return backward_error("Missing " + label);
    }
    if (const auto* value = std::get_if<bool>(&found->second)) {
        return *value;
    }
    return backward_error("Backward expected bool value for " + label);
}

SimpleTensor zeros_like(const SimpleTensor& tensor) {
    return SimpleTensor{tensor.shape, std::vector<float>(tensor.data.size(), 0.0F), tensor.dtype};
}

SimpleTensor ones_like(const SimpleTensor& tensor) {
    return SimpleTensor{tensor.shape, std::vector<float>(tensor.data.size(), 1.0F), tensor.dtype};
}

SimpleTensor make_backward_linear_weight(std::int64_t in_features, std::int64_t out_features, const std::string& dtype) {
    const std::size_t element_count = static_cast<std::size_t>(in_features * out_features);
    std::vector<float> data;
    data.reserve(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        data.push_back((static_cast<float>(index % 11) - 5.0F) / 16.0F);
    }
    return SimpleTensor{{in_features, out_features}, std::move(data), dtype};
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
        return backward_error("gradient tensor shape mismatch");
    }
    for (std::size_t index = 0; index < dst.data.size(); ++index) {
        dst.data[index] += src.data[index];
    }
    return std::nullopt;
}

std::optional<Diagnostic> accumulate(
    std::map<std::size_t, SimpleTensor>& gradients,
    std::size_t value_id,
    const SimpleTensor& gradient
) {
    auto found = gradients.find(value_id);
    if (found == gradients.end()) {
        gradients.emplace(value_id, gradient);
        return std::nullopt;
    }
    return add_in_place(found->second, gradient);
}

std::variant<SimpleTensor, Diagnostic> multiply(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    return elementwise_binary(FeBinaryOp::Mul, lhs, rhs);
}

std::variant<SimpleTensor, Diagnostic> divide(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    return elementwise_binary(FeBinaryOp::Div, lhs, rhs);
}

std::variant<SimpleTensor, Diagnostic> matmul_checked(const SimpleTensor& lhs, const SimpleTensor& rhs) {
    auto result = matmul(lhs, rhs);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        return *diagnostic;
    }
    return std::get<SimpleTensor>(std::move(result));
}

std::variant<SimpleTensor, Diagnostic> transpose_checked(const SimpleTensor& tensor) {
    auto result = transpose_2d(tensor);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        return *diagnostic;
    }
    return std::get<SimpleTensor>(std::move(result));
}

std::variant<LinearClosure, Diagnostic> build_linear_closure(
    const PlanOp& op,
    const PlanValue& output,
    const GraphExecutionResult& execution
) {
    LinearClosure closure;
    if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
        closure.dtype = *output.type.callable_return->tensor_dtype;
    }
    if (op.inputs.size() == 1) {
        auto out_features = require_int(execution, op.inputs[0], "linear out_features");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
            return *diagnostic;
        }
        closure.out_features = std::get<std::int64_t>(out_features);
        return closure;
    }
    if (op.inputs.size() == 2) {
        auto maybe_bool = require_bool(execution, op.inputs[1], "linear with_bias");
        if (std::holds_alternative<bool>(maybe_bool)) {
            auto out_features = require_int(execution, op.inputs[0], "linear out_features");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
                return *diagnostic;
            }
            closure.out_features = std::get<std::int64_t>(out_features);
            closure.with_bias = std::get<bool>(maybe_bool);
            return closure;
        }
    }
    auto in_features = require_int(execution, op.inputs[0], "linear in_features");
    auto out_features = require_int(execution, op.inputs[1], "linear out_features");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&in_features)) {
        return *diagnostic;
    }
    if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
        return *diagnostic;
    }
    closure.in_features = std::get<std::int64_t>(in_features);
    closure.out_features = std::get<std::int64_t>(out_features);
    if (op.inputs.size() == 3) {
        auto with_bias = require_bool(execution, op.inputs[2], "linear with_bias");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&with_bias)) {
            return *diagnostic;
        }
        closure.with_bias = std::get<bool>(with_bias);
    }
    return closure;
}

const PlanOp* find_producer(const ExecutionPlan& plan, std::size_t output_id) {
    auto found = std::find_if(plan.ops.begin(), plan.ops.end(), [&](const PlanOp& op) {
        return op.output == output_id;
    });
    return found == plan.ops.end() ? nullptr : &*found;
}

std::variant<std::size_t, Diagnostic> resolve_objective_id(
    const LoweredModule& lowered,
    const ExecutionPlan& plan,
    const std::string& entry
) {
    if (lowered.execution_plan) {
        auto run = std::find_if(lowered.execution_plan->runs.begin(), lowered.execution_plan->runs.end(), [&](const FeExecutionRun& item) {
            return item.model_name == entry;
        });
        if (run != lowered.execution_plan->runs.end() && run->objective_symbol) {
            auto named = plan.named_values.find(*run->objective_symbol);
            if (named == plan.named_values.end()) {
                return backward_error("Could not resolve objective '" + *run->objective_symbol + "' in execution plan");
            }
            return named->second;
        }
    }
    if (plan.outputs.empty()) {
        return backward_error("Entry function did not return a value");
    }
    return plan.outputs.front();
}

std::variant<SimpleTensor, Diagnostic> backward_silu(const SimpleTensor& input, const SimpleTensor& grad) {
    if (input.shape != grad.shape) {
        return backward_error("backward_silu shape mismatch");
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
        return backward_error("backward_gelu shape mismatch");
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
        return backward_error("backward_tanh shape mismatch");
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
        return backward_error("backward_sigmoid shape mismatch");
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
        return backward_error("backward_softmax shape mismatch");
    }
    SimpleTensor result = output;
    for (std::size_t index = 0; index < result.data.size(); ++index) {
        result.data[index] = output.data[index] * (1.0F - output.data[index]) * grad.data[index];
    }
    return result;
}

std::variant<SimpleTensor, Diagnostic> backward_sum_input(const SimpleTensor& grad, const SimpleTensor& input) {
    if (grad.data.size() != 1) {
        return backward_error("backward_sum_input expects a single gradient value");
    }
    return SimpleTensor{input.shape, std::vector<float>(input.data.size(), grad.data[0]), input.dtype};
}

std::variant<SimpleTensor, Diagnostic> backward_sum_axis_input(const SimpleTensor& grad, const SimpleTensor& input, std::int64_t axis) {
    if (input.shape.size() != 2) {
        return backward_error("backward_sum_axis_input expects rank-2 input");
    }
    const auto rows = static_cast<std::size_t>(input.shape[0]);
    const auto cols = static_cast<std::size_t>(input.shape[1]);
    SimpleTensor output = zeros_like(input);
    if (axis == 0) {
        if (grad.data.size() != cols) {
            return backward_error("backward_sum_axis_input axis 0 gradient shape mismatch");
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                output.data[row * cols + col] = grad.data[col];
            }
        }
        return output;
    }
    if (axis == 1) {
        if (grad.data.size() != rows) {
            return backward_error("backward_sum_axis_input axis 1 gradient shape mismatch");
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                output.data[row * cols + col] = grad.data[row];
            }
        }
        return output;
    }
    return backward_error("backward_sum_axis_input supports axis 0 or 1");
}

std::variant<SimpleTensor, Diagnostic> backward_mean_input(const SimpleTensor& grad, const SimpleTensor& input) {
    if (grad.data.size() != 1) {
        return backward_error("backward_mean_input expects a single gradient value");
    }
    const float value = grad.data[0] / static_cast<float>(input.data.size());
    return SimpleTensor{input.shape, std::vector<float>(input.data.size(), value), input.dtype};
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

std::variant<SimpleTensor, Diagnostic> backward_cross_entropy_logits(const SimpleTensor& logits, const SimpleTensor& target) {
    if (logits.shape != target.shape) {
        return backward_error("backward_cross_entropy_logits shape mismatch");
    }
    auto softmax = apply_softmax(logits);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&softmax)) {
        return *diagnostic;
    }
    SimpleTensor grad = std::get<SimpleTensor>(std::move(softmax));
    for (std::size_t index = 0; index < grad.data.size(); ++index) {
        grad.data[index] -= target.data[index];
    }
    return grad;
}

std::variant<SimpleTensor, Diagnostic> backward_cross_entropy_target(const SimpleTensor& target) {
    return zeros_like(target);
}

std::variant<SimpleTensor, Diagnostic> backward_binary(
    const PlanOp& op,
    const GraphExecutionResult& execution,
    const SimpleTensor& grad,
    std::map<std::size_t, SimpleTensor>& gradients
) {
    auto lhs_found = execution.values.find(op.inputs[0]);
    auto rhs_found = execution.values.find(op.inputs[1]);
    if (lhs_found == execution.values.end() || rhs_found == execution.values.end()) {
        return backward_error("Missing binary runtime value");
    }
    const auto* lhs_tensor = std::get_if<SimpleTensor>(&lhs_found->second);
    const auto* rhs_tensor = std::get_if<SimpleTensor>(&rhs_found->second);
    if (lhs_tensor != nullptr && rhs_tensor != nullptr) {
        if (op.binary_op == FeBinaryOp::Add) {
            if (auto diagnostic = accumulate(gradients, op.inputs[0], grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], grad)) return *diagnostic;
            return grad;
        }
        if (op.binary_op == FeBinaryOp::Sub) {
            if (auto diagnostic = accumulate(gradients, op.inputs[0], grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], negate(grad))) return *diagnostic;
            return grad;
        }
        if (op.binary_op == FeBinaryOp::Mul) {
            auto lhs_grad = multiply(grad, *rhs_tensor);
            auto rhs_grad = multiply(grad, *lhs_tensor);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], std::get<SimpleTensor>(rhs_grad))) return *diagnostic;
            return grad;
        }
        if (op.binary_op == FeBinaryOp::Div) {
            auto lhs_grad = divide(grad, *rhs_tensor);
            auto rhs_sq = multiply(*rhs_tensor, *rhs_tensor);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_sq)) return *diagnostic;
            auto numerator = multiply(grad, *lhs_tensor);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&numerator)) return *diagnostic;
            auto rhs_grad = divide(std::get<SimpleTensor>(numerator), std::get<SimpleTensor>(rhs_sq));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], negate(std::get<SimpleTensor>(rhs_grad)))) return *diagnostic;
            return grad;
        }
        return backward_error("Unsupported tensor binary op in backward");
    }
    if (lhs_tensor != nullptr) {
        auto scalar = require_scalar(execution, op.inputs[1], "binary rhs");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
        if (op.binary_op == FeBinaryOp::Add || op.binary_op == FeBinaryOp::Sub) {
            if (auto diagnostic = accumulate(gradients, op.inputs[0], grad)) return *diagnostic;
            return grad;
        }
        if (op.binary_op == FeBinaryOp::Mul || op.binary_op == FeBinaryOp::Div) {
            auto lhs_grad = tensor_scalar_binary(op.binary_op, grad, std::get<double>(scalar));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
            return grad;
        }
        return backward_error("Unsupported tensor-scalar op in backward");
    }
    if (rhs_tensor != nullptr) {
        auto scalar = require_scalar(execution, op.inputs[0], "binary lhs");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
        if (op.binary_op == FeBinaryOp::Add) {
            if (auto diagnostic = accumulate(gradients, op.inputs[1], grad)) return *diagnostic;
            return grad;
        }
        if (op.binary_op == FeBinaryOp::Sub) {
            if (auto diagnostic = accumulate(gradients, op.inputs[1], negate(grad))) return *diagnostic;
            return grad;
        }
        if (op.binary_op == FeBinaryOp::Mul) {
            auto rhs_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad, std::get<double>(scalar));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], std::get<SimpleTensor>(rhs_grad))) return *diagnostic;
            return grad;
        }
        return backward_error("Backward for scalar/tensor division is not implemented yet");
    }
    return grad;
}

std::optional<Diagnostic> compute_gradients(
    const LoweredModule& lowered,
    const ExecutionPlan& plan,
    const GraphExecutionResult& execution,
    std::map<std::size_t, SimpleTensor>& gradients
) {
    auto objective_id = resolve_objective_id(lowered, plan, plan.name);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&objective_id)) {
        return *diagnostic;
    }
    auto objective = require_tensor(execution, std::get<std::size_t>(objective_id), "objective runtime value");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&objective)) {
        return *diagnostic;
    }
    gradients[std::get<std::size_t>(objective_id)] = ones_like(**std::get_if<const SimpleTensor*>(&objective));

    for (auto op_iter = plan.ops.rbegin(); op_iter != plan.ops.rend(); ++op_iter) {
        const PlanOp& op = *op_iter;
        auto gradient = gradients.find(op.output);
        if (gradient == gradients.end()) {
            continue;
        }
        const SimpleTensor grad = gradient->second;

        if (op.kind == PlanOpKind::Constant || op.kind == PlanOpKind::LibraryCtor) {
            continue;
        }
        if (op.kind == PlanOpKind::Binary) {
            auto result = backward_binary(op, execution, grad, gradients);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                return *diagnostic;
            }
            continue;
        }
        if (op.kind == PlanOpKind::PrimitiveCall && op.op == "matmul") {
            auto lhs = require_tensor(execution, op.inputs[0], "matmul lhs");
            auto rhs = require_tensor(execution, op.inputs[1], "matmul rhs");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) return *diagnostic;
            auto rhs_t = transpose_checked(**std::get_if<const SimpleTensor*>(&rhs));
            auto lhs_t = transpose_checked(**std::get_if<const SimpleTensor*>(&lhs));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_t)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_t)) return *diagnostic;
            auto lhs_grad = matmul_checked(grad, std::get<SimpleTensor>(rhs_t));
            auto rhs_grad = matmul_checked(std::get<SimpleTensor>(lhs_t), grad);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_grad)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(lhs_grad))) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], std::get<SimpleTensor>(rhs_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::PrimitiveCall && op.op == "relu") {
            auto output = require_tensor(execution, op.output, "relu output");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) return *diagnostic;
            SimpleTensor mask = zeros_like(**std::get_if<const SimpleTensor*>(&output));
            for (std::size_t index = 0; index < mask.data.size(); ++index) {
                mask.data[index] = (**std::get_if<const SimpleTensor*>(&output)).data[index] > 0.0F ? 1.0F : 0.0F;
            }
            auto input_grad = multiply(grad, mask);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::PrimitiveCall && op.op == "scale") {
            auto scalar = require_scalar(execution, op.inputs[1], "scale");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) return *diagnostic;
            auto input_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad, std::get<double>(scalar));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::Apply) {
            const PlanOp* callee = find_producer(plan, op.inputs[0]);
            if (callee == nullptr) {
                return backward_error("Backward could not resolve apply callee producer");
            }
            auto input = require_tensor(execution, op.inputs[1], "apply input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            std::variant<SimpleTensor, Diagnostic> input_grad;
            if (callee->op == "linear") {
                auto closure = build_linear_closure(*callee, plan.values[callee->output], execution);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&closure)) return *diagnostic;
                const LinearClosure linear = std::get<LinearClosure>(closure);
                const std::int64_t in_features = linear.in_features.value_or((**std::get_if<const SimpleTensor*>(&input)).shape[1]);
                const SimpleTensor weight = make_backward_linear_weight(in_features, linear.out_features, linear.dtype);
                auto weight_t = transpose_checked(weight);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&weight_t)) return *diagnostic;
                input_grad = matmul_checked(grad, std::get<SimpleTensor>(weight_t));
            } else if (callee->op == "SiLU") {
                input_grad = backward_silu(**std::get_if<const SimpleTensor*>(&input), grad);
            } else if (callee->op == "GELU") {
                input_grad = backward_gelu(**std::get_if<const SimpleTensor*>(&input), grad);
            } else if (callee->op == "Tanh") {
                input_grad = backward_tanh(**std::get_if<const SimpleTensor*>(&input), grad);
            } else if (callee->op == "Sigmoid") {
                input_grad = backward_sigmoid(**std::get_if<const SimpleTensor*>(&input), grad);
            } else if (callee->op == "Softmax") {
                auto output = require_tensor(execution, op.output, "softmax output");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&output)) return *diagnostic;
                input_grad = backward_softmax(**std::get_if<const SimpleTensor*>(&output), grad);
            } else if (callee->op == "Dropout") {
                auto probability = require_scalar(execution, callee->inputs[0], "dropout probability");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&probability)) return *diagnostic;
                input_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad, 1.0 - std::get<double>(probability));
            } else {
                return backward_error("Backward does not support apply constructor '" + callee->op + "' yet");
            }
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "rms_norm") {
            if (auto diagnostic = accumulate(gradients, op.inputs[0], grad)) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "cross_entropy") {
            auto logits = require_tensor(execution, op.inputs[0], "cross_entropy logits");
            auto target = require_tensor(execution, op.inputs[1], "cross_entropy target");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&logits)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&target)) return *diagnostic;
            auto logits_grad = backward_cross_entropy_logits(**std::get_if<const SimpleTensor*>(&logits), **std::get_if<const SimpleTensor*>(&target));
            auto target_grad = backward_cross_entropy_target(**std::get_if<const SimpleTensor*>(&target));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&logits_grad)) return *diagnostic;
            if (const auto* diagnostic = std::get_if<Diagnostic>(&target_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(logits_grad))) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[1], std::get<SimpleTensor>(target_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && (op.op == "sum" || op.op == "mean")) {
            auto input = require_tensor(execution, op.inputs[0], op.op + " input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
            std::variant<SimpleTensor, Diagnostic> input_grad;
            if (op.inputs.size() > 1) {
                auto axis = require_int(execution, op.inputs[1], op.op + " axis");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) return *diagnostic;
                input_grad = op.op == "sum"
                    ? backward_sum_axis_input(grad, **std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(axis))
                    : backward_mean_axis_input(grad, **std::get_if<const SimpleTensor*>(&input), std::get<std::int64_t>(axis));
            } else {
                input_grad = op.op == "sum"
                    ? backward_sum_input(grad, **std::get_if<const SimpleTensor*>(&input))
                    : backward_mean_input(grad, **std::get_if<const SimpleTensor*>(&input));
            }
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        if (op.kind == PlanOpKind::LibraryCall && op.op == "transpose") {
            auto input_grad = transpose_checked(grad);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_grad)) return *diagnostic;
            if (auto diagnostic = accumulate(gradients, op.inputs[0], std::get<SimpleTensor>(input_grad))) return *diagnostic;
            continue;
        }
        return backward_error("Unsupported op '" + op.op + "' in C++ backward executor");
    }
    return std::nullopt;
}

void print_named_tensor(const std::string& name, const SimpleTensor& tensor) {
    std::cout << name << ":\n" << format_tensor(tensor) << '\n';
}

} // namespace

std::optional<Diagnostic> run_backward_plan_module(
    const LoweredModule& lowered,
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
) {
    if (module.backend != BackendKind::Local) {
        return backward_error("C++ backward currently supports the local backend only");
    }
    const ExecutionPlan* plan = find_plan(module, entry);
    if (plan == nullptr) {
        return backward_error("Entry function '" + entry + "' not found in execution plan module");
    }
    auto execution = execute_plan_module(module, entry, options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        return *diagnostic;
    }
    const GraphExecutionResult& result = std::get<GraphExecutionResult>(execution);
    std::map<std::size_t, SimpleTensor> gradients;
    if (auto diagnostic = compute_gradients(lowered, *plan, result, gradients)) {
        return *diagnostic;
    }

    std::cout << "\n--- Gradient Output ---\n";
    for (const auto& value : plan->values) {
        if (!value.is_parameter || value.type.kind != FeTypeKind::Tensor) {
            continue;
        }
        auto gradient = gradients.find(value.id);
        if (gradient != gradients.end()) {
            print_named_tensor(value.name + "_grad", gradient->second);
            continue;
        }
        auto tensor = require_tensor(result, value.id, value.name);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&tensor)) {
            return *diagnostic;
        }
        print_named_tensor(value.name + "_grad", zeros_like(**std::get_if<const SimpleTensor*>(&tensor)));
    }
    std::cout << "-----------------------\n";
    return std::nullopt;
}
