#include "cpu_executor.h"
#include "cpu_kernels.h"

#include "runtime_utils.h"

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
  return Diagnostic::error(DiagnosticCode::RuntimeBackwardError,
                           std::move(message));
}

std::variant<SimpleTensor, Diagnostic>
backward_reshape_input(const SimpleTensor &grad, const SimpleTensor &input) {
  if (grad.data.size() != input.data.size()) {
    return backward_error("reshape backward requires matching element counts");
  }
  return SimpleTensor{input.shape, grad.data, grad.dtype};
}

std::variant<SimpleTensor, Diagnostic>
backward_repeat_kv_input(const SimpleTensor &grad, const SimpleTensor &input,
                         std::int64_t repeats) {
  if (repeats <= 0) {
    return backward_error("repeat_kv backward requires repeats > 0");
  }
  if (input.shape.size() < 2 || grad.shape.size() != input.shape.size()) {
    return backward_error("repeat_kv backward expects matching ranks >= 2");
  }
  const auto inner = static_cast<std::size_t>(
      std::accumulate(input.shape.begin() + 2, input.shape.end(),
                      std::int64_t{1}, std::multiplies<>()));
  const auto outer = static_cast<std::size_t>(input.shape[0]);
  const auto heads = static_cast<std::size_t>(input.shape[1]);
  SimpleTensor result = zeros_like(input);
  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t out_base = (outer_index * heads + head) * inner;
      for (std::size_t rep = 0; rep < static_cast<std::size_t>(repeats);
           ++rep) {
        const std::size_t grad_head =
            head * static_cast<std::size_t>(repeats) + rep;
        const std::size_t grad_base =
            (outer_index * static_cast<std::size_t>(grad.shape[1]) +
             grad_head) *
            inner;
        for (std::size_t index = 0; index < inner; ++index) {
          result.data[out_base + index] += grad.data[grad_base + index];
        }
      }
    }
  }
  return result;
}

std::variant<SimpleTensor, Diagnostic>
backward_causal_mask_input(const SimpleTensor &grad) {
  if (grad.shape.size() < 2) {
    return backward_error("causal_mask backward expects rank >= 2");
  }
  SimpleTensor result = grad;
  const auto q = static_cast<std::size_t>(grad.shape[grad.shape.size() - 2]);
  const auto k = static_cast<std::size_t>(grad.shape.back());
  const std::size_t inner_stride = q * k;
  const std::size_t outer =
      grad.data.size() / std::max<std::size_t>(inner_stride, 1);
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

std::variant<SimpleTensor, Diagnostic>
backward_rope_input(const SimpleTensor &grad, std::int64_t head_dim,
                    double theta) {
  if (grad.shape.empty() || grad.shape.back() != head_dim) {
    return backward_error("rope backward head_dim mismatch");
  }
  if (head_dim % 2 != 0) {
    return backward_error("rope backward requires even head_dim");
  }
  const auto half = static_cast<std::size_t>(head_dim / 2);
  const auto seq_len = static_cast<std::size_t>(
      grad.shape.size() >= 2 ? grad.shape[grad.shape.size() - 2]
                             : grad.shape[0]);
  const std::size_t outer =
      grad.data.size() / (seq_len * static_cast<std::size_t>(head_dim));
  SimpleTensor result = zeros_like(grad);
  std::vector<float> inv_freq(half, 0.0F);
  for (std::size_t index = 0; index < half; ++index) {
    inv_freq[index] =
        std::pow(static_cast<float>(theta),
                 -static_cast<float>(index) / static_cast<float>(half));
  }
  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    const std::size_t outer_base =
        outer_index * seq_len * static_cast<std::size_t>(head_dim);
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
      const std::size_t pos_base =
          outer_base + pos * static_cast<std::size_t>(head_dim);
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

std::variant<SimpleTensor, Diagnostic>
backward_embedding_weight(const SimpleTensor &grad, const SimpleTensor &indices,
                          std::int64_t num_embeddings,
                          std::int64_t embedding_dim) {
  TensorData weight_grad_data;
  weight_grad_data.assign(
      static_cast<std::size_t>(num_embeddings * embedding_dim), 0.0F);
  SimpleTensor weight_grad{
      {num_embeddings, embedding_dim},
      std::move(weight_grad_data),
      grad.dtype};
  for (std::size_t index = 0; index < indices.data.size(); ++index) {
    const auto embedding_index = static_cast<std::int64_t>(indices.data[index]);
    if (embedding_index < 0 || embedding_index >= num_embeddings) {
      continue;
    }
    for (std::size_t dim = 0; dim < static_cast<std::size_t>(embedding_dim);
         ++dim) {
      weight_grad.data[static_cast<std::size_t>(embedding_index) *
                           static_cast<std::size_t>(embedding_dim) +
                       dim] +=
          grad.data[index * static_cast<std::size_t>(embedding_dim) + dim];
    }
  }
  return weight_grad;
}

std::optional<Diagnostic>
accumulate_parameter(std::map<std::string, SimpleTensor> &param_grads,
                     const std::string &name, const SimpleTensor &grad) {
  auto existing = param_grads.find(name);
  if (existing == param_grads.end()) {
    param_grads.emplace(name, grad);
    return std::nullopt;
  }
  if (existing->second.shape != grad.shape ||
      existing->second.data.size() != grad.data.size()) {
    return backward_error("Parameter gradient shape mismatch for '" + name +
                          "'");
  }
  for (std::size_t index = 0; index < grad.data.size(); ++index) {
    existing->second.data[index] += grad.data[index];
  }
  return std::nullopt;
}

bool is_trailing_vector_broadcast(const SimpleTensor &lhs,
                                  const SimpleTensor &rhs) {
  return lhs.shape.size() == 2 && rhs.shape.size() == 1 &&
         lhs.shape[1] == rhs.shape[0];
}

std::variant<SimpleTensor, Diagnostic>
reduce_trailing_vector_gradient(const SimpleTensor &grad) {
  if (grad.shape.size() != 2) {
    return backward_error(
        "trailing-vector broadcast backward requires rank-2 gradients");
  }
  const auto rows = static_cast<std::size_t>(grad.shape[0]);
  const auto cols = static_cast<std::size_t>(grad.shape[1]);
  SimpleTensor reduced{{static_cast<std::int64_t>(cols)},
                       std::vector<float>(cols, 0.0F),
                       grad.dtype};
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      reduced.data[col] += grad.data[row * cols + col];
    }
  }
  return reduced;
}

SimpleTensor make_backward_linear_weight(std::int64_t in_features,
                                         std::int64_t out_features,
                                         const std::string &dtype) {
  const std::size_t element_count =
      static_cast<std::size_t>(in_features * out_features);
  std::vector<float> data;
  data.reserve(element_count);
  for (std::size_t index = 0; index < element_count; ++index) {
    data.push_back((static_cast<float>(index % 11) - 5.0F) / 16.0F);
  }
  return SimpleTensor{{in_features, out_features}, std::move(data), dtype};
}

SimpleTensor negate(const SimpleTensor &tensor) {
  SimpleTensor output = tensor;
  for (auto &value : output.data) {
    value = -value;
  }
  return output;
}

std::optional<Diagnostic> add_in_place(SimpleTensor &dst,
                                       const SimpleTensor &src) {
  if (dst.shape != src.shape || dst.data.size() != src.data.size()) {
    return backward_error("gradient tensor shape mismatch");
  }
  for (std::size_t index = 0; index < dst.data.size(); ++index) {
    dst.data[index] += src.data[index];
  }
  return std::nullopt;
}

std::optional<Diagnostic>
accumulate(std::map<std::size_t, SimpleTensor> &gradients, std::size_t valueId,
           const SimpleTensor &gradient) {
  auto found = gradients.find(valueId);
  if (found == gradients.end()) {
    gradients.emplace(valueId, gradient);
    return std::nullopt;
  }
  return add_in_place(found->second, gradient);
}

std::variant<SimpleTensor, Diagnostic> multiply(const SimpleTensor &lhs,
                                                const SimpleTensor &rhs) {
  return elementwise_binary(FeBinaryOp::Mul, lhs, rhs);
}

std::variant<SimpleTensor, Diagnostic> divide(const SimpleTensor &lhs,
                                              const SimpleTensor &rhs) {
  return elementwise_binary(FeBinaryOp::Div, lhs, rhs);
}

std::variant<SimpleTensor, Diagnostic> matmul_checked(const SimpleTensor &lhs,
                                                      const SimpleTensor &rhs) {
  auto result = matmul(lhs, rhs);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&result)) {
    return *diagnostic;
  }
  return std::get<SimpleTensor>(std::move(result));
}

std::variant<SimpleTensor, Diagnostic>
transpose_checked(const SimpleTensor &tensor) {
  auto result = transpose_2d(tensor);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&result)) {
    return *diagnostic;
  }
  return std::get<SimpleTensor>(std::move(result));
}

std::variant<LinearClosure, Diagnostic>
build_linear_closure(const PlanOp &op, const PlanValue &output,
                     const GraphExecutionResult &execution) {
  LinearClosure closure;
  if (output.type.callableReturn && output.type.callableReturn->tensorDtype) {
    closure.dtype = *output.type.callableReturn->tensorDtype;
  }
  if (op.inputs.size() == 1) {
    auto out_features =
        require_int(execution, op.inputs[0], "linear out_features");
    if (const auto *diagnostic = std::get_if<Diagnostic>(&out_features)) {
      return *diagnostic;
    }
    closure.out_features = std::get<std::int64_t>(out_features);
    return closure;
  }
  if (op.inputs.size() == 2) {
    auto maybe_bool = require_bool(execution, op.inputs[1], "linear with_bias");
    if (std::holds_alternative<bool>(maybe_bool)) {
      auto out_features =
          require_int(execution, op.inputs[0], "linear out_features");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&out_features)) {
        return *diagnostic;
      }
      closure.out_features = std::get<std::int64_t>(out_features);
      closure.with_bias = std::get<bool>(maybe_bool);
      return closure;
    }
  }
  auto in_features = require_int(execution, op.inputs[0], "linear in_features");
  auto out_features =
      require_int(execution, op.inputs[1], "linear out_features");
  if (const auto *diagnostic = std::get_if<Diagnostic>(&in_features)) {
    return *diagnostic;
  }
  if (const auto *diagnostic = std::get_if<Diagnostic>(&out_features)) {
    return *diagnostic;
  }
  closure.in_features = std::get<std::int64_t>(in_features);
  closure.out_features = std::get<std::int64_t>(out_features);
  if (op.inputs.size() == 3) {
    auto with_bias = require_bool(execution, op.inputs[2], "linear with_bias");
    if (const auto *diagnostic = std::get_if<Diagnostic>(&with_bias)) {
      return *diagnostic;
    }
    closure.with_bias = std::get<bool>(with_bias);
  }
  return closure;
}

const PlanOp *find_producer(const ExecutionPlan &plan, std::size_t output_id) {
  auto found =
      std::find_if(plan.ops.begin(), plan.ops.end(),
                   [&](const PlanOp &op) { return op.output == output_id; });
  return found == plan.ops.end() ? nullptr : &*found;
}

std::variant<std::size_t, Diagnostic>
resolve_objective_id(const LoweredModule &lowered, const ExecutionPlan &plan,
                     const std::string &entry) {
  if (lowered.executionPlan) {
    auto run = std::find_if(
        lowered.executionPlan->runs.begin(), lowered.executionPlan->runs.end(),
        [&](const FeExecutionRun &item) { return item.modelName == entry; });
    if (run != lowered.executionPlan->runs.end() && run->objectiveSymbol) {
      auto named = plan.namedValues.find(*run->objectiveSymbol);
      if (named == plan.namedValues.end()) {
        return backward_error("Could not resolve objective '" +
                              *run->objectiveSymbol + "' in execution plan");
      }
      return named->second;
    }
  }
  if (plan.outputs.empty()) {
    return backward_error("Entry function did not return a value");
  }
  return plan.outputs.front();
}

std::variant<SimpleTensor, Diagnostic> backward_silu(const SimpleTensor &input,
                                                     const SimpleTensor &grad) {
  if (input.shape != grad.shape) {
    return backward_error("backward_silu shape mismatch");
  }
  SimpleTensor output = input;
  for (std::size_t index = 0; index < output.data.size(); ++index) {
    const float x = input.data[index];
    const float sigmoid = 1.0F / (1.0F + std::exp(-x));
    output.data[index] =
        grad.data[index] * (sigmoid + x * sigmoid * (1.0F - sigmoid));
  }
  return output;
}

std::variant<SimpleTensor, Diagnostic> backward_gelu(const SimpleTensor &input,
                                                     const SimpleTensor &grad) {
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
    output.data[index] = grad.data[index] * (0.5F * (1.0F + tanh_inner) +
                                             0.5F * x * sech_sq * inner_grad);
  }
  return output;
}

std::variant<SimpleTensor, Diagnostic> backward_tanh(const SimpleTensor &input,
                                                     const SimpleTensor &grad) {
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

std::variant<SimpleTensor, Diagnostic>
backward_sigmoid(const SimpleTensor &input, const SimpleTensor &grad) {
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

std::variant<SimpleTensor, Diagnostic>
backward_softmax(const SimpleTensor &output, const SimpleTensor &grad) {
  if (output.shape != grad.shape) {
    return backward_error("backward_softmax shape mismatch");
  }
  SimpleTensor result = output;
  for (std::size_t index = 0; index < result.data.size(); ++index) {
    result.data[index] =
        output.data[index] * (1.0F - output.data[index]) * grad.data[index];
  }
  return result;
}

std::variant<SimpleTensor, Diagnostic>
backward_sum_input(const SimpleTensor &grad, const SimpleTensor &input) {
  if (grad.data.size() != 1) {
    return backward_error("backward_sum_input expects a single gradient value");
  }
  return SimpleTensor{input.shape,
                      std::vector<float>(input.data.size(), grad.data[0]),
                      input.dtype};
}

std::variant<SimpleTensor, Diagnostic>
backward_sum_axis_input(const SimpleTensor &grad, const SimpleTensor &input,
                        std::int64_t axis) {
  if (input.shape.size() != 2) {
    return backward_error("backward_sum_axis_input expects rank-2 input");
  }
  const auto rows = static_cast<std::size_t>(input.shape[0]);
  const auto cols = static_cast<std::size_t>(input.shape[1]);
  SimpleTensor output = zeros_like(input);
  if (axis == 0) {
    if (grad.data.size() != cols) {
      return backward_error(
          "backward_sum_axis_input axis 0 gradient shape mismatch");
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
      return backward_error(
          "backward_sum_axis_input axis 1 gradient shape mismatch");
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

std::variant<SimpleTensor, Diagnostic>
backward_mean_input(const SimpleTensor &grad, const SimpleTensor &input) {
  if (grad.data.size() != 1) {
    return backward_error(
        "backward_mean_input expects a single gradient value");
  }
  const float value = grad.data[0] / static_cast<float>(input.data.size());
  return SimpleTensor{input.shape, std::vector<float>(input.data.size(), value),
                      input.dtype};
}

std::variant<SimpleTensor, Diagnostic>
backward_mean_axis_input(const SimpleTensor &grad, const SimpleTensor &input,
                         std::int64_t axis) {
  auto summed = backward_sum_axis_input(grad, input, axis);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&summed)) {
    return *diagnostic;
  }
  SimpleTensor output = std::get<SimpleTensor>(std::move(summed));
  const float divisor =
      static_cast<float>(axis == 0 ? input.shape[0] : input.shape[1]);
  for (auto &value : output.data) {
    value /= divisor;
  }
  return output;
}

std::variant<SimpleTensor, Diagnostic>
backward_cross_entropy_logits(const SimpleTensor &logits,
                              const SimpleTensor &target) {
  if (logits.shape != target.shape) {
    return backward_error("backward_cross_entropy_logits shape mismatch");
  }
  auto softmax = apply_softmax(logits);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&softmax)) {
    return *diagnostic;
  }
  SimpleTensor grad = std::get<SimpleTensor>(std::move(softmax));
  for (std::size_t index = 0; index < grad.data.size(); ++index) {
    grad.data[index] -= target.data[index];
  }
  return grad;
}

std::variant<SimpleTensor, Diagnostic>
backward_cross_entropy_target(const SimpleTensor &target) {
  return zeros_like(target);
}

std::variant<SimpleTensor, Diagnostic>
backward_binary(const PlanOp &op, const GraphExecutionResult &execution,
                const SimpleTensor &grad,
                std::map<std::size_t, SimpleTensor> &gradients) {
  // Computes and accumulates gradients for generic binary operations (e.g. Add,
  // Mul). Automatically handles trailing vector broadcast reduction if required
  // by the shapes.
  auto lhs_found = execution.values.find(op.inputs[0]);
  auto rhs_found = execution.values.find(op.inputs[1]);
  if (lhs_found == execution.values.end() ||
      rhs_found == execution.values.end()) {
    return backward_error("Missing binary runtime value");
  }
  const auto *lhs_tensor = std::get_if<SimpleTensor>(&lhs_found->second);
  const auto *rhs_tensor = std::get_if<SimpleTensor>(&rhs_found->second);
  if (lhs_tensor != nullptr && rhs_tensor != nullptr) {
    const bool rhs_broadcast =
        is_trailing_vector_broadcast(*lhs_tensor, *rhs_tensor);
    if (op.binaryOp == FeBinaryOp::Add) {
      if (auto diagnostic = accumulate(gradients, op.inputs[0], grad))
        return *diagnostic;
      SimpleTensor rhs_grad = grad;
      if (rhs_broadcast) {
        auto reduced = reduce_trailing_vector_gradient(grad);
        if (const auto *diagnostic = std::get_if<Diagnostic>(&reduced))
          return *diagnostic;
        rhs_grad = std::get<SimpleTensor>(std::move(reduced));
      }
      if (auto diagnostic = accumulate(gradients, op.inputs[1], rhs_grad))
        return *diagnostic;
      return grad;
    }
    if (op.binaryOp == FeBinaryOp::Sub) {
      if (auto diagnostic = accumulate(gradients, op.inputs[0], grad))
        return *diagnostic;
      SimpleTensor rhs_grad = negate(grad);
      if (rhs_broadcast) {
        auto reduced = reduce_trailing_vector_gradient(rhs_grad);
        if (const auto *diagnostic = std::get_if<Diagnostic>(&reduced))
          return *diagnostic;
        rhs_grad = std::get<SimpleTensor>(std::move(reduced));
      }
      if (auto diagnostic = accumulate(gradients, op.inputs[1], rhs_grad))
        return *diagnostic;
      return grad;
    }
    if (op.binaryOp == FeBinaryOp::Mul) {
      auto lhs_grad = multiply(grad, *rhs_tensor);
      auto rhs_grad = multiply(grad, *lhs_tensor);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&lhs_grad))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs_grad))
        return *diagnostic;
      SimpleTensor rhs_value_grad = std::get<SimpleTensor>(std::move(rhs_grad));
      if (rhs_broadcast) {
        auto reduced = reduce_trailing_vector_gradient(rhs_value_grad);
        if (const auto *diagnostic = std::get_if<Diagnostic>(&reduced))
          return *diagnostic;
        rhs_value_grad = std::get<SimpleTensor>(std::move(reduced));
      }
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(lhs_grad)))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[1], rhs_value_grad))
        return *diagnostic;
      return grad;
    }
    if (op.binaryOp == FeBinaryOp::Div) {
      auto lhs_grad = divide(grad, *rhs_tensor);
      auto rhs_sq = multiply(*rhs_tensor, *rhs_tensor);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&lhs_grad))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs_sq))
        return *diagnostic;
      auto numerator = multiply(grad, *lhs_tensor);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&numerator))
        return *diagnostic;
      auto rhs_grad = divide(std::get<SimpleTensor>(numerator),
                             std::get<SimpleTensor>(rhs_sq));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(lhs_grad)))
        return *diagnostic;
      if (auto diagnostic =
              accumulate(gradients, op.inputs[1],
                         negate(std::get<SimpleTensor>(rhs_grad))))
        return *diagnostic;
      return grad;
    }
    return backward_error("Unsupported tensor binary op in backward");
  }
  if (lhs_tensor != nullptr) {
    auto scalar = require_scalar(execution, op.inputs[1], "binary rhs");
    if (const auto *diagnostic = std::get_if<Diagnostic>(&scalar))
      return *diagnostic;
    if (op.binaryOp == FeBinaryOp::Add || op.binaryOp == FeBinaryOp::Sub) {
      if (auto diagnostic = accumulate(gradients, op.inputs[0], grad))
        return *diagnostic;
      return grad;
    }
    if (op.binaryOp == FeBinaryOp::Mul || op.binaryOp == FeBinaryOp::Div) {
      auto lhs_grad =
          tensor_scalar_binary(op.binaryOp, grad, std::get<double>(scalar));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&lhs_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(lhs_grad)))
        return *diagnostic;
      return grad;
    }
    return backward_error("Unsupported tensor-scalar op in backward");
  }
  if (rhs_tensor != nullptr) {
    auto scalar = require_scalar(execution, op.inputs[0], "binary lhs");
    if (const auto *diagnostic = std::get_if<Diagnostic>(&scalar))
      return *diagnostic;
    if (op.binaryOp == FeBinaryOp::Add) {
      if (auto diagnostic = accumulate(gradients, op.inputs[1], grad))
        return *diagnostic;
      return grad;
    }
    if (op.binaryOp == FeBinaryOp::Sub) {
      if (auto diagnostic = accumulate(gradients, op.inputs[1], negate(grad)))
        return *diagnostic;
      return grad;
    }
    if (op.binaryOp == FeBinaryOp::Mul) {
      auto rhs_grad =
          tensor_scalar_binary(FeBinaryOp::Mul, grad, std::get<double>(scalar));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[1],
                                       std::get<SimpleTensor>(rhs_grad)))
        return *diagnostic;
      return grad;
    }
    return backward_error(
        "Backward for scalar/tensor division is not implemented yet");
  }
  return grad;
}

std::variant<BackwardResult, Diagnostic>
compute_gradients_internal(
    const ExecutionPlan &plan, const GraphExecutionResult &execution,
    std::size_t objective_id,
    const std::map<std::string, SimpleTensor> *parameters) {
  BackwardResult result;
  std::map<std::size_t, SimpleTensor> &gradients = result.value_grads;
  auto objective =
      require_tensor(execution, objective_id, "objective runtime value");
  if (const auto *diagnostic = std::get_if<Diagnostic>(&objective)) {
    return *diagnostic;
  }
  gradients[objective_id] =
      ones_like(**std::get_if<const SimpleTensor *>(&objective));

  for (auto op_iter = plan.ops.rbegin(); op_iter != plan.ops.rend();
       ++op_iter) {
    const PlanOp &op = *op_iter;
    auto gradient = gradients.find(op.output);
    if (gradient == gradients.end()) {
      continue;
    }
    const SimpleTensor grad = gradient->second;

    if (op.kind == PlanOpKind::Constant || op.kind == PlanOpKind::LibraryCtor) {
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall &&
        (op.op == "tensor" || op.op == "zeros" || op.op == "ones" ||
         op.op == "randn")) {
      continue;
    }
    if (op.kind == PlanOpKind::Binary) {
      auto result = backward_binary(op, execution, grad, gradients);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&result)) {
        return *diagnostic;
      }
      continue;
    }
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "matmul") {
      auto lhs = require_tensor(execution, op.inputs[0], "matmul lhs");
      auto rhs = require_tensor(execution, op.inputs[1], "matmul rhs");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&lhs))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs))
        return *diagnostic;
      auto rhs_t = transpose_checked(**std::get_if<const SimpleTensor *>(&rhs));
      auto lhs_t = transpose_checked(**std::get_if<const SimpleTensor *>(&lhs));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs_t))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&lhs_t))
        return *diagnostic;
      auto lhs_grad = matmul_checked(grad, std::get<SimpleTensor>(rhs_t));
      auto rhs_grad = matmul_checked(std::get<SimpleTensor>(lhs_t), grad);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&lhs_grad))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&rhs_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(lhs_grad)))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[1],
                                       std::get<SimpleTensor>(rhs_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "relu") {
      auto output = require_tensor(execution, op.output, "relu output");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&output))
        return *diagnostic;
      SimpleTensor mask =
          zeros_like(**std::get_if<const SimpleTensor *>(&output));
      for (std::size_t index = 0; index < mask.data.size(); ++index) {
        mask.data[index] =
            (**std::get_if<const SimpleTensor *>(&output)).data[index] > 0.0F
                ? 1.0F
                : 0.0F;
      }
      auto input_grad = multiply(grad, mask);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::PrimitiveCall && op.op == "scale") {
      auto scalar = require_scalar(execution, op.inputs[1], "scale");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&scalar))
        return *diagnostic;
      auto input_grad =
          tensor_scalar_binary(FeBinaryOp::Mul, grad, std::get<double>(scalar));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::Apply) {
      const PlanOp *callee = find_producer(plan, op.inputs[0]);
      if (callee == nullptr) {
        return backward_error(
            "Backward could not resolve apply callee producer");
      }
      auto input = require_tensor(execution, op.inputs[1], "apply input");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input))
        return *diagnostic;
      std::variant<SimpleTensor, Diagnostic> input_grad;
      if (callee->op == "linear") {
        auto closure = build_linear_closure(
            *callee, plan.values[callee->output], execution);
        if (const auto *diagnostic = std::get_if<Diagnostic>(&closure))
          return *diagnostic;
        const LinearClosure linear = std::get<LinearClosure>(closure);
        const std::int64_t in_features = linear.in_features.value_or(
            (**std::get_if<const SimpleTensor *>(&input)).shape[1]);
        const PlanParameter *weight_param =
            find_plan_parameter(plan, op.inputs[0], "weight");
        const std::string weight_name =
            weight_param != nullptr
                ? weight_param->name
                : ("linear_" + std::to_string(op.inputs[0]) + "_weight");
        const SimpleTensor weight =
            (parameters != nullptr && parameters->count(weight_name) > 0)
                ? parameters->at(weight_name)
                : make_backward_linear_weight(in_features, linear.out_features,
                                              linear.dtype);
        auto weight_t = transpose_checked(weight);
        if (const auto *diagnostic = std::get_if<Diagnostic>(&weight_t))
          return *diagnostic;
        input_grad = matmul_checked(grad, std::get<SimpleTensor>(weight_t));
        if (parameters != nullptr) {
          auto input_t =
              transpose_checked(**std::get_if<const SimpleTensor *>(&input));
          if (const auto *diagnostic = std::get_if<Diagnostic>(&input_t))
            return *diagnostic;
          auto weight_grad =
              matmul_checked(std::get<SimpleTensor>(input_t), grad);
          if (const auto *diagnostic = std::get_if<Diagnostic>(&weight_grad))
            return *diagnostic;
          if (auto diagnostic = accumulate_parameter(
                  result.param_grads, weight_name,
                  std::get<SimpleTensor>(weight_grad)))
            return *diagnostic;
          if (linear.with_bias) {
            const PlanParameter *bias_param =
                find_plan_parameter(plan, op.inputs[0], "bias");
            const std::string bias_name =
                bias_param != nullptr
                    ? bias_param->name
                    : ("linear_" + std::to_string(op.inputs[0]) + "_bias");
            auto bias_grad = reduce_trailing_vector_gradient(grad);
            if (const auto *diagnostic = std::get_if<Diagnostic>(&bias_grad))
              return *diagnostic;
            if (auto diagnostic = accumulate_parameter(
                    result.param_grads, bias_name,
                    std::get<SimpleTensor>(bias_grad)))
              return *diagnostic;
          }
        }
      } else if (callee->op == "Embedding") {
        auto num_embeddings =
            require_int(execution, callee->inputs[0], "num_embeddings");
        auto embedding_dim =
            require_int(execution, callee->inputs[1], "embedding_dim");
        if (const auto *diagnostic = std::get_if<Diagnostic>(&num_embeddings))
          return *diagnostic;
        if (const auto *diagnostic = std::get_if<Diagnostic>(&embedding_dim))
          return *diagnostic;
        const PlanParameter *weight_param =
            find_plan_parameter(plan, op.inputs[0], "weight");
        const std::string weight_name =
            weight_param != nullptr
                ? weight_param->name
                : ("embedding_" + std::to_string(op.inputs[0]) + "_weight");
        auto weight_grad = backward_embedding_weight(
            grad, **std::get_if<const SimpleTensor *>(&input),
            std::get<std::int64_t>(num_embeddings),
            std::get<std::int64_t>(embedding_dim));
        if (const auto *diagnostic = std::get_if<Diagnostic>(&weight_grad))
          return *diagnostic;
        if (auto diagnostic = accumulate_parameter(
                result.param_grads, weight_name,
                std::get<SimpleTensor>(weight_grad)))
          return *diagnostic;
        continue;
      } else if (callee->op == "SiLU") {
        input_grad =
            backward_silu(**std::get_if<const SimpleTensor *>(&input), grad);
      } else if (callee->op == "GELU") {
        input_grad =
            backward_gelu(**std::get_if<const SimpleTensor *>(&input), grad);
      } else if (callee->op == "Tanh") {
        input_grad =
            backward_tanh(**std::get_if<const SimpleTensor *>(&input), grad);
      } else if (callee->op == "Sigmoid") {
        input_grad =
            backward_sigmoid(**std::get_if<const SimpleTensor *>(&input), grad);
      } else if (callee->op == "Softmax") {
        auto output = require_tensor(execution, op.output, "softmax output");
        if (const auto *diagnostic = std::get_if<Diagnostic>(&output))
          return *diagnostic;
        input_grad = backward_softmax(
            **std::get_if<const SimpleTensor *>(&output), grad);
      } else if (callee->op == "Dropout") {
        auto probability =
            require_scalar(execution, callee->inputs[0], "dropout probability");
        if (const auto *diagnostic = std::get_if<Diagnostic>(&probability))
          return *diagnostic;
        input_grad = tensor_scalar_binary(FeBinaryOp::Mul, grad,
                                          1.0 - std::get<double>(probability));
      } else {
        return backward_error("Backward does not support apply constructor '" +
                              callee->op + "' yet");
      }
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[1],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rms_norm") {
      if (auto diagnostic = accumulate(gradients, op.inputs[0], grad))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "cross_entropy") {
      auto logits =
          require_tensor(execution, op.inputs[0], "cross_entropy logits");
      auto target =
          require_tensor(execution, op.inputs[1], "cross_entropy target");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&logits))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&target))
        return *diagnostic;
      auto logits_grad = backward_cross_entropy_logits(
          **std::get_if<const SimpleTensor *>(&logits),
          **std::get_if<const SimpleTensor *>(&target));
      auto target_grad = backward_cross_entropy_target(
          **std::get_if<const SimpleTensor *>(&target));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&logits_grad))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&target_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(logits_grad)))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[1],
                                       std::get<SimpleTensor>(target_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall &&
        (op.op == "sum" || op.op == "mean")) {
      auto input = require_tensor(execution, op.inputs[0], op.op + " input");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input))
        return *diagnostic;
      std::variant<SimpleTensor, Diagnostic> input_grad;
      if (op.inputs.size() > 1) {
        auto axis = require_int(execution, op.inputs[1], op.op + " axis");
        if (const auto *diagnostic = std::get_if<Diagnostic>(&axis))
          return *diagnostic;
        input_grad =
            op.op == "sum"
                ? backward_sum_axis_input(
                      grad, **std::get_if<const SimpleTensor *>(&input),
                      std::get<std::int64_t>(axis))
                : backward_mean_axis_input(
                      grad, **std::get_if<const SimpleTensor *>(&input),
                      std::get<std::int64_t>(axis));
      } else {
        input_grad =
            op.op == "sum"
                ? backward_sum_input(
                      grad, **std::get_if<const SimpleTensor *>(&input))
                : backward_mean_input(
                      grad, **std::get_if<const SimpleTensor *>(&input));
      }
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "transpose") {
      auto input_grad = transpose_checked(grad);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall &&
        (op.op == "reshape" || op.op == "flatten_heads")) {
      auto input = require_tensor(execution, op.inputs[0], op.op + " input");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input))
        return *diagnostic;
      auto input_grad = backward_reshape_input(
          grad, **std::get_if<const SimpleTensor *>(&input));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "repeat_kv") {
      auto input = require_tensor(execution, op.inputs[0], "repeat_kv input");
      auto repeats = require_int(execution, op.inputs[1], "repeats");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&repeats))
        return *diagnostic;
      auto input_grad = backward_repeat_kv_input(
          grad, **std::get_if<const SimpleTensor *>(&input),
          std::get<std::int64_t>(repeats));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "causal_mask") {
      auto input_grad = backward_causal_mask_input(grad);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    if (op.kind == PlanOpKind::LibraryCall && op.op == "rope") {
      auto head_dim = require_int(execution, op.inputs[1], "rope head_dim");
      auto theta = require_scalar(execution, op.inputs[2], "rope theta");
      if (const auto *diagnostic = std::get_if<Diagnostic>(&head_dim))
        return *diagnostic;
      if (const auto *diagnostic = std::get_if<Diagnostic>(&theta))
        return *diagnostic;
      auto input_grad = backward_rope_input(
          grad, std::get<std::int64_t>(head_dim), std::get<double>(theta));
      if (const auto *diagnostic = std::get_if<Diagnostic>(&input_grad))
        return *diagnostic;
      if (auto diagnostic = accumulate(gradients, op.inputs[0],
                                       std::get<SimpleTensor>(input_grad)))
        return *diagnostic;
      continue;
    }
    return backward_error("Unsupported op '" + op.op +
                          "' in C++ backward executor");
  }
  for (const auto &parameter : plan.parameters) {
    if (!parameter.trainable) {
      continue;
    }
    auto found = gradients.find(parameter.valueId);
    if (found == gradients.end()) {
      continue;
    }
    if (auto diagnostic = accumulate_parameter(result.param_grads,
                                               parameter.name, found->second)) {
      return *diagnostic;
    }
  }
  return result;
}

void print_named_tensor(const std::string &name, const SimpleTensor &tensor) {
  std::cout << name << ":\n" << format_tensor(tensor) << '\n';
}

} // namespace

std::variant<BackwardResult, Diagnostic>
compute_parameter_gradients(const ExecutionPlan &plan,
                            const GraphExecutionResult &execution,
                            std::size_t objective_id,
                            const std::map<std::string, SimpleTensor> *parameters) {
  return compute_gradients_internal(plan, execution, objective_id, parameters);
}

std::optional<Diagnostic>
run_backward_plan_module(const LoweredModule &lowered, const PlanModule &module,
                         const std::string &entry,
                         const GraphExecutorOptions &options) {
  if (module.backend != BackendKind::Host) {
    return backward_error(
        "C++ backward currently supports the local backend only");
  }
  const ExecutionPlan *plan = find_plan(module, entry);
  if (plan == nullptr) {
    return backward_error("Entry function '" + entry +
                          "' not found in execution plan module");
  }
  auto objective_id = resolve_objective_id(lowered, *plan, plan->name);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&objective_id)) {
    return *diagnostic;
  }
  auto execution = execute_plan_module(module, entry, options);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&execution)) {
    return *diagnostic;
  }
  const GraphExecutionResult &result =
      std::get<GraphExecutionResult>(execution);
  auto backward_result = compute_parameter_gradients(
      *plan, result, std::get<std::size_t>(objective_id), nullptr);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&backward_result)) {
    return *diagnostic;
  }
  const auto &gradients = std::get<BackwardResult>(backward_result).value_grads;

  std::cout << "\n--- Gradient Output ---\n";
  for (const auto &value : plan->values) {
    if ((!value.isParameter && !value.isModelParameter) ||
        value.type.kind != FeTypeKind::Tensor) {
      continue;
    }
    auto gradient = gradients.find(value.id);
    if (gradient != gradients.end()) {
      print_named_tensor(value.name + "_grad", gradient->second);
      continue;
    }
    auto tensor = require_tensor(result, value.id, value.name);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&tensor)) {
      return *diagnostic;
    }
    print_named_tensor(
        value.name + "_grad",
        zeros_like(**std::get_if<const SimpleTensor *>(&tensor)));
  }
  std::cout << "-----------------------\n";
  return std::nullopt;
}
