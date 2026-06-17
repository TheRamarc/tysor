#include "graph_executor.h"

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace {

struct ActivationClosure {
    // Store activation constructors as OpId so apply dispatch stays enum-based.
    OpId op = OpId::Silu;
    double probability = 0.0;
};

// RuntimeValue includes callable closures because graph lowering represents
// layer constructors and later application as separate graph operations.
using RuntimeValue = std::variant<std::int64_t, double, bool, SimpleTensor, LinearClosure, EmbeddingClosure, ActivationClosure>;

struct RuntimeValueStore {
    explicit RuntimeValueStore(std::size_t size) : slots(size) {}

    void set(std::size_t id, RuntimeValue value) {
        if (id >= slots.size()) {
            slots.resize(id + 1);
        }
        slots[id] = std::move(value);
    }

    const RuntimeValue* get(std::size_t id) const {
        if (id >= slots.size() || !slots[id]) {
            return nullptr;
        }
        return &*slots[id];
    }

    RuntimeValue* get_mut(std::size_t id) {
        if (id >= slots.size() || !slots[id]) {
            return nullptr;
        }
        return &*slots[id];
    }

    void reset(std::size_t id) {
        if (id < slots.size()) {
            slots[id].reset();
        }
    }

    // ExecutionPlan validation keeps value ids dense and index-matched, so a
    // vector is cheaper than map lookup for the local executor's hot path.
    std::vector<std::optional<RuntimeValue>> slots;
};

struct RuntimeExecutionState {
    explicit RuntimeExecutionState(std::size_t value_count) : values(value_count) {}

    RuntimeValueStore values;
    std::map<std::size_t, RuntimeValue> outputs;
};

Diagnostic runtime_error(std::string message) {
    return Diagnostic::error("runtime", "R0001", std::move(message));
}

std::variant<double, Diagnostic> require_number(const RuntimeValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return static_cast<double>(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item ? 1.0 : 0.0;
    }
    return runtime_error("Graph executor expected scalar value");
}

std::variant<std::int64_t, Diagnostic> require_int(const RuntimeValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return *item;
    }
    return runtime_error("Graph executor expected integer value");
}

std::variant<bool, Diagnostic> require_bool(const RuntimeValue& value) {
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item;
    }
    return runtime_error("Graph executor expected bool value");
}

std::variant<const SimpleTensor*, Diagnostic> require_tensor_ref(const RuntimeValue& value) {
    if (const auto* item = std::get_if<SimpleTensor>(&value)) {
        return item;
    }
    return runtime_error("Graph executor expected tensor value");
}

const SimpleTensor& tensor_ref(const std::variant<const SimpleTensor*, Diagnostic>& value) {
    return *std::get<const SimpleTensor*>(value);
}

std::variant<RuntimeValue, Diagnostic> require_value(
    const RuntimeValueStore& values,
    std::size_t id,
    std::string label
) {
    const RuntimeValue* found = values.get(id);
    if (found == nullptr) {
        return runtime_error("Missing " + std::move(label));
    }
    return *found;
}

std::variant<const RuntimeValue*, Diagnostic> require_value_ref(
    const RuntimeValueStore& values,
    std::size_t id,
    std::string label
) {
    const RuntimeValue* found = values.get(id);
    if (found == nullptr) {
        return runtime_error("Missing " + std::move(label));
    }
    return found;
}

const RuntimeValue& runtime_value_ref(const std::variant<const RuntimeValue*, Diagnostic>& value) {
    return *std::get<const RuntimeValue*>(value);
}

std::variant<SimpleTensor, Diagnostic> make_tensor_argument(
    const PlanValue& value,
    const GraphExecutorOptions& options,
    RuntimeTensorWorkspace* workspace
) {
    if (value.placement != Placement::Host) {
        return runtime_error("Local executor requires host-resident tensor parameters");
    }
    auto shape = options.tensor_shapes.find(value.name);
    if (shape == options.tensor_shapes.end()) {
        return runtime_error("Missing --shape for tensor parameter '" + value.name + "'");
    }
    return make_synthetic_tensor(
        shape->second,
        value.type.tensor_dtype.value_or("float32"),
        workspace
    );
}

std::variant<GraphRuntimeValue, Diagnostic> to_graph_value(const RuntimeValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<SimpleTensor>(&value)) {
        return *item;
    }
    return runtime_error("Runtime interpreter cannot materialize callable values");
}

std::variant<RuntimeValue, Diagnostic> constant_to_runtime_value(const FeValue& value) {
    return std::visit(
        [](const auto& inner) -> std::variant<RuntimeValue, Diagnostic> {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                return RuntimeValue{inner};
            } else if constexpr (std::is_same_v<T, double>) {
                return RuntimeValue{inner};
            } else if constexpr (std::is_same_v<T, bool>) {
                return RuntimeValue{inner};
            } else {
                return runtime_error("Unsupported graph constant");
            }
        },
        value.value
    );
}

std::optional<OpId> resolved_op_id(const PlanOp& op) {
    if (op.resolved_op) {
        return op.resolved_op;
    }
    return lookup_op_id(op.op);
}

std::vector<std::size_t> compute_runtime_use_counts(const ExecutionPlan& plan) {
    std::vector<std::size_t> counts(plan.values.size(), 0);
    for (const auto& step : plan.steps) {
        if (step.kind == PlanStepKind::ExecuteOp && step.op_index) {
            for (const auto input_id : plan.ops[*step.op_index].inputs) {
                if (input_id < counts.size()) {
                    ++counts[input_id];
                }
            }
            continue;
        }
        if (step.kind == PlanStepKind::MaterializeOutput && step.value_id < counts.size()) {
            ++counts[step.value_id];
        }
    }
    return counts;
}

void release_value_slot(RuntimeValueStore& values, std::size_t id, RuntimeTensorWorkspace& workspace) {
    RuntimeValue* value = values.get_mut(id);
    if (value == nullptr) {
        return;
    }
    if (auto* tensor = std::get_if<SimpleTensor>(value)) {
        workspace.release(std::move(*tensor));
    }
    values.reset(id);
}

void release_consumed_inputs(
    RuntimeValueStore& values,
    const std::vector<std::size_t>& inputs,
    std::vector<std::size_t>& use_counts,
    RuntimeTensorWorkspace& workspace,
    bool collect_intermediate_values
) {
    if (collect_intermediate_values) {
        return;
    }
    for (const auto input_id : inputs) {
        if (input_id >= use_counts.size() || use_counts[input_id] == 0) {
            continue;
        }
        --use_counts[input_id];
        if (use_counts[input_id] == 0) {
            release_value_slot(values, input_id, workspace);
        }
    }
}

bool has_resolved_op(const PlanOp& op, OpId id) {
    const std::optional<OpId> resolved = resolved_op_id(op);
    return resolved && *resolved == id;
}

bool can_fuse_matmul_relu(
    const ExecutionPlan& plan,
    const std::vector<std::size_t>& use_counts,
    std::size_t step_index,
    bool collect_intermediate_values
) {
    if (collect_intermediate_values || step_index + 1 >= plan.steps.size()) {
        return false;
    }

    const PlanStep& matmul_step = plan.steps[step_index];
    const PlanStep& relu_step = plan.steps[step_index + 1];
    if (matmul_step.kind != PlanStepKind::ExecuteOp || relu_step.kind != PlanStepKind::ExecuteOp ||
        !matmul_step.op_index || !relu_step.op_index) {
        return false;
    }

    const PlanOp& matmul_op = plan.ops[*matmul_step.op_index];
    const PlanOp& relu_op = plan.ops[*relu_step.op_index];
    if (matmul_op.kind != PlanOpKind::PrimitiveCall || relu_op.kind != PlanOpKind::PrimitiveCall ||
        !has_resolved_op(matmul_op, OpId::Matmul) || !has_resolved_op(relu_op, OpId::Relu)) {
        return false;
    }
    if (matmul_op.inputs.size() != 2 || relu_op.inputs.size() != 1 || relu_op.inputs[0] != matmul_op.output) {
        return false;
    }
    if (matmul_op.output >= use_counts.size() || use_counts[matmul_op.output] != 1) {
        return false;
    }
    return true;
}

std::variant<RuntimeValue, Diagnostic> execute_matmul_relu_fusion(
    const PlanOp& matmul_op,
    const RuntimeValueStore& values,
    RuntimeTensorWorkspace& workspace
) {
    auto lhs_value = require_value_ref(values, matmul_op.inputs[0], "lhs");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_value)) {
        return *diagnostic;
    }
    auto rhs_value = require_value_ref(values, matmul_op.inputs[1], "rhs");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_value)) {
        return *diagnostic;
    }
    auto lhs = require_tensor_ref(runtime_value_ref(lhs_value));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
        return *diagnostic;
    }
    auto rhs = require_tensor_ref(runtime_value_ref(rhs_value));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
        return *diagnostic;
    }
    auto result = matmul_relu(tensor_ref(lhs), tensor_ref(rhs), &workspace);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        return *diagnostic;
    }
    return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
}

std::variant<RuntimeValue, Diagnostic> execute_primitive(
    const PlanOp& op,
    const RuntimeValueStore& values,
    RuntimeTensorWorkspace& workspace
) {
    const std::optional<OpId> op_id = resolved_op_id(op);
    if (!op_id) {
        return runtime_error("Unsupported primitive graph op '" + op.op + "'");
    }
    switch (*op_id) {
        case OpId::Matmul: {
            auto lhs_value = require_value_ref(values, op.inputs[0], "lhs");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_value)) {
                return *diagnostic;
            }
            auto rhs_value = require_value_ref(values, op.inputs[1], "rhs");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_value)) {
                return *diagnostic;
            }
            auto lhs = require_tensor_ref(runtime_value_ref(lhs_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
                return *diagnostic;
            }
            auto rhs = require_tensor_ref(runtime_value_ref(rhs_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
                return *diagnostic;
            }
            auto result = matmul(tensor_ref(lhs), tensor_ref(rhs), &workspace);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                return *diagnostic;
            }
            return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
        }
        case OpId::Relu: {
            auto input_value = require_value_ref(values, op.inputs[0], "input");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
                return *diagnostic;
            }
            auto input = require_tensor_ref(runtime_value_ref(input_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
                return *diagnostic;
            }
            return RuntimeValue{apply_relu(tensor_ref(input), &workspace)};
        }
        case OpId::Scale: {
            auto tensor_value = require_value_ref(values, op.inputs[0], "tensor");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&tensor_value)) {
                return *diagnostic;
            }
            auto scalar_value = require_value_ref(values, op.inputs[1], "scale");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar_value)) {
                return *diagnostic;
            }
            auto tensor = require_tensor_ref(runtime_value_ref(tensor_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&tensor)) {
                return *diagnostic;
            }
            auto scalar = require_number(runtime_value_ref(scalar_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) {
                return *diagnostic;
            }
            auto result = tensor_scalar_binary(FeBinaryOp::Mul, tensor_ref(tensor), std::get<double>(scalar), &workspace);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                return *diagnostic;
            }
            return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
        }
        default:
            return runtime_error("Unsupported primitive graph op '" + op.op + "'");
    }
}

std::variant<RuntimeValue, Diagnostic> execute_binary(
    const PlanOp& op,
    const RuntimeValueStore& values,
    RuntimeTensorWorkspace& workspace
) {
    auto lhs_value = require_value_ref(values, op.inputs[0], "lhs");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs_value)) {
        return *diagnostic;
    }
    auto rhs_value = require_value_ref(values, op.inputs[1], "rhs");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs_value)) {
        return *diagnostic;
    }
    const RuntimeValue& lhs = runtime_value_ref(lhs_value);
    const RuntimeValue& rhs = runtime_value_ref(rhs_value);

    if (const auto* lhs_tensor = std::get_if<SimpleTensor>(&lhs)) {
        if (const auto* rhs_tensor = std::get_if<SimpleTensor>(&rhs)) {
            auto result = elementwise_binary(op.binary_op, *lhs_tensor, *rhs_tensor, &workspace);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                return *diagnostic;
            }
            return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
        }
        auto scalar = require_number(rhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) {
            return *diagnostic;
        }
        auto result = tensor_scalar_binary(op.binary_op, *lhs_tensor, std::get<double>(scalar), &workspace);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }

    if (const auto* rhs_tensor = std::get_if<SimpleTensor>(&rhs)) {
        auto scalar = require_number(lhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&scalar)) {
            return *diagnostic;
        }
        auto result = scalar_tensor_binary(op.binary_op, std::get<double>(scalar), *rhs_tensor, &workspace);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }

    auto left = require_number(lhs);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
        return *diagnostic;
    }
    auto right = require_number(rhs);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
        return *diagnostic;
    }
    const double lhs_number = std::get<double>(left);
    const double rhs_number = std::get<double>(right);
    switch (op.binary_op) {
        case FeBinaryOp::Add:
            return RuntimeValue{lhs_number + rhs_number};
        case FeBinaryOp::Sub:
            return RuntimeValue{lhs_number - rhs_number};
        case FeBinaryOp::Mul:
            return RuntimeValue{lhs_number * rhs_number};
        case FeBinaryOp::Div:
            return RuntimeValue{lhs_number / rhs_number};
        case FeBinaryOp::FloorDiv:
            return RuntimeValue{std::floor(lhs_number / rhs_number)};
        case FeBinaryOp::Eq:
        case FeBinaryOp::NotEq:
        case FeBinaryOp::Lt:
        case FeBinaryOp::Gt:
        case FeBinaryOp::LtEq:
        case FeBinaryOp::GtEq:
        case FeBinaryOp::And:
        case FeBinaryOp::Or:
        case FeBinaryOp::Not:
            return runtime_error("Unsupported scalar graph binary op");
    }
    return runtime_error("Unsupported scalar graph binary op");
}

template <typename Function>
std::variant<RuntimeValue, Diagnostic> unary_tensor_result(
    const PlanOp& op,
    const RuntimeValueStore& values,
    const std::string& label,
    RuntimeTensorWorkspace& workspace,
    Function function
) {
    auto input_value = require_value_ref(values, op.inputs[0], label);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
        return *diagnostic;
    }
    auto input = require_tensor_ref(runtime_value_ref(input_value));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
        return *diagnostic;
    }
    auto result = function(tensor_ref(input), &workspace);
    if constexpr (std::is_same_v<decltype(result), SimpleTensor>) {
        return RuntimeValue{result};
    } else {
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
}

std::variant<RuntimeValue, Diagnostic> execute_library_call(
    const PlanOp& op,
    const RuntimeValueStore& values,
    RuntimeTensorWorkspace& workspace
) {
    const std::optional<OpId> op_id = resolved_op_id(op);
    if (!op_id) {
        return runtime_error("Unsupported library graph call '" + op.op + "'");
    }
    if (*op_id == OpId::RmsNorm) {
        auto input_value = require_value_ref(values, op.inputs[0], "tensor");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
            return *diagnostic;
        }
        auto hidden_value = require_value_ref(values, op.inputs[1], "hidden size");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&hidden_value)) {
            return *diagnostic;
        }
        auto input = require_tensor_ref(runtime_value_ref(input_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
            return *diagnostic;
        }
        auto hidden = require_int(runtime_value_ref(hidden_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&hidden)) {
            return *diagnostic;
        }
        auto result = apply_rms_norm(tensor_ref(input), std::get<std::int64_t>(hidden), &workspace);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (*op_id == OpId::CrossEntropy) {
        auto logits_value = require_value_ref(values, op.inputs[0], "logits");
        auto target_value = require_value_ref(values, op.inputs[1], "target");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&logits_value)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&target_value)) {
            return *diagnostic;
        }
        auto logits = require_tensor_ref(runtime_value_ref(logits_value));
        auto target = require_tensor_ref(runtime_value_ref(target_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&logits)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&target)) {
            return *diagnostic;
        }
        auto result = apply_cross_entropy(
            tensor_ref(logits),
            tensor_ref(target),
            &workspace
        );
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (*op_id == OpId::Reshape) {
        std::vector<std::int64_t> shape;
        for (std::size_t index = 1; index < op.inputs.size(); ++index) {
            auto dim_value = require_value_ref(values, op.inputs[index], "reshape dim");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&dim_value)) {
                return *diagnostic;
            }
            auto dim = require_int(runtime_value_ref(dim_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&dim)) {
                return *diagnostic;
            }
            shape.push_back(std::get<std::int64_t>(dim));
        }
        auto input_value = require_value_ref(values, op.inputs[0], "reshape input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
            return *diagnostic;
        }
        auto input = require_tensor_ref(runtime_value_ref(input_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
            return *diagnostic;
        }
        auto result = apply_reshape(tensor_ref(input), shape, &workspace);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (*op_id == OpId::Transpose) {
        return unary_tensor_result(op, values, "transpose input", workspace, apply_transpose);
    }
    if (*op_id == OpId::Sum || *op_id == OpId::Mean) {
        auto input_value = require_value_ref(values, op.inputs[0], op.op + " input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
            return *diagnostic;
        }
        auto input = require_tensor_ref(runtime_value_ref(input_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
            return *diagnostic;
        }
        if (op.inputs.size() > 1) {
            auto axis_value = require_value_ref(values, op.inputs[1], op.op + " axis");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&axis_value)) {
                return *diagnostic;
            }
            auto axis = require_int(runtime_value_ref(axis_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&axis)) {
                return *diagnostic;
            }
            auto result = *op_id == OpId::Sum
                ? apply_sum_axis(tensor_ref(input), std::get<std::int64_t>(axis), &workspace)
                : apply_mean_axis(tensor_ref(input), std::get<std::int64_t>(axis), &workspace);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                return *diagnostic;
            }
            return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
        }
        return RuntimeValue{
            *op_id == OpId::Sum
                ? apply_sum(tensor_ref(input), &workspace)
                : apply_mean(tensor_ref(input), &workspace)
        };
    }
    if (*op_id == OpId::Sqrt) {
        return unary_tensor_result(op, values, "sqrt input", workspace, apply_sqrt);
    }
    if (*op_id == OpId::Rsqrt) {
        return unary_tensor_result(op, values, "rsqrt input", workspace, apply_rsqrt);
    }
    if (*op_id == OpId::RepeatKv) {
        auto input_value = require_value_ref(values, op.inputs[0], "repeat_kv input");
        auto repeats_value = require_value_ref(values, op.inputs[1], "repeats");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&repeats_value)) {
            return *diagnostic;
        }
        auto input = require_tensor_ref(runtime_value_ref(input_value));
        auto repeats = require_int(runtime_value_ref(repeats_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&repeats)) {
            return *diagnostic;
        }
        auto result = apply_repeat_kv(tensor_ref(input), std::get<std::int64_t>(repeats), &workspace);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (*op_id == OpId::FlattenHeads) {
        return unary_tensor_result(op, values, "flatten_heads input", workspace, apply_flatten_heads);
    }
    if (*op_id == OpId::CausalMask) {
        return unary_tensor_result(op, values, "causal_mask input", workspace, apply_causal_mask);
    }
    if (*op_id == OpId::Rope) {
        auto input_value = require_value_ref(values, op.inputs[0], "rope input");
        auto head_dim_value = require_value_ref(values, op.inputs[1], "rope head_dim");
        auto theta_value = require_value_ref(values, op.inputs[2], "rope theta");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&head_dim_value)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&theta_value)) {
            return *diagnostic;
        }
        auto input = require_tensor_ref(runtime_value_ref(input_value));
        auto head_dim = require_int(runtime_value_ref(head_dim_value));
        auto theta = require_number(runtime_value_ref(theta_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&head_dim)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&theta)) {
            return *diagnostic;
        }
        auto result = apply_rope(
            tensor_ref(input),
            std::get<std::int64_t>(head_dim),
            std::get<double>(theta),
            &workspace
        );
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    return runtime_error("Unsupported library graph call '" + op.op + "'");
}

std::variant<RuntimeValue, Diagnostic> execute_library_ctor(
    const PlanOp& op,
    const PlanValue& output,
    const RuntimeValueStore& values
) {
    const std::optional<OpId> op_id = resolved_op_id(op);
    if (!op_id) {
        return runtime_error("Unsupported library constructor '" + op.op + "'");
    }
    if (*op_id == OpId::Linear) {
        LinearClosure closure;
        if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
            closure.dtype = *output.type.callable_return->tensor_dtype;
        }
        if (op.inputs.size() == 1) {
            auto out_features_value = require_value(values, op.inputs[0], "out_features");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features_value)) {
                return *diagnostic;
            }
            auto out_features = require_int(std::get<RuntimeValue>(out_features_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
                return *diagnostic;
            }
            closure.out_features = std::get<std::int64_t>(out_features);
            return RuntimeValue{closure};
        }
        if (op.inputs.size() == 2) {
            auto second = require_value(values, op.inputs[1], "linear argument");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&second)) {
                return *diagnostic;
            }
            if (std::holds_alternative<bool>(std::get<RuntimeValue>(second))) {
                auto out_features_value = require_value(values, op.inputs[0], "out_features");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features_value)) {
                    return *diagnostic;
                }
                auto out_features = require_int(std::get<RuntimeValue>(out_features_value));
                if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
                    return *diagnostic;
                }
                closure.out_features = std::get<std::int64_t>(out_features);
                closure.with_bias = std::get<bool>(std::get<RuntimeValue>(second));
                return RuntimeValue{closure};
            }
        }
        auto in_features_value = require_value(values, op.inputs[0], "in_features");
        auto out_features_value = require_value(values, op.inputs[1], "out_features");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&in_features_value)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features_value)) {
            return *diagnostic;
        }
        auto in_features = require_int(std::get<RuntimeValue>(in_features_value));
        auto out_features = require_int(std::get<RuntimeValue>(out_features_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&in_features)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&out_features)) {
            return *diagnostic;
        }
        closure.in_features = std::get<std::int64_t>(in_features);
        closure.out_features = std::get<std::int64_t>(out_features);
        if (op.inputs.size() == 3) {
            auto with_bias_value = require_value(values, op.inputs[2], "with_bias");
            if (const auto* diagnostic = std::get_if<Diagnostic>(&with_bias_value)) {
                return *diagnostic;
            }
            auto with_bias = require_bool(std::get<RuntimeValue>(with_bias_value));
            if (const auto* diagnostic = std::get_if<Diagnostic>(&with_bias)) {
                return *diagnostic;
            }
            closure.with_bias = std::get<bool>(with_bias);
        }
        return RuntimeValue{closure};
    }
    if (*op_id == OpId::Silu || *op_id == OpId::Gelu || *op_id == OpId::Tanh ||
        *op_id == OpId::Sigmoid || *op_id == OpId::Softmax) {
        return RuntimeValue{ActivationClosure{*op_id, 0.0}};
    }
    if (*op_id == OpId::Dropout) {
        auto probability_value = require_value(values, op.inputs[0], "probability");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&probability_value)) {
            return *diagnostic;
        }
        auto probability = require_number(std::get<RuntimeValue>(probability_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&probability)) {
            return *diagnostic;
        }
        return RuntimeValue{ActivationClosure{OpId::Dropout, std::get<double>(probability)}};
    }
    if (*op_id == OpId::Embedding) {
        auto num_value = require_value(values, op.inputs[0], "num_embeddings");
        auto dim_value = require_value(values, op.inputs[1], "embedding_dim");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&num_value)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&dim_value)) {
            return *diagnostic;
        }
        auto num = require_int(std::get<RuntimeValue>(num_value));
        auto dim = require_int(std::get<RuntimeValue>(dim_value));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&num)) {
            return *diagnostic;
        }
        if (const auto* diagnostic = std::get_if<Diagnostic>(&dim)) {
            return *diagnostic;
        }
        EmbeddingClosure closure;
        closure.num_embeddings = std::get<std::int64_t>(num);
        closure.embedding_dim = std::get<std::int64_t>(dim);
        if (output.type.callable_return && output.type.callable_return->tensor_dtype) {
            closure.dtype = *output.type.callable_return->tensor_dtype;
        }
        return RuntimeValue{closure};
    }
    return runtime_error("Unsupported library constructor '" + op.op + "'");
}

std::variant<RuntimeValue, Diagnostic> execute_apply(
    const PlanOp& op,
    const PlanValue& output,
    const RuntimeValueStore& values,
    RuntimeTensorWorkspace& workspace
) {
    auto callee_value = require_value_ref(values, op.inputs[0], "apply callee");
    auto input_value = require_value_ref(values, op.inputs[1], "apply input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&callee_value)) {
        return *diagnostic;
    }
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input_value)) {
        return *diagnostic;
    }
    auto input = require_tensor_ref(runtime_value_ref(input_value));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
        return *diagnostic;
    }

    const RuntimeValue& callee = runtime_value_ref(callee_value);
    const SimpleTensor& tensor = tensor_ref(input);
    if (const auto* closure = std::get_if<LinearClosure>(&callee)) {
        auto result = apply_linear(*closure, tensor, &workspace);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (const auto* closure = std::get_if<EmbeddingClosure>(&callee)) {
        const std::string output_dtype = output.type.tensor_dtype.value_or(closure->dtype);
        SimpleTensor weight = make_embedding_weight(closure->num_embeddings, closure->embedding_dim, output_dtype, &workspace);
        auto result = apply_embedding_with_parameters(tensor, weight, closure->num_embeddings, closure->embedding_dim, &workspace);
        workspace.release(std::move(weight));
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            return *diagnostic;
        }
        return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
    }
    if (const auto* closure = std::get_if<ActivationClosure>(&callee)) {
        switch (closure->op) {
            case OpId::Silu:
                return RuntimeValue{apply_silu(tensor, &workspace)};
            case OpId::Gelu:
                return RuntimeValue{apply_gelu(tensor, &workspace)};
            case OpId::Tanh:
                return RuntimeValue{apply_tanh(tensor, &workspace)};
            case OpId::Sigmoid:
                return RuntimeValue{apply_sigmoid(tensor, &workspace)};
            case OpId::Softmax: {
                auto result = apply_softmax(tensor, &workspace);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                    return *diagnostic;
                }
                return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
            }
            case OpId::Dropout: {
                auto result = apply_dropout(tensor, closure->probability, &workspace);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
                    return *diagnostic;
                }
                return RuntimeValue{std::get<SimpleTensor>(std::move(result))};
            }
            default:
                break;
        }
    }
    return runtime_error("Unsupported graph apply operation");
}

const ExecutionPlan* find_plan(const PlanModule* module, const std::string& name) {
    if (module == nullptr) {
        return nullptr;
    }
    auto found = std::find_if(module->plans.begin(), module->plans.end(), [&](const ExecutionPlan& plan) {
        return plan.name == name;
    });
    return found == module->plans.end() ? nullptr : &*found;
}

std::variant<RuntimeExecutionState, Diagnostic> execute_plan_internal(
    const ExecutionPlan& plan,
    const GraphExecutorOptions& options,
    const PlanModule* module,
    const std::vector<RuntimeValue>& arguments,
    std::set<std::string>& active_calls,
    RuntimeTensorWorkspace& workspace
) {
    if (plan.backend != BackendKind::Local) {
        return runtime_error("Execution plan backend is not implemented yet");
    }
    if (active_calls.find(plan.name) != active_calls.end()) {
        return runtime_error("Recursive graph execution is not supported for function '" + plan.name + "'");
    }
    active_calls.insert(plan.name);

    RuntimeExecutionState state(plan.values.size());
    std::vector<std::size_t> use_counts = compute_runtime_use_counts(plan);
    std::size_t argument_index = 0;

    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const PlanStep& step = plan.steps[step_index];
        switch (step.kind) {
            case PlanStepKind::AllocateHostValue: {
                const PlanValue& value = plan.values[step.value_id];
                if (value.placement != Placement::Host) {
                    return runtime_error("AllocateHostValue step requires a host-resident value");
                }
                if (value.is_parameter) {
                    if (argument_index < arguments.size()) {
                        state.values.set(value.id, arguments[argument_index++]);
                    } else {
                        if (value.type.kind != FeTypeKind::Tensor) {
                            return runtime_error("Graph executor currently supports tensor parameters only");
                        }
                        auto tensor = make_tensor_argument(value, options, &workspace);
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&tensor)) {
                            return *diagnostic;
                        }
                        state.values.set(value.id, std::get<SimpleTensor>(std::move(tensor)));
                    }
                }
                break;
            }
            case PlanStepKind::AllocateDeviceValue:
                return runtime_error("Local executor does not support device value allocation");
            case PlanStepKind::ExecuteOp: {
                if (!step.op_index) {
                    return runtime_error("ExecuteOp step is missing an op index");
                }
                const PlanOp& op = plan.ops[*step.op_index];
                if (op.backend != BackendKind::Local) {
                    return runtime_error("Local executor cannot run non-local plan operations");
                }

                if (can_fuse_matmul_relu(plan, use_counts, step_index, options.collect_intermediate_values)) {
                    const PlanOp& relu_op = plan.ops[*plan.steps[step_index + 1].op_index];
                    auto fused_result = execute_matmul_relu_fusion(op, state.values, workspace);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&fused_result)) {
                        return *diagnostic;
                    }
                    state.values.set(relu_op.output, std::get<RuntimeValue>(std::move(fused_result)));
                    release_consumed_inputs(
                        state.values,
                        op.inputs,
                        use_counts,
                        workspace,
                        options.collect_intermediate_values
                    );
                    release_consumed_inputs(
                        state.values,
                        relu_op.inputs,
                        use_counts,
                        workspace,
                        options.collect_intermediate_values
                    );
                    ++step_index;
                    break;
                }

                std::optional<std::variant<RuntimeValue, Diagnostic>> result;
                if (op.kind == PlanOpKind::LibraryCall && module != nullptr && !op.op_id && !op.resolved_op) {
                    const ExecutionPlan* callee = find_plan(module, op.op);
                    if (callee != nullptr) {
                        std::vector<RuntimeValue> call_args;
                        call_args.reserve(op.inputs.size());
                        for (const auto input_id : op.inputs) {
                            auto input = require_value(state.values, input_id, "function call argument");
                            if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
                                return *diagnostic;
                            }
                            call_args.push_back(std::get<RuntimeValue>(std::move(input)));
                        }
                        auto call_result = execute_plan_internal(*callee, options, module, call_args, active_calls, workspace);
                        if (const auto* diagnostic = std::get_if<Diagnostic>(&call_result)) {
                            return *diagnostic;
                        }
                        RuntimeExecutionState child_state = std::get<RuntimeExecutionState>(std::move(call_result));
                        if (callee->outputs.size() != 1 || child_state.outputs.empty()) {
                            return runtime_error("Runtime interpreter currently supports a single function-call return value");
                        }
                        result = std::move(child_state.outputs.begin()->second);
                    }
                }

                if (!result) {
                    switch (op.kind) {
                        case PlanOpKind::Constant:
                            result = constant_to_runtime_value(op.constant);
                            break;
                        case PlanOpKind::Binary:
                            result = execute_binary(op, state.values, workspace);
                            break;
                        case PlanOpKind::PrimitiveCall:
                            result = execute_primitive(op, state.values, workspace);
                            break;
                        case PlanOpKind::LibraryCall:
                            result = execute_library_call(op, state.values, workspace);
                            break;
                        case PlanOpKind::LibraryCtor:
                            result = execute_library_ctor(op, plan.values[op.output], state.values);
                            break;
                        case PlanOpKind::Apply:
                            result = execute_apply(op, plan.values[op.output], state.values, workspace);
                            break;
                    }
                }
                if (const auto* diagnostic = std::get_if<Diagnostic>(&*result)) {
                    return *diagnostic;
                }
                state.values.set(op.output, std::get<RuntimeValue>(std::move(*result)));
                release_consumed_inputs(
                    state.values,
                    op.inputs,
                    use_counts,
                    workspace,
                    options.collect_intermediate_values
                );
                break;
            }
            case PlanStepKind::MaterializeOutput: {
                if (options.collect_intermediate_values) {
                    auto found = require_value(state.values, step.value_id, "graph output value");
                    if (std::get_if<Diagnostic>(&found) != nullptr) {
                        return runtime_error("Runtime interpreter could not resolve the graph output value");
                    }
                    state.outputs[step.value_id] = std::get<RuntimeValue>(std::move(found));
                } else {
                    RuntimeValue* found = state.values.get_mut(step.value_id);
                    if (found == nullptr) {
                        return runtime_error("Runtime interpreter could not resolve the graph output value");
                    }
                    state.outputs[step.value_id] = std::move(*found);
                    state.values.reset(step.value_id);
                }
                break;
            }
            case PlanStepKind::UploadToDevice:
            case PlanStepKind::DispatchDeviceOp:
            case PlanStepKind::DownloadToHost:
                return runtime_error("Device execution plan steps are not implemented yet");
        }
    }

    active_calls.erase(plan.name);
    return state;
}

GraphExecutionResultVariant public_result_from_state(RuntimeExecutionState state, bool collect_intermediate_values) {
    GraphExecutionResult result;
    if (collect_intermediate_values) {
        for (std::size_t id = 0; id < state.values.slots.size(); ++id) {
            if (!state.values.slots[id]) {
                continue;
            }
            auto converted = to_graph_value(*state.values.slots[id]);
            if (std::holds_alternative<Diagnostic>(converted)) {
                continue;
            }
            result.values[id] = std::get<GraphRuntimeValue>(std::move(converted));
        }
    }
    for (auto& item : state.outputs) {
        auto converted = to_graph_value(item.second);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&converted)) {
            return *diagnostic;
        }
        result.outputs[item.first] = std::get<GraphRuntimeValue>(std::move(converted));
    }
    return result;
}

} // namespace

GraphExecutionResultVariant execute_execution_plan(const ExecutionPlan& plan, const GraphExecutorOptions& options) {
    std::set<std::string> active_calls;
    RuntimeTensorWorkspace local_workspace;
    RuntimeTensorWorkspace& workspace = options.tensor_workspace != nullptr ? *options.tensor_workspace : local_workspace;
    auto state = execute_plan_internal(plan, options, nullptr, {}, active_calls, workspace);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&state)) {
        return *diagnostic;
    }
    return public_result_from_state(std::get<RuntimeExecutionState>(std::move(state)), options.collect_intermediate_values);
}

GraphExecutionResultVariant execute_plan_module(
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
) {
    const ExecutionPlan* plan = find_plan(&module, entry);
    if (plan == nullptr) {
        return runtime_error("Entry function '" + entry + "' not found in execution plan module");
    }
    std::set<std::string> active_calls;
    RuntimeTensorWorkspace local_workspace;
    RuntimeTensorWorkspace& workspace = options.tensor_workspace != nullptr ? *options.tensor_workspace : local_workspace;
    auto state = execute_plan_internal(*plan, options, &module, {}, active_calls, workspace);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&state)) {
        return *diagnostic;
    }
    return public_result_from_state(std::get<RuntimeExecutionState>(std::move(state)), options.collect_intermediate_values);
}

void print_graph_runtime_value(const GraphRuntimeValue& value) {
    if (const auto* tensor = std::get_if<SimpleTensor>(&value)) {
        print_tensor(*tensor);
        return;
    }
    std::cout << "\n--- Execution Output ---\n";
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        std::cout << "value=" << *item << '\n';
    } else if (const auto* item = std::get_if<double>(&value)) {
        std::cout << "value=" << *item << '\n';
    } else if (const auto* item = std::get_if<bool>(&value)) {
        std::cout << "value=" << (*item ? "true" : "false") << '\n';
    }
    std::cout << "------------------------\n";
}
