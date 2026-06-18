#include "execution_plan.h"

#include "ops.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace {

const char* plan_backend_name(BackendKind backend);

Diagnostic plan_error(std::string message) {
    return Diagnostic::error("backend", "B0001", std::move(message))
        .with_help("Execution plan validation failed before runtime/backend execution.");
}

Diagnostic capability_error(BackendKind backend, std::size_t op_index, std::string reason) {
    return Diagnostic::error(
               "backend",
               "B0001",
               std::string("Backend capability error in ") + plan_backend_name(backend) +
                   " op #" + std::to_string(op_index) + ": " + reason
           )
        .with_help("The frontend accepted this program, but the selected backend cannot execute one of the planned operations yet.");
}

const std::vector<std::string>& primitive_ops() {
    static const std::vector<std::string> names{"matmul", "relu", "scale"};
    return names;
}

const std::vector<std::string>& primitive_op_ids() {
    static const std::vector<std::string> names{"Matmul", "Relu", "Scale"};
    return names;
}

const std::vector<std::string>& library_ops() {
    static const std::vector<std::string> names{
        "rms_norm",
        "cross_entropy",
        "reshape",
        "transpose",
        "sum",
        "mean",
        "sqrt",
        "rsqrt",
        "repeat_kv",
        "flatten_heads",
        "causal_mask",
        "rope",
    };
    return names;
}

const std::vector<std::string>& library_op_ids() {
    static const std::vector<std::string> names{
        "RmsNorm",
        "CrossEntropy",
        "Reshape",
        "Transpose",
        "Sum",
        "Mean",
        "Sqrt",
        "Rsqrt",
        "RepeatKv",
        "FlattenHeads",
        "CausalMask",
        "Rope",
    };
    return names;
}

const std::vector<std::string>& constructor_ops() {
    static const std::vector<std::string> names{
        "linear",
        "Embedding",
        "SiLU",
        "GELU",
        "Tanh",
        "Sigmoid",
        "Softmax",
        "Dropout",
    };
    return names;
}

const std::vector<std::string>& constructor_op_ids() {
    static const std::vector<std::string> names{
        "Linear",
        "Embedding",
        "Silu",
        "Gelu",
        "Tanh",
        "Sigmoid",
        "Softmax",
        "Dropout",
    };
    return names;
}

bool contains_string(const std::vector<std::string>& values, const std::string& target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}

CapabilityCheck supported() {
    return CapabilityCheck{CapabilityStatus::Supported, std::nullopt};
}

CapabilityCheck unsupported(std::string reason) {
    return CapabilityCheck{CapabilityStatus::Unsupported, std::move(reason)};
}

CapabilityCheck check_named_op(
    BackendKind backend,
    const std::string& category,
    const PlanOp& op,
    const std::vector<std::string>& supported_names,
    const std::vector<std::string>& supported_ids
) {
    if (op.resolved_op && contains_string(supported_ids, op_id_name(*op.resolved_op))) {
        return supported();
    }
    if (op.op_id && contains_string(supported_ids, *op.op_id)) {
        return supported();
    }
    auto resolved_id = lookup_op_id(op.op);
    if (resolved_id && contains_string(supported_ids, op_id_name(*resolved_id))) {
        return supported();
    }
    if (contains_string(supported_names, op.op)) {
        return supported();
    }
    return unsupported(
        std::string(plan_backend_name(backend)) + " backend does not support " +
        category + " op '" + op.op + "'"
    );
}

CapabilityCheck check_plan_op_capability_with_callables(
    BackendKind backend,
    const PlanOp& op,
    const std::vector<std::string>& callable_functions
) {
    if (op.kind == PlanOpKind::LibraryCall &&
        !op.op_id &&
        !op.resolved_op &&
        contains_string(callable_functions, op.op)) {
        return supported();
    }
    return check_plan_op_capability(backend, op);
}

std::optional<OpId> resolved_plan_op_id(const PlanOp& op) {
    if (op.resolved_op) {
        return op.resolved_op;
    }
    return lookup_op_id(op.op);
}

bool has_resolved_op(const PlanOp& op, OpId id) {
    const std::optional<OpId> resolved = resolved_plan_op_id(op);
    return resolved && *resolved == id;
}

std::vector<std::size_t> compute_plan_use_counts(const ExecutionPlan& plan) {
    std::vector<std::size_t> counts(plan.values.size(), 0);
    for (const auto& op : plan.ops) {
        for (const auto input : op.inputs) {
            if (input < counts.size()) {
                ++counts[input];
            }
        }
    }
    for (const auto output : plan.outputs) {
        if (output < counts.size()) {
            ++counts[output];
        }
    }
    return counts;
}

bool can_fuse_matmul_relu(
    const ExecutionPlan& plan,
    const std::vector<std::size_t>& use_counts,
    std::size_t step_index
) {
    if (step_index + 1 >= plan.steps.size()) {
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
    if (matmul_op.backend != BackendKind::Local || relu_op.backend != BackendKind::Local ||
        matmul_op.kind != PlanOpKind::PrimitiveCall || relu_op.kind != PlanOpKind::PrimitiveCall ||
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

PlanOp make_matmul_relu_op(const PlanOp& matmul_op, const PlanOp& relu_op) {
    return PlanOp{
        PlanOpKind::PrimitiveCall,
        relu_op.output,
        "matmul_relu",
        std::optional<std::string>{"MatmulRelu"},
        FeBinaryOp::Add,
        FeValue::none(),
        matmul_op.inputs,
        BackendKind::Local,
        std::optional<OpId>{OpId::MatmulRelu},
    };
}

PlanOpKind lower_plan_kind(GraphNodeKind kind) {
    switch (kind) {
        case GraphNodeKind::Constant:
            return PlanOpKind::Constant;
        case GraphNodeKind::Binary:
            return PlanOpKind::Binary;
        case GraphNodeKind::PrimitiveCall:
            return PlanOpKind::PrimitiveCall;
        case GraphNodeKind::LibraryCall:
            return PlanOpKind::LibraryCall;
        case GraphNodeKind::LibraryCtor:
            return PlanOpKind::LibraryCtor;
        case GraphNodeKind::Apply:
            return PlanOpKind::Apply;
    }
    return PlanOpKind::Constant;
}

PlanValue lower_plan_value(const GraphValue& value, Placement placement) {
    return PlanValue{
        value.id,
        value.name,
        value.type,
        value.is_parameter,
        value.requires_grad,
        placement,
        value.tensor_type,
    };
}

PlanParameter lower_plan_parameter(const GraphParameter& parameter) {
    return PlanParameter{
        parameter.name,
        parameter.role,
        parameter.owner_value,
        parameter.tensor_type,
        parameter.trainable,
    };
}

PlanOp lower_plan_op(const GraphNode& node, BackendKind backend) {
    return PlanOp{
        lower_plan_kind(node.kind),
        node.output,
        node.op,
        node.op_id,
        node.binary_op,
        node.constant,
        node.inputs,
        backend,
        lookup_op_id(node.op),
    };
}

ExecutionPlan make_base_plan(const GraphFunction& graph, BackendKind backend) {
    ExecutionPlan plan;
    plan.backend = backend;
    plan.name = graph.name;
    plan.return_type = graph.return_type;
    plan.outputs = graph.outputs;
    plan.named_values = graph.named_values;
    plan.parameters.reserve(graph.parameters.size());
    for (const auto& parameter : graph.parameters) {
        plan.parameters.push_back(lower_plan_parameter(parameter));
    }
    return plan;
}

void append_local_allocate_steps(ExecutionPlan& plan) {
    for (const auto& value : plan.values) {
        plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, value.id, std::nullopt});
    }
}

const char* placement_name(Placement placement) {
    switch (placement) {
        case Placement::Host:
            return "host";
        case Placement::Device:
            return "device";
    }
    return "host";
}

const char* plan_backend_name(BackendKind backend) {
    switch (backend) {
        case BackendKind::Local:
            return "local";
        case BackendKind::Metal:
            return "metal";
        case BackendKind::PyTorch:
            return "pytorch";
        case BackendKind::Cuda:
            return "cuda";
        case BackendKind::Rocm:
            return "rocm";
    }
    return "local";
}

const char* plan_op_kind_name(PlanOpKind kind) {
    switch (kind) {
        case PlanOpKind::Constant:
            return "constant";
        case PlanOpKind::Binary:
            return "binary";
        case PlanOpKind::PrimitiveCall:
            return "primitive_call";
        case PlanOpKind::LibraryCall:
            return "library_call";
        case PlanOpKind::LibraryCtor:
            return "library_ctor";
        case PlanOpKind::Apply:
            return "apply";
    }
    return "unknown";
}

const char* plan_step_kind_name(PlanStepKind kind) {
    switch (kind) {
        case PlanStepKind::AllocateHostValue:
            return "allocate_host";
        case PlanStepKind::AllocateDeviceValue:
            return "allocate_device";
        case PlanStepKind::ExecuteOp:
            return "execute_op";
        case PlanStepKind::MaterializeOutput:
            return "materialize_output";
        case PlanStepKind::UploadToDevice:
            return "upload_to_device";
        case PlanStepKind::DispatchDeviceOp:
            return "dispatch_device_op";
        case PlanStepKind::DownloadToHost:
            return "download_to_host";
    }
    return "unknown";
}

std::string fe_type_to_plan_string(const FeType& type) {
    switch (type.kind) {
        case FeTypeKind::Unknown:
            return "unknown";
        case FeTypeKind::Int:
            return type.scalar_dtype.value_or("int");
        case FeTypeKind::Float:
            return type.scalar_dtype.value_or("float");
        case FeTypeKind::Bool:
            return "bool";
        case FeTypeKind::str:
            return "str";
        case FeTypeKind::Tensor:
            if (type.tensor_dtype && type.tensor_shape_expr) {
                return "tensor[" + *type.tensor_dtype + ", " + *type.tensor_shape_expr + "]";
            }
            if (type.tensor_dtype) {
                return "tensor[" + *type.tensor_dtype + "]";
            }
            if (type.tensor_shape_expr) {
                return "tensor[" + *type.tensor_shape_expr + "]";
            }
            return "tensor";
        case FeTypeKind::Tuple: {
            std::ostringstream out;
            out << '(';
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    out << ", ";
                }
                out << fe_type_to_plan_string(type.elements[index]);
            }
            out << ')';
            return out.str();
        }
        case FeTypeKind::List: {
            std::ostringstream out;
            out << '[';
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    out << ", ";
                }
                out << fe_type_to_plan_string(type.elements[index]);
            }
            out << ']';
            return out.str();
        }
        case FeTypeKind::Callable:
            return "callable -> " + (type.callable_return ? fe_type_to_plan_string(*type.callable_return) : "void");
        case FeTypeKind::Void:
            return "void";
        case FeTypeKind::None:
            return "none";
    }
    return "unknown";
}

std::string fe_value_to_plan_string(const FeValue& value) {
    return std::visit(
        [](const auto& inner) -> std::string {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return "None";
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(inner);
            } else if constexpr (std::is_same_v<T, double>) {
                std::ostringstream out;
                out << inner;
                return out.str();
            } else if constexpr (std::is_same_v<T, bool>) {
                return inner ? "true" : "false";
            } else if constexpr (std::is_same_v<T, std::string>) {
                return '"' + inner + '"';
            } else if constexpr (std::is_same_v<T, FeTupleValue>) {
                std::ostringstream out;
                out << '[';
                for (std::size_t index = 0; index < inner.values.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    out << fe_value_to_plan_string(inner.values[index]);
                }
                out << ']';
                return out.str();
            } else if constexpr (std::is_same_v<T, FeListValue>) {
                std::ostringstream out;
                out << '[';
                for (std::size_t index = 0; index < inner.values.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    out << fe_value_to_plan_string(inner.values[index]);
                }
                out << ']';
                return out.str();
            }
        },
        value.value
    );
}

std::string fe_binary_op_to_plan_string(FeBinaryOp op) {
    switch (op) {
        case FeBinaryOp::Add:
            return "+";
        case FeBinaryOp::Sub:
            return "-";
        case FeBinaryOp::Mul:
            return "*";
        case FeBinaryOp::Div:
            return "/";
        case FeBinaryOp::FloorDiv:
            return "//";
        case FeBinaryOp::Eq:
            return "==";
        case FeBinaryOp::NotEq:
            return "!=";
        case FeBinaryOp::Lt:
            return "<";
        case FeBinaryOp::Gt:
            return ">";
        case FeBinaryOp::LtEq:
            return "<=";
        case FeBinaryOp::GtEq:
            return ">=";
        case FeBinaryOp::And:
            return "&&";
        case FeBinaryOp::Or:
            return "||";
        case FeBinaryOp::Not:
            return "!";
    }
    return "?";
}

void append_value_ids(std::ostringstream& out, const std::vector<std::size_t>& values) {
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << '%' << values[index];
    }
    out << ']';
}

std::optional<Diagnostic> validate_step(const ExecutionPlan& plan, const std::set<std::size_t>& value_ids, std::size_t index) {
    const PlanStep& step = plan.steps[index];
    if (value_ids.find(step.value_id) == value_ids.end()) {
        return plan_error(
            "Execution plan '" + plan.name + "' step #" + std::to_string(index) +
            " references missing value " + std::to_string(step.value_id)
        );
    }
    if (step.op_index && *step.op_index >= plan.ops.size()) {
        return plan_error(
            "Execution plan '" + plan.name + "' step #" + std::to_string(index) +
            " references missing op " + std::to_string(*step.op_index)
        );
    }
    if ((step.kind == PlanStepKind::ExecuteOp || step.kind == PlanStepKind::DispatchDeviceOp) && !step.op_index) {
        return plan_error(
            "Execution plan '" + plan.name + "' step #" + std::to_string(index) +
            " must reference an op"
        );
    }
    return std::nullopt;
}

} // namespace

ExecutionPlan compile_local_execution_plan(const GraphFunction& graph) {
    ExecutionPlan plan = make_base_plan(graph, BackendKind::Local);
    for (const auto& value : graph.values) {
        plan.values.push_back(lower_plan_value(value, Placement::Host));
    }
    append_local_allocate_steps(plan);
    for (const auto& node : graph.nodes) {
        plan.ops.push_back(lower_plan_op(node, BackendKind::Local));
        plan.steps.push_back(PlanStep{PlanStepKind::ExecuteOp, node.output, plan.ops.size() - 1});
    }
    for (const auto output : plan.outputs) {
        plan.steps.push_back(PlanStep{PlanStepKind::MaterializeOutput, output, std::nullopt});
    }
    return plan;
}

ExecutionPlan compile_metal_execution_plan(const GraphFunction& graph) {
    ExecutionPlan plan = make_base_plan(graph, BackendKind::Metal);
    for (const auto& value : graph.values) {
        const Placement placement =
            value.type.kind == FeTypeKind::Tensor && !value.is_parameter ? Placement::Device : Placement::Host;
        plan.values.push_back(lower_plan_value(value, placement));
    }
    for (const auto& value : plan.values) {
        if (value.placement == Placement::Host) {
            plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, value.id, std::nullopt});
            if (value.is_parameter) {
                plan.steps.push_back(PlanStep{PlanStepKind::UploadToDevice, value.id, std::nullopt});
            }
        } else {
            plan.steps.push_back(PlanStep{PlanStepKind::AllocateDeviceValue, value.id, std::nullopt});
        }
    }
    for (const auto& node : graph.nodes) {
        plan.ops.push_back(lower_plan_op(node, BackendKind::Metal));
        plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, node.output, plan.ops.size() - 1});
    }
    for (const auto output : plan.outputs) {
        plan.steps.push_back(PlanStep{PlanStepKind::DownloadToHost, output, std::nullopt});
        plan.steps.push_back(PlanStep{PlanStepKind::MaterializeOutput, output, std::nullopt});
    }
    return plan;
}

ExecutionPlan compile_execution_plan(const GraphFunction& graph, BackendKind backend) {
    switch (backend) {
        case BackendKind::Metal:
            return compile_metal_execution_plan(graph);
        case BackendKind::Local:
        case BackendKind::PyTorch:
        case BackendKind::Cuda:
        case BackendKind::Rocm: {
            ExecutionPlan plan = compile_local_execution_plan(graph);
            plan.backend = backend;
            for (auto& op : plan.ops) {
                op.backend = backend;
            }
            return plan;
        }
    }
    return compile_local_execution_plan(graph);
}

ExecutionPlan optimize_execution_plan(ExecutionPlan plan, const PlanOptimizationOptions& options) {
    if (plan.backend != BackendKind::Local || options.preserve_intermediate_values || !options.enable_operator_fusion) {
        return plan;
    }

    const std::vector<std::size_t> use_counts = compute_plan_use_counts(plan);
    std::vector<PlanOp> optimized_ops;
    std::vector<PlanStep> optimized_steps;
    optimized_ops.reserve(plan.ops.size());
    optimized_steps.reserve(plan.steps.size());

    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const PlanStep& step = plan.steps[step_index];
        if (can_fuse_matmul_relu(plan, use_counts, step_index)) {
            const PlanOp& matmul_op = plan.ops[*step.op_index];
            const PlanOp& relu_op = plan.ops[*plan.steps[step_index + 1].op_index];
            optimized_ops.push_back(make_matmul_relu_op(matmul_op, relu_op));
            optimized_steps.push_back(PlanStep{
                PlanStepKind::ExecuteOp,
                relu_op.output,
                optimized_ops.size() - 1,
            });
            ++step_index;
            continue;
        }

        if ((step.kind == PlanStepKind::ExecuteOp || step.kind == PlanStepKind::DispatchDeviceOp) && step.op_index) {
            optimized_ops.push_back(plan.ops[*step.op_index]);
            optimized_steps.push_back(PlanStep{step.kind, step.value_id, optimized_ops.size() - 1});
            continue;
        }

        optimized_steps.push_back(step);
    }

    plan.ops = std::move(optimized_ops);
    plan.steps = std::move(optimized_steps);
    return plan;
}

PlanModule optimize_plan_module(PlanModule module, const PlanOptimizationOptions& options) {
    for (auto& plan : module.plans) {
        plan = optimize_execution_plan(std::move(plan), options);
    }
    return module;
}

PlanModuleResult compile_plan_module(const GraphModule& graph_module, BackendKind backend) {
    PlanModule module;
    module.backend = backend;
    std::vector<std::string> callable_functions;
    callable_functions.reserve(graph_module.functions.size());
    for (const auto& graph : graph_module.functions) {
        callable_functions.push_back(graph.name);
    }

    for (const auto& graph : graph_module.functions) {
        ExecutionPlan plan = compile_execution_plan(graph, backend);
        if (auto diagnostic = validate_execution_plan(plan)) {
            return *diagnostic;
        }
        for (std::size_t index = 0; index < plan.ops.size(); ++index) {
            CapabilityCheck check = check_plan_op_capability_with_callables(backend, plan.ops[index], callable_functions);
            if (check.status == CapabilityStatus::Unsupported) {
                return capability_error(
                    backend,
                    index,
                    check.reason.value_or("operation is not supported")
                );
            }
        }
        module.plans.push_back(std::move(plan));
    }
    for (const auto& skipped : graph_module.skipped) {
        module.skipped.push_back(PlanBuildSkipped{skipped.function_name, skipped.reason});
    }
    return module;
}

BackendCapabilitySummary backend_capability_summary(BackendKind backend) {
    BackendCapabilitySummary summary;
    summary.backend = backend;
    summary.primitive_ops = primitive_ops();
    summary.library_ops = library_ops();
    summary.constructors = constructor_ops();
    summary.supports_binary_ops = true;

    switch (backend) {
        case BackendKind::Local:
            summary.notes = {
                "Runs through the host graph executor.",
                "Training and backward paths use additional runtime support outside the graph plan.",
            };
            break;
        case BackendKind::PyTorch:
            summary.notes = {
                "Uses PyTorch codegen/runtime paths.",
                "Requires Python and torch at execution time.",
            };
            break;
        case BackendKind::Cuda:
            summary.notes = {
                "Declared backend target for future NVIDIA GPU execution.",
                "Runtime execution is not implemented in this build yet.",
                "Planning support exists so CUDA can participate in backend contracts.",
            };
            break;
        case BackendKind::Rocm:
            summary.notes = {
                "Declared backend target for future AMD GPU execution.",
                "Runtime execution is not implemented in this build yet.",
                "Planning support exists so ROCm can participate in backend contracts.",
            };
            break;
        case BackendKind::Metal:
            summary.notes = {
                "Requires macOS for native Metal execution.",
                "Tensor/device placement is explicit in the execution plan.",
                "Some scalar-only shapes may still be rejected by runtime checks.",
            };
            break;
    }

    return summary;
}

CapabilityCheck check_plan_op_capability(BackendKind backend, const PlanOp& op) {
    switch (op.kind) {
        case PlanOpKind::Constant:
            return supported();
        case PlanOpKind::Binary:
            if (backend_capability_summary(backend).supports_binary_ops) {
                return supported();
            }
            return unsupported(std::string(plan_backend_name(backend)) + " backend does not support binary operations");
        case PlanOpKind::PrimitiveCall:
            if (op.resolved_op && *op.resolved_op == OpId::MatmulRelu) {
                return backend == BackendKind::Local
                    ? supported()
                    : unsupported(std::string(plan_backend_name(backend)) + " backend does not support fused matmul_relu");
            }
            return check_named_op(backend, "primitive", op, primitive_ops(), primitive_op_ids());
        case PlanOpKind::LibraryCall:
            return check_named_op(backend, "library", op, library_ops(), library_op_ids());
        case PlanOpKind::LibraryCtor:
            return check_named_op(backend, "constructor", op, constructor_ops(), constructor_op_ids());
        case PlanOpKind::Apply:
            return supported();
    }
    return unsupported("operation is not supported");
}

std::optional<Diagnostic> validate_execution_plan_capabilities(const ExecutionPlan& plan) {
    for (std::size_t index = 0; index < plan.ops.size(); ++index) {
        CapabilityCheck check = check_plan_op_capability(plan.backend, plan.ops[index]);
        if (check.status == CapabilityStatus::Unsupported) {
            return capability_error(
                plan.backend,
                index,
                check.reason.value_or("operation is not supported")
            );
        }
    }
    return std::nullopt;
}

std::optional<Diagnostic> validate_execution_plan(const ExecutionPlan& plan) {
    std::set<std::size_t> value_ids;
    for (const auto& value : plan.values) {
        value_ids.insert(value.id);
    }
    if (value_ids.size() != plan.values.size()) {
        return plan_error("Execution plan '" + plan.name + "' has duplicate value ids");
    }
    for (std::size_t index = 0; index < plan.values.size(); ++index) {
        if (plan.values[index].id != index) {
            return plan_error(
                "Execution plan '" + plan.name + "' value id " + std::to_string(plan.values[index].id) +
                " is out of order at index " + std::to_string(index)
            );
        }
    }
    for (const auto output : plan.outputs) {
        if (value_ids.find(output) == value_ids.end()) {
            return plan_error("Execution plan '" + plan.name + "' output " + std::to_string(output) + " does not reference a value");
        }
    }
    for (const auto& named : plan.named_values) {
        if (named.first.empty()) {
            return plan_error("Execution plan '" + plan.name + "' has an empty named value");
        }
        if (value_ids.find(named.second) == value_ids.end()) {
            return plan_error(
                "Execution plan '" + plan.name + "' named value '" + named.first +
                "' references missing value " + std::to_string(named.second)
            );
        }
    }
    std::set<std::string> parameter_names;
    for (const auto& parameter : plan.parameters) {
        if (parameter.name.empty()) {
            return plan_error("Execution plan '" + plan.name + "' has an unnamed parameter");
        }
        if (parameter.role.empty()) {
            return plan_error("Execution plan '" + plan.name + "' parameter '" + parameter.name + "' has an empty role");
        }
        if (value_ids.find(parameter.owner_value) == value_ids.end()) {
            return plan_error(
                "Execution plan '" + plan.name + "' parameter '" + parameter.name +
                "' references missing owner value " + std::to_string(parameter.owner_value)
            );
        }
        if (!parameter_names.insert(parameter.name).second) {
            return plan_error("Execution plan '" + plan.name + "' has duplicate parameter '" + parameter.name + "'");
        }
    }
    for (std::size_t index = 0; index < plan.ops.size(); ++index) {
        const PlanOp& op = plan.ops[index];
        if (value_ids.find(op.output) == value_ids.end()) {
            return plan_error(
                "Execution plan '" + plan.name + "' op #" + std::to_string(index) +
                " outputs missing value " + std::to_string(op.output)
            );
        }
        for (const auto input : op.inputs) {
            if (value_ids.find(input) == value_ids.end()) {
                return plan_error(
                    "Execution plan '" + plan.name + "' op #" + std::to_string(index) +
                    " input " + std::to_string(input) + " does not reference a value"
                );
            }
        }
        if (op.kind == PlanOpKind::Binary && op.inputs.size() != 2) {
            return plan_error(
                "Execution plan '" + plan.name + "' binary op #" + std::to_string(index) +
                " must have exactly two inputs"
            );
        }
        if (op.kind == PlanOpKind::Apply && op.inputs.size() < 2) {
            return plan_error(
                "Execution plan '" + plan.name + "' apply op #" + std::to_string(index) +
                " must have a callee and an input"
            );
        }
    }
    for (std::size_t index = 0; index < plan.steps.size(); ++index) {
        if (auto diagnostic = validate_step(plan, value_ids, index)) {
            return diagnostic;
        }
    }
    return std::nullopt;
}

std::string execution_plan_summary(const PlanModule& module) {
    std::ostringstream out;
    out << "plan=backend:" << plan_backend_name(module.backend)
        << " functions:" << module.plans.size()
        << " skipped:" << module.skipped.size();
    return out.str();
}

std::string execution_plan_to_string(const PlanModule& module) {
    std::ostringstream out;
    out << execution_plan_summary(module) << '\n';
    for (const auto& plan : module.plans) {
        out << "execution_plan " << plan.name
            << " backend=" << plan_backend_name(plan.backend)
            << " values=" << plan.values.size()
            << " parameters=" << plan.parameters.size()
            << " ops=" << plan.ops.size()
            << " steps=" << plan.steps.size()
            << " outputs=" << plan.outputs.size()
            << " -> " << fe_type_to_plan_string(plan.return_type) << '\n';
        for (const auto& value : plan.values) {
            out << "  %" << value.id;
            if (!value.name.empty()) {
                out << ' ' << value.name;
            }
            out << ": " << fe_type_to_plan_string(value.type)
                << " placement=" << placement_name(value.placement);
            if (value.tensor_type) {
                out << " tensor=" << graph_tensor_type_to_string(*value.tensor_type);
            }
            if (value.is_parameter) {
                out << " param";
            }
            if (value.requires_grad) {
                out << " requires_grad";
            }
            out << '\n';
        }
        for (const auto& parameter : plan.parameters) {
            out << "  param " << parameter.name
                << " role=" << parameter.role
                << " owner=%" << parameter.owner_value
                << " tensor=" << graph_tensor_type_to_string(parameter.tensor_type);
            if (parameter.trainable) {
                out << " trainable";
            }
            out << '\n';
        }
        for (std::size_t index = 0; index < plan.ops.size(); ++index) {
            const auto& op = plan.ops[index];
            out << "  op #" << index << ' ' << plan_op_kind_name(op.kind)
                << " -> %" << op.output;
            if (!op.op.empty()) {
                out << " op=" << op.op;
            }
            if (op.op_id) {
                out << " op_id=" << *op.op_id;
            }
            if (op.kind == PlanOpKind::Binary) {
                out << " op=" << fe_binary_op_to_plan_string(op.binary_op);
            }
            if (op.kind == PlanOpKind::Constant) {
                out << " value=" << fe_value_to_plan_string(op.constant);
            }
            out << " inputs=";
            append_value_ids(out, op.inputs);
            out << '\n';
        }
        for (std::size_t index = 0; index < plan.steps.size(); ++index) {
            const auto& step = plan.steps[index];
            out << "  step #" << index << ' ' << plan_step_kind_name(step.kind)
                << " value=%" << step.value_id;
            if (step.op_index) {
                out << " op=#" << *step.op_index;
            }
            out << '\n';
        }
        out << "  outputs=";
        append_value_ids(out, plan.outputs);
        out << '\n';
    }
    for (const auto& skipped : module.skipped) {
        out << "// execution plan: " << skipped.function_name << "\n"
            << "// unavailable: " << skipped.reason << '\n';
    }
    return out.str();
}
