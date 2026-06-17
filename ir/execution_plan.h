#pragma once

#include "cli_args.h"
#include "diagnostic.h"
#include "graph_ir.h"
#include "ops.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

enum class Placement {
    Host,
    Device,
};

struct PlanValue {
    std::size_t id = 0;
    std::string name;
    FeType type;
    bool is_parameter = false;
    bool requires_grad = false;
    Placement placement = Placement::Host;
};

enum class PlanOpKind {
    Constant,
    Binary,
    PrimitiveCall,
    LibraryCall,
    LibraryCtor,
    Apply,
};

struct PlanOp {
    PlanOp() = default;

    PlanOp(
        PlanOpKind plan_kind,
        std::size_t output_id,
        std::string op_name,
        std::optional<std::string> op_identifier,
        FeBinaryOp binary,
        FeValue constant_value,
        std::vector<std::size_t> input_ids,
        BackendKind target_backend,
        std::optional<OpId> resolved = std::nullopt
    )
        : kind(plan_kind),
          output(output_id),
          op(std::move(op_name)),
          op_id(std::move(op_identifier)),
          binary_op(binary),
          constant(std::move(constant_value)),
          inputs(std::move(input_ids)),
          backend(target_backend),
          resolved_op(resolved) {}

    PlanOpKind kind = PlanOpKind::Constant;
    std::size_t output = 0;
    std::string op;
    std::optional<std::string> op_id;
    FeBinaryOp binary_op = FeBinaryOp::Add;
    FeValue constant = FeValue::none();
    std::vector<std::size_t> inputs;
    BackendKind backend = BackendKind::Local;
    std::optional<OpId> resolved_op;
};

enum class PlanStepKind {
    AllocateHostValue,
    AllocateDeviceValue,
    ExecuteOp,
    MaterializeOutput,
    UploadToDevice,
    DispatchDeviceOp,
    DownloadToHost,
};

enum class CapabilityStatus {
    Supported,
    Unsupported,
};

struct CapabilityCheck {
    CapabilityStatus status = CapabilityStatus::Supported;
    std::optional<std::string> reason;
};

struct BackendCapabilitySummary {
    BackendKind backend = BackendKind::Local;
    std::vector<std::string> primitive_ops;
    std::vector<std::string> library_ops;
    std::vector<std::string> constructors;
    bool supports_binary_ops = true;
    std::vector<std::string> notes;
};

struct PlanStep {
    PlanStepKind kind = PlanStepKind::AllocateHostValue;
    std::size_t value_id = 0;
    std::optional<std::size_t> op_index;
};

struct ExecutionPlan {
    BackendKind backend = BackendKind::Local;
    std::string name;
    FeType return_type;
    std::vector<PlanValue> values;
    std::vector<PlanOp> ops;
    std::vector<PlanStep> steps;
    std::vector<std::size_t> outputs;
    std::map<std::string, std::size_t> named_values;
};

struct PlanBuildSkipped {
    std::string function_name;
    std::string reason;
};

struct PlanModule {
    BackendKind backend = BackendKind::Local;
    std::vector<ExecutionPlan> plans;
    std::vector<PlanBuildSkipped> skipped;
};

using ExecutionPlanResult = std::variant<ExecutionPlan, Diagnostic>;
using PlanModuleResult = std::variant<PlanModule, Diagnostic>;

ExecutionPlan compile_execution_plan(const GraphFunction& graph, BackendKind backend);
ExecutionPlan compile_local_execution_plan(const GraphFunction& graph);
ExecutionPlan compile_metal_execution_plan(const GraphFunction& graph);
PlanModuleResult compile_plan_module(const GraphModule& graph_module, BackendKind backend);
std::optional<Diagnostic> validate_execution_plan(const ExecutionPlan& plan);
BackendCapabilitySummary backend_capability_summary(BackendKind backend);
CapabilityCheck check_plan_op_capability(BackendKind backend, const PlanOp& op);
std::optional<Diagnostic> validate_execution_plan_capabilities(const ExecutionPlan& plan);
std::string execution_plan_summary(const PlanModule& module);
std::string execution_plan_to_string(const PlanModule& module);
