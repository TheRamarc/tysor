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

// Placement is the planner's contract with runtime backends: host values are
// materialized locally, device values require upload/dispatch/download steps.
enum class Placement {
    Host,
    Device,
};

// Represents a runtime value to be mapped to a physical buffer by the executor.
// Tracks placement (host or device) and properties like whether it needs gradients.
struct PlanValue {
    // Why it exists: To provide a fast, dense handle for referencing this value.
    // What it tracks: The unique numeric ID assigned to this graph value.
    // What mutates it: Sequentially assigned by the GraphBuilder.
    std::size_t id = 0;
    // Why it exists: To declare a new symbol in the local scope.
    // What it tracks: The variable's text identifier.
    // What mutates it: Set during lowering.
    std::string name;
    // Why it exists: To define the variable's memory layout and constraints.
    // What it tracks: The statically resolved type of the variable.
    // What mutates it: Set during semantic checks.
    FeType type;
    // Why it exists: To differentiate computed intermediates from persistent state.
    // What it tracks: True if this value corresponds to a persistent model parameter.
    // What mutates it: Set to true when the value is generated from a layer constructor or explicit param.
    bool isParameter = false;
    // Why it exists: To instruct the autodiff engine whether this value needs a gradient.
    // What it tracks: True if a parameter is trainable or if an intermediate depends on a trainable param.
    // What mutates it: Initially set by parameter flags; propagated forward during graph analysis.
    bool requiresGrad = false;
    // Why it exists: To manage device memory limits and orchestrate data transfers.
    // What it tracks: Whether the value physically resides on the CPU (Host) or an accelerator (Device).
    // What mutates it: Resolved by the planner based on backend capabilities.
    Placement placement = Placement::Host;
    // Why it exists: To identify top-level configuration values passed to the model.
    // What it tracks: True if the value comes directly from the model's signature.
    // What mutates it: Marked true for inputs derived from model function arguments.
    bool isModelParameter = false;
    // Why it exists: To enforce shape and type constraints on the parameter.
    // What it tracks: The data type and dimensionality of the parameter buffer.
    // What mutates it: Established during parameter creation.
    std::optional<GraphTensorType> tensorType = std::nullopt;
};

struct PlanParameter {
    // Why it exists: To declare a new symbol in the local scope.
    // What it tracks: The variable's text identifier.
    // What mutates it: Set during lowering.
    std::string name;
    // Why it exists: To distinguish weights, biases, or auxiliary buffers.
    // What it tracks: The semantic purpose of the parameter (e.g., "weight", "bias").
    // What mutates it: Assigned by the frontend node or layer constructor.
    std::string role;
    // Why it exists: To link a parameter to the structure that owns it.
    // What it tracks: The value ID of the layer or tuple that logically owns this parameter.
    // What mutates it: Assigned during graph compilation when layers are flattened.
    std::size_t ownerValue = 0;
    // Why it exists: To connect the parameter metadata to its concrete dataflow node.
    // What it tracks: The graph value ID that produces or represents this parameter.
    // What mutates it: Assigned when the parameter node is added to the graph.
    std::size_t valueId = 0;
    // Why it exists: To enforce shape and type constraints on the parameter.
    // What it tracks: The data type and dimensionality of the parameter buffer.
    // What mutates it: Established during parameter creation.
    GraphTensorType tensorType;
    // Why it exists: To determine if gradients should be computed and applied to it.
    // What it tracks: Whether this parameter is subject to optimization updates.
    // What mutates it: Set based on frontend configuration; immutable.
    bool trainable = true;
};

enum class PlanOpKind {
    Constant,
    Binary,
    PrimitiveCall,
    LibraryCall,
    LibraryCtor,
    Apply,
};

// PlanOp is GraphNode plus backend-specific scheduling metadata. It still keeps
// source operation text for diagnostics, but hot paths should prefer resolvedOp.
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
          opId(std::move(op_identifier)),
          binaryOp(binary),
          constant(std::move(constant_value)),
          inputs(std::move(input_ids)),
          backend(target_backend),
          resolvedOp(resolved) {}

    // Why it exists: To drive dispatch to the appropriate executor routines.
    // What it tracks: The category of operation this node represents.
    // What mutates it: Inherited from GraphNode and transformed if fused/optimized.
    PlanOpKind kind = PlanOpKind::Constant;
    // Why it exists: To link the node to the value it produces.
    // What it tracks: The value ID generated by this node's evaluation.
    // What mutates it: Assigned during node addition to the graph.
    std::size_t output = 0;
    // Why it exists: For readable serialization and diagnostic messages.
    // What it tracks: The string name of the operation (e.g., "Add", "Relu").
    // What mutates it: Extracted from frontend IR node.
    std::string op;
    // Why it exists: To bypass string matching for known builtin ops.
    // What it tracks: A stable identifier for resolved primitive operations.
    // What mutates it: Resolved during graph building if a match is found.
    std::optional<std::string> opId;
    // Why it exists: To define what arithmetic or logic operation is performed.
    // What it tracks: The specific frontend binary operator (e.g., Add, Mul, Eq).
    // What mutates it: Assigned when parsing/lowering a binary operation.
    FeBinaryOp binaryOp = FeBinaryOp::Add;
    // Why it exists: To hold a runtime or compile-time evaluated constant.
    // What it tracks: The exact underlying primitive or composite data value (using std::variant).
    // What mutates it: Constructed by factory methods and typically passed by value; mutable by assignment if needed.
    FeValue constant = FeValue::none();
    // Why it exists: To capture dataflow dependencies.
    // What it tracks: The value IDs consumed as arguments by this node.
    // What mutates it: Populated with argument IDs when the node is lowered.
    std::vector<std::size_t> inputs;
    // Why it exists: To determine which hardware executes this specific operation.
    // What it tracks: The execution environment assigned to this op (e.g., Local CPU, Metal).
    // What mutates it: Assigned by the execution planner.
    BackendKind backend = BackendKind::Local;
    // Resolved once during planning so hot runtime paths can dispatch by enum
    // instead of repeatedly resolving or comparing operation strings.
    // Why it exists: To avoid string lookups during the hot loop of execution.
    // What it tracks: The fast enum representation of the backend operation.
    // What mutates it: Looked up and set during plan compilation.
    std::optional<OpId> resolvedOp;
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
    // Why it exists: To report backend compatibility checks.
    // What it tracks: Whether the backend can run the given op.
    // What mutates it: Set by capability checking functions.
    CapabilityStatus status = CapabilityStatus::Supported;
    // Why it exists: To inform the user why graph generation was skipped.
    // What it tracks: The error message or unsupported feature reason.
    // What mutates it: Extracted from diagnostics.
    std::optional<std::string> reason;
};

struct BackendCapabilitySummary {
    // Why it exists: To determine which hardware executes this specific operation.
    // What it tracks: The execution environment assigned to this op (e.g., Local CPU, Metal).
    // What mutates it: Assigned by the execution planner.
    BackendKind backend = BackendKind::Local;
    // Why it exists: To enumerate the fundamental math ops the backend handles natively.
    // What it tracks: Allowed string identifiers for primitive ops.
    // What mutates it: Hardcoded in backend capability reporters.
    std::vector<std::string> primitiveOps;
    // Why it exists: To list complex fused ops (like matmul_relu) the backend provides.
    // What it tracks: Allowed string identifiers for library calls.
    // What mutates it: Hardcoded in backend capability reporters.
    std::vector<std::string> libraryOps;
    // Why it exists: To list parameter-generating nodes the backend supports.
    // What it tracks: Allowed identifiers for constructors.
    // What mutates it: Hardcoded in backend capability reporters.
    std::vector<std::string> constructors;
    // Why it exists: To indicate if standard binary math is broadly supported.
    // What it tracks: True if generic FeBinaryOps can be dispatched to this backend.
    // What mutates it: Set by backend capability summaries.
    bool supportsBinaryOps = true;
    // Why it exists: To provide developer details about backend limitations.
    // What it tracks: Free-form text noting edge cases or missing features.
    // What mutates it: Hardcoded in backend capabilities.
    std::vector<std::string> notes;
};

struct PlanStep {
    // Why it exists: To specify the action taken by the executor engine at this step.
    // What it tracks: The phase of execution (allocate, compute, move data, etc.).
    // What mutates it: Set during plan step generation.
    PlanStepKind kind = PlanStepKind::AllocateHostValue;
    // Why it exists: To connect the parameter metadata to its concrete dataflow node.
    // What it tracks: The graph value ID that produces or represents this parameter.
    // What mutates it: Assigned when the parameter node is added to the graph.
    std::size_t valueId = 0;
    // Why it exists: To bind a compute step to its underlying plan operation.
    // What it tracks: The index into the ops vector for an ExecuteOp or DispatchDeviceOp step.
    // What mutates it: Set when generating execution steps.
    std::optional<std::size_t> opIndex;
};

// ExecutionPlan is deliberately explicit: values define storage, ops define
// computation, and steps define execution order/data movement. Value ids are
// validated to match their vector index. This struct drives backend execution.
struct ExecutionPlan {
    // Why it exists: To determine which hardware executes this specific operation.
    // What it tracks: The execution environment assigned to this op (e.g., Local CPU, Metal).
    // What mutates it: Assigned by the execution planner.
    BackendKind backend = BackendKind::Local;
    // Why it exists: To identify the function or layer.
    // What it tracks: The name of the defined callable.
    // What mutates it: Set during parsing/lowering.
    std::string name;
    // Why it exists: To enforce type safety on function/layer outputs.
    // What it tracks: The declared or inferred return type of the callable.
    // What mutates it: Resolved during semantic analysis.
    FeType returnType;
    // Why it exists: To outline the storage requirements for execution.
    // What it tracks: Every buffer that needs to exist during the run.
    // What mutates it: Translated from GraphValues and optimized.
    std::vector<PlanValue> values;
    // Why it exists: To outline the static memory requirements.
    // What it tracks: Trainable and fixed weights bound to value IDs.
    // What mutates it: Translated from GraphParameters.
    std::vector<PlanParameter> parameters;
    // Why it exists: To contain the raw computation dependency graph.
    // What it tracks: The logical mathematical operations, stripped of memory semantics.
    // What mutates it: Lowered from GraphNodes and updated by fusion passes.
    std::vector<PlanOp> ops;
    // Why it exists: To provide a linear, instruction-like sequence for the executor.
    // What it tracks: Ordered memory allocations, transfers, and compute dispatches.
    // What mutates it: Generated by the planner examining op placements and dependencies.
    std::vector<PlanStep> steps;
    // Why it exists: To identify which values are yielded when the function returns.
    // What it tracks: The value IDs returned by the function.
    // What mutates it: Populated upon encountering return statements.
    std::vector<std::size_t> outputs;
    // Why it exists: To bridge source-level symbols with graph IDs for debugging and objectives.
    // What it tracks: A mapping from original variable names to their graph value IDs.
    // What mutates it: Inserted into whenever a named variable is declared or assigned.
    std::map<std::string, std::size_t> namedValues;
};

struct PlanBuildSkipped {
    // Why it exists: To identify which function failed to compile.
    // What it tracks: The original name of the function skipped.
    // What mutates it: Set upon encountering a build failure.
    std::string functionName;
    // Why it exists: To inform the user why graph generation was skipped.
    // What it tracks: The error message or unsupported feature reason.
    // What mutates it: Extracted from diagnostics.
    std::string reason;
};

struct PlanModule {
    // Why it exists: To determine which hardware executes this specific operation.
    // What it tracks: The execution environment assigned to this op (e.g., Local CPU, Metal).
    // What mutates it: Assigned by the execution planner.
    BackendKind backend = BackendKind::Local;
    // Why it exists: To hold multiple executable units.
    // What it tracks: Built plans ready for runtime consumption.
    // What mutates it: Pushed into during module compilation.
    std::vector<ExecutionPlan> plans;
    // Why it exists: To keep a record of what could not be compiled without halting entirely.
    // What it tracks: Information about functions skipped during graph building.
    // What mutates it: Pushed into when function lowering returns a diagnostic.
    std::vector<PlanBuildSkipped> skipped;
};

struct PlanOptimizationOptions {
    // Preserve intermediates by default because plan dumps, debugging, backward,
    // and training paths may need every graph value to be materialized.
    // Why it exists: To toggle memory optimization features.
    // What it tracks: True if intermediate values shouldn't be destroyed eagerly (needed for backprop/debug).
    // What mutates it: Specified as input to the optimizer.
    bool preserveIntermediateValues = true;
    // Why it exists: To allow the compiler to merge ops for better performance.
    // What it tracks: True if fusion passes should run.
    // What mutates it: Specified as input to the optimizer.
    bool enableOperatorFusion = true;
};

using ExecutionPlanResult = std::variant<ExecutionPlan, Diagnostic>;
using PlanModuleResult = std::variant<PlanModule, Diagnostic>;

ExecutionPlan compileExecutionPlan(const GraphFunction& graph, BackendKind backend);
ExecutionPlan compileExecutionPlan(const GraphLayer& graph, BackendKind backend);
ExecutionPlan compileLocalExecutionPlan(const GraphFunction& graph);
ExecutionPlan compileLocalExecutionPlan(const GraphLayer& graph);
ExecutionPlan compileMetalExecutionPlan(const GraphFunction& graph);
ExecutionPlan compileMetalExecutionPlan(const GraphLayer& graph);
PlanModuleResult compilePlanModule(const GraphModule& graph_module, BackendKind backend);
ExecutionPlan optimizeExecutionPlan(ExecutionPlan plan, const PlanOptimizationOptions& options);
PlanModule optimizePlanModule(PlanModule module, const PlanOptimizationOptions& options);
std::optional<Diagnostic> validateExecutionPlan(const ExecutionPlan& plan);
BackendCapabilitySummary backendCapabilitySummary(BackendKind backend);
CapabilityCheck checkPlanOpCapability(BackendKind backend, const PlanOp& op);
std::optional<Diagnostic> validateExecutionPlanCapabilities(const ExecutionPlan& plan);
std::string executionPlanSummary(const PlanModule& module);
std::string executionPlanToString(const PlanModule& module);
