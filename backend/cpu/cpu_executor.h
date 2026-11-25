#pragma once

#include "graph_ir.h"
#include "frontend_ir.h"
#include "runtime_tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

struct GraphExecutorOptions {
    std::map<std::string, std::vector<std::int64_t>> tensor_shapes;
    // Keep true for debugging/backward passes that need every intermediate.
    // Inference-only paths can set this false so temporary buffers are recycled.
    bool collect_intermediate_values = true;
    RuntimeTensorWorkspace* tensor_workspace = nullptr;
    // Optional persistent parameter store (keyed by parameter name) reused across
    // repeated training steps so parameter updates persist between runs.
    std::map<std::string, SimpleTensor>* persistent_parameters = nullptr;
};

using GraphRuntimeValue = std::variant<std::int64_t, double, bool, std::string, SimpleTensor>;

struct GraphExecutionResult {
    std::map<std::size_t, GraphRuntimeValue> values;
    std::map<std::size_t, GraphRuntimeValue> outputs;
};

using GraphExecutionResultVariant = std::variant<GraphExecutionResult, Diagnostic>;

// Forward execution routines for the local CPU backend.
GraphExecutionResultVariant execute_execution_plan(const Graph& plan, const GraphExecutorOptions& options);
GraphExecutionResultVariant execute_plan_module(const GraphModule& module, const std::string& entry, const GraphExecutorOptions& options);
void print_graph_runtime_value(const GraphRuntimeValue& value);

struct BackwardResult {
    std::map<std::size_t, SimpleTensor> value_grads;
    std::map<std::string, SimpleTensor> param_grads;
};

// Reverse-mode autodiff over a completed forward execution on the CPU backend.
std::variant<BackwardResult, Diagnostic> compute_parameter_gradients(
    const Graph& plan,
    const GraphExecutionResult& execution,
    std::size_t objective_id,
    const std::map<std::string, SimpleTensor>* parameters = nullptr
);

// Runs forward + backward pass for a named entry and prints parameter gradients.
std::optional<Diagnostic> run_backward_plan_module(
    const LoweredModule& lowered,
    const GraphModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
);
