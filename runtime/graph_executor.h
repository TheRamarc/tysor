#pragma once

#include "execution_plan.h"
#include "runtime_tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

struct GraphExecutorOptions {
    std::map<std::string, std::vector<std::int64_t>> tensor_shapes;
    // Keep true for debugging/backward passes that need every intermediate.
    // Benchmarks and inference-only paths can set this false so the executor
    // can recycle temporary tensor buffers as soon as their final use is done.
    bool collect_intermediate_values = true;
    RuntimeTensorWorkspace* tensor_workspace = nullptr;
};

using GraphRuntimeValue = std::variant<std::int64_t, double, bool, SimpleTensor>;

// Stores the evaluated values of intermediate nodes and final outputs for a graph execution.
struct GraphExecutionResult {
    std::map<std::size_t, GraphRuntimeValue> values;
    std::map<std::size_t, GraphRuntimeValue> outputs;
};

using GraphExecutionResultVariant = std::variant<GraphExecutionResult, Diagnostic>;

// Core execution routine for a given plan.
GraphExecutionResultVariant execute_execution_plan(const ExecutionPlan& plan, const GraphExecutorOptions& options);

// Looks up and executes a named plan within a compiled module.
GraphExecutionResultVariant execute_plan_module(const PlanModule& module, const std::string& entry, const GraphExecutorOptions& options);

void print_graph_runtime_value(const GraphRuntimeValue& value);
