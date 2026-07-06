#pragma once

#include "execution_plan.h"
#include "runtime_tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

struct GraphExecutorOptions {
    // Why it exists: To let the runtime know concrete sizes for dynamic graph inputs.
    // What it tracks: A mapping of parameter/input names to their expected shapes.
    // What mutates it: Provided by the caller before execution starts.
    std::map<std::string, std::vector<std::int64_t>> tensor_shapes;
    // Keep true for debugging/backward passes that need every intermediate.
    // Benchmarks and inference-only paths can set this false so the executor
    // can recycle temporary tensor buffers as soon as their final use is done.
    // Why it exists: To control memory lifetime policy.
    // What it tracks: True if every intermediate node should be kept in the output map (needed for backprop).
    // What mutates it: Configured by caller prior to execution.
    bool collect_intermediate_values = true;
    // Why it exists: To inject memory pooling capabilities into the stateless executor.
    // What it tracks: A pointer to the workspace managing temporary buffer allocations.
    // What mutates it: Provided by the caller environment.
    RuntimeTensorWorkspace* tensor_workspace = nullptr;
};

using GraphRuntimeValue = std::variant<std::int64_t, double, bool, SimpleTensor>;

// Stores the evaluated values of intermediate nodes and final outputs for a graph execution.
struct GraphExecutionResult {
    // Why it exists: To persist computed state for analysis or backprop.
    // What it tracks: Every materialized value ID and its concrete runtime data.
    // What mutates it: Continuously updated as ops run; values may be erased if collect_intermediate_values is false.
    std::map<std::size_t, GraphRuntimeValue> values;
    // Why it exists: To yield the requested result of the function run.
    // What it tracks: A map linking requested graph output IDs to their computed runtime values.
    // What mutates it: Populated by the executor at the end of the step sequence.
    std::map<std::size_t, GraphRuntimeValue> outputs;
};

using GraphExecutionResultVariant = std::variant<GraphExecutionResult, Diagnostic>;

// Core execution routine for a given plan.
GraphExecutionResultVariant execute_execution_plan(const ExecutionPlan& plan, const GraphExecutorOptions& options);

// Looks up and executes a named plan within a compiled module.
GraphExecutionResultVariant execute_plan_module(const PlanModule& module, const std::string& entry, const GraphExecutorOptions& options);

void print_graph_runtime_value(const GraphRuntimeValue& value);
