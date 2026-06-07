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
};

using GraphRuntimeValue = std::variant<std::int64_t, double, bool, SimpleTensor>;

struct GraphExecutionResult {
    std::map<std::size_t, GraphRuntimeValue> values;
    std::map<std::size_t, GraphRuntimeValue> outputs;
};

using GraphExecutionResultVariant = std::variant<GraphExecutionResult, Diagnostic>;

GraphExecutionResultVariant execute_execution_plan(const ExecutionPlan& plan, const GraphExecutorOptions& options);
GraphExecutionResultVariant execute_plan_module(const PlanModule& module, const std::string& entry, const GraphExecutorOptions& options);
void print_graph_runtime_value(const GraphRuntimeValue& value);
