#pragma once

#include "graph_executor.h"

#include <string>
#include <variant>

GraphExecutionResultVariant execute_metal_plan_module(
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
);

std::variant<std::string, Diagnostic> probe_native_metal_device();
