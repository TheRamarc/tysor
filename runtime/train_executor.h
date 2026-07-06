#pragma once

#include "execution_plan.h"
#include "frontend_ir.h"
#include "graph_executor.h"

#include <string>

// Executes a forward pass for training, handling parameter synthesis and state
// capture needed for backward passes. Similar to graph_executor but with
// extensions for training-specific semantics.
std::optional<Diagnostic> run_train_plan_module(
    const LoweredModule& lowered,
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
);
