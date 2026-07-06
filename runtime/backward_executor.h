#pragma once

#include "execution_plan.h"
#include "frontend_ir.h"
#include "graph_executor.h"

#include <string>

// Executes a backward pass using the cached results from a prior forward pass.
// This function computes and prints the gradients of model parameters.
std::optional<Diagnostic> run_backward_plan_module(
    const LoweredModule& lowered,
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
);
