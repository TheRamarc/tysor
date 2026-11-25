#pragma once

#include "cpu_executor.h"
#include "graph_ir.h"
#include "frontend_ir.h"

#include <string>

// Executes a forward pass for training, handling parameter synthesis and state
// capture needed for backward passes. Similar to cpu_executor but with
// extensions for training-specific semantics.
std::optional<Diagnostic>
run_train_plan_module(const LoweredModule &lowered, const GraphModule &module,
                      const std::string &entry,
                      const GraphExecutorOptions &options);
