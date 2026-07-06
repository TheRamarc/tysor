#pragma once

#include "graph_executor.h"

#include <string>
#include <variant>

/**
 * @brief Executes an execution plan module using the Metal backend.
 *
 * This function locates the specified entry point within the plan module, builds a
 * Metal context, and executes the lowered plan operations directly on a Metal device.
 * It manages allocating buffers, uploading parameters, executing GPU dispatches, and
 * downloading outputs back to the host.
 *
 * @param module The compiled execution plan module.
 * @param entry The name of the entry function to execute.
 * @param options Execution options, such as tensor shapes.
 * @return GraphExecutionResultVariant The runtime values representing the execution results or a Diagnostic on error.
 */
GraphExecutionResultVariant execute_metal_plan_module(
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
);

/**
 * @brief Probes for a native Metal device on the current system.
 *
 * Checks if a Metal-compatible GPU is available. It returns a string containing
 * device report details on success, or a Diagnostic error if probing fails or
 * the platform is unsupported.
 *
 * @return std::variant<std::string, Diagnostic> A device report string, or an error.
 */
std::variant<std::string, Diagnostic> probe_native_metal_device();
