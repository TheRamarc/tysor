#pragma once

#include "diagnostic.h"
#include "execution_plan.h"
#include "cpu_executor.h"
#include "runtime_tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

inline const ExecutionPlan* find_plan(const PlanModule& module, const std::string& entry) {
    auto found = std::find_if(
        module.plans.begin(),
        module.plans.end(),
        [&](const ExecutionPlan& plan) { return plan.name == entry; }
    );
    return found == module.plans.end() ? nullptr : &*found;
}

inline const PlanParameter* find_plan_parameter(
    const ExecutionPlan& plan,
    std::size_t owner_value,
    const std::string& role
) {
    auto found = std::find_if(
        plan.parameters.begin(),
        plan.parameters.end(),
        [&](const PlanParameter& parameter) {
            return parameter.ownerValue == owner_value && parameter.role == role;
        }
    );
    return found == plan.parameters.end() ? nullptr : &*found;
}

inline const PlanParameter* find_plan_parameter_by_value(
    const ExecutionPlan& plan,
    std::size_t value_id
) {
    auto found = std::find_if(
        plan.parameters.begin(),
        plan.parameters.end(),
        [&](const PlanParameter& parameter) {
            return parameter.valueId == value_id;
        }
    );
    return found == plan.parameters.end() ? nullptr : &*found;
}

inline std::optional<std::int64_t> known_graph_dim(
    const GraphTensorType& type,
    std::size_t index
) {
    if (!type.hasKnownRank || index >= type.shape.size()) {
        return std::nullopt;
    }
    const GraphDim& dim = type.shape[index];
    if (dim.kind != GraphDimKind::Known) {
        return std::nullopt;
    }
    return dim.value;
}

inline SimpleTensor zeros_like(const SimpleTensor& tensor) {
    TensorData data;
    data.assign(tensor.data.size(), 0.0F);
    return SimpleTensor{
        tensor.shape,
        std::move(data),
        tensor.dtype
    };
}

inline SimpleTensor ones_like(const SimpleTensor& tensor) {
    TensorData data;
    data.assign(tensor.data.size(), 1.0F);
    return SimpleTensor{
        tensor.shape,
        std::move(data),
        tensor.dtype
    };
}

inline std::variant<const SimpleTensor*, Diagnostic> require_tensor(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label,
    DiagnosticCode code = DiagnosticCode::RuntimeError
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return Diagnostic::error(code, "Missing " + label);
    }
    const auto* tensor = std::get_if<SimpleTensor>(&found->second);
    if (tensor == nullptr) {
        return Diagnostic::error(code, "Expected tensor value for " + label);
    }
    return tensor;
}

inline std::variant<double, Diagnostic> require_scalar(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label,
    DiagnosticCode code = DiagnosticCode::RuntimeError
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return Diagnostic::error(code, "Missing " + label);
    }
    if (const auto* value = std::get_if<std::int64_t>(&found->second)) {
        return static_cast<double>(*value);
    }
    if (const auto* value = std::get_if<double>(&found->second)) {
        return *value;
    }
    if (const auto* value = std::get_if<bool>(&found->second)) {
        return *value ? 1.0 : 0.0;
    }
    return Diagnostic::error(code, "Expected scalar value for " + label);
}

inline std::variant<std::int64_t, Diagnostic> require_int(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label,
    DiagnosticCode code = DiagnosticCode::RuntimeError
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return Diagnostic::error(code, "Missing " + label);
    }
    if (const auto* value = std::get_if<std::int64_t>(&found->second)) {
        return *value;
    }
    return Diagnostic::error(code, "Expected integer value for " + label);
}

inline std::variant<bool, Diagnostic> require_bool(
    const GraphExecutionResult& execution,
    std::size_t value_id,
    const std::string& label,
    DiagnosticCode code = DiagnosticCode::RuntimeError
) {
    auto found = execution.values.find(value_id);
    if (found == execution.values.end()) {
        return Diagnostic::error(code, "Missing " + label);
    }
    if (const auto* value = std::get_if<bool>(&found->second)) {
        return *value;
    }
    return Diagnostic::error(code, "Expected bool value for " + label);
}

inline std::variant<std::vector<std::int64_t>, Diagnostic> resolve_tensor_argument_shape(
    const PlanValue& value,
    const GraphExecutorOptions& options,
    DiagnosticCode code = DiagnosticCode::RuntimeError
) {
    auto cli_shape = options.tensor_shapes.find(value.name);
    if (cli_shape != options.tensor_shapes.end()) {
        const std::vector<std::int64_t>& provided = cli_shape->second;
        if (value.tensorType && value.tensorType->hasKnownRank) {
            const auto& declared = value.tensorType->shape;
            if (provided.size() != declared.size()) {
                return Diagnostic::error(
                    code,
                    "CLI --shape for '" + value.name + "' has rank " +
                    std::to_string(provided.size()) + ", expected rank " +
                    std::to_string(declared.size())
                );
            }
            for (std::size_t i = 0; i < declared.size(); ++i) {
                if (declared[i].kind == GraphDimKind::Known &&
                    declared[i].value != provided[i]) {
                    return Diagnostic::error(
                        code,
                        "CLI --shape for '" + value.name + "' dim " + std::to_string(i) +
                        " is " + std::to_string(provided[i]) + ", expected " +
                        std::to_string(declared[i].value)
                    );
                }
            }
        }
        return provided;
    }

    if (value.tensorType && value.tensorType->hasKnownRank && !value.tensorType->shape.empty()) {
        std::vector<std::int64_t> resolved;
        resolved.reserve(value.tensorType->shape.size());
        for (const GraphDim& dim : value.tensorType->shape) {
            if (dim.kind == GraphDimKind::Known) {
                resolved.push_back(dim.value);
                continue;
            }
            if (dim.kind == GraphDimKind::Symbolic) {
                return Diagnostic::error(
                    code,
                    "Missing --shape for tensor parameter '" + value.name +
                    "' (symbolic dimension '" + dim.symbol + "' requires --shape)"
                );
            }
            return Diagnostic::error(
                code,
                "Missing --shape for tensor parameter '" + value.name + "'"
            );
        }
        return resolved;
    }

    return Diagnostic::error(code, "Missing --shape for tensor parameter '" + value.name + "'");
}

inline std::variant<SimpleTensor, Diagnostic> build_tensor_from_plan_constant(
    const PlanOp& op,
    const PlanValue& output,
    RuntimeTensorWorkspace* workspace = nullptr,
    DiagnosticCode code = DiagnosticCode::RuntimeError
) {
    const auto* list = std::get_if<FeListValue>(&op.constant.value);
    if (list == nullptr) {
        return Diagnostic::error(code, "tensor() op is missing constant element list");
    }
    if (!output.tensorType || !output.tensorType->hasKnownRank || output.tensorType->shape.empty()) {
        return Diagnostic::error(code, "tensor() op is missing static shape metadata");
    }
    std::vector<std::int64_t> shape;
    shape.reserve(output.tensorType->shape.size());
    for (const GraphDim& dim : output.tensorType->shape) {
        if (dim.kind != GraphDimKind::Known || dim.value <= 0) {
            return Diagnostic::error(code, "tensor() op requires positive known dimensions");
        }
        shape.push_back(dim.value);
    }
    if (list->values.size() != numElements(shape)) {
        return Diagnostic::error(code, "tensor() element count does not match shape");
    }
    TensorData data = workspace != nullptr ? workspace->acquire(list->values.size()) : TensorData{};
    data.reserve(list->values.size());
    for (const FeValue& item : list->values) {
        if (const auto* f = std::get_if<double>(&item.value)) {
            data.push_back(static_cast<float>(*f));
        } else if (const auto* i = std::get_if<std::int64_t>(&item.value)) {
            data.push_back(static_cast<float>(*i));
        } else {
            return Diagnostic::error(code, "tensor() elements must be numeric");
        }
    }
    std::string dtype = output.tensorType->dtype.value_or(output.type.tensorDtype.value_or("float32"));
    return SimpleTensor{std::move(shape), std::move(data), std::move(dtype)};
}
