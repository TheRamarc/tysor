#include "train_executor.h"

#include "cpu_executor.h"
#include "runtime_utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <variant>

namespace {

Diagnostic train_error(std::string message) {
    return Diagnostic::error(DiagnosticCode::RuntimeTrainError, std::move(message));
}

std::optional<std::string> find_skipped_plan_reason(const PlanModule& module, const std::string& entry) {
    for (const auto& skipped : module.skipped) {
        if (skipped.functionName == entry) {
            return skipped.reason;
        }
    }
    return std::nullopt;
}

const FeExecutionRun* find_train_run(const LoweredModule& lowered, const std::string& entry) {
    if (!lowered.executionPlan) {
        return nullptr;
    }
    for (const auto& run : lowered.executionPlan->runs) {
        if (run.modelName == entry) {
            return &run;
        }
    }
    return nullptr;
}

std::variant<std::string, Diagnostic> train_optimizer(const FeExecutionRun& run) {
    if (!run.optimizer) {
        return std::string("sgd");
    }
    if (const auto* value = std::get_if<std::string>(&run.optimizer->value)) {
        return *value;
    }
    return train_error("Train optimizer must be a string");
}

double train_learning_rate(const FeExecutionRun& run) {
    if (!run.learningRate) {
        return 0.01;
    }
    if (const auto* value = std::get_if<double>(&run.learningRate->value)) {
        return *value;
    }
    if (const auto* value = std::get_if<std::int64_t>(&run.learningRate->value)) {
        return static_cast<double>(*value);
    }
    return 0.01;
}

std::int64_t train_iterations(const FeExecutionRun& run) {
    if (!run.iteration) {
        return 1;
    }
    if (const auto* value = std::get_if<std::int64_t>(&run.iteration->value)) {
        return *value;
    }
    return 1;
}

std::variant<std::size_t, Diagnostic> resolve_objective_id(const FeExecutionRun& run, const ExecutionPlan& plan) {
    if (run.objectiveSymbol) {
        auto found = plan.namedValues.find(*run.objectiveSymbol);
        if (found == plan.namedValues.end()) {
            return train_error("Could not resolve objective '" + *run.objectiveSymbol + "' in execution plan");
        }
        return found->second;
    }
    if (plan.outputs.empty()) {
        return train_error("Graph does not have a return value");
    }
    return plan.outputs.front();
}

float tensor_sum(const SimpleTensor& tensor) {
    return std::accumulate(tensor.data.begin(), tensor.data.end(), 0.0F);
}

std::optional<Diagnostic> apply_sgd(
    std::map<std::string, SimpleTensor>& parameters,
    const std::map<std::string, SimpleTensor>& gradients,
    double learning_rate
) {
    for (const auto& [name, grad] : gradients) {
        auto parameter = parameters.find(name);
        if (parameter == parameters.end()) {
            continue;
        }
        if (parameter->second.shape != grad.shape || parameter->second.data.size() != grad.data.size()) {
            return train_error("Parameter gradient shape mismatch for '" + name + "'");
        }
        for (std::size_t index = 0; index < parameter->second.data.size(); ++index) {
            parameter->second.data[index] -= static_cast<float>(learning_rate) * grad.data[index];
        }
    }
    return std::nullopt;
}

} // namespace

std::optional<Diagnostic> run_train_plan_module(
    const LoweredModule& lowered,
    const PlanModule& module,
    const std::string& entry,
    const GraphExecutorOptions& options
) {
    if (module.backend != BackendKind::Host) {
        return train_error("C++ train currently supports the local backend only");
    }
    const ExecutionPlan* plan = find_plan(module, entry);
    if (plan == nullptr) {
        if (auto skipped_reason = find_skipped_plan_reason(module, entry)) {
            return train_error("Entry function '" + entry + "' was skipped during graph planning: " + *skipped_reason);
        }
        return train_error("Entry function '" + entry + "' not found in execution plan module");
    }
    const FeExecutionRun* run = find_train_run(lowered, entry);
    if (run == nullptr) {
        return train_error("Train executor requires a resolved 'config model' execution run");
    }
    auto optimizer = train_optimizer(*run);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&optimizer)) {
        return *diagnostic;
    }
    if (std::get<std::string>(optimizer) != "sgd") {
        return train_error("Train executor currently supports only optimizer='sgd'");
    }
    auto objective_id = resolve_objective_id(*run, *plan);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&objective_id)) {
        return *diagnostic;
    }

    std::map<std::string, SimpleTensor> parameters;
    const double learning_rate = train_learning_rate(*run);
    const std::int64_t iterations = train_iterations(*run);

    GraphExecutorOptions step_options = options;
    step_options.collect_intermediate_values = true;
    step_options.persistent_parameters = &parameters;

    std::cout << "\n--- Training Output ---\n";
    std::cout << "backend=local\n";
    for (std::int64_t step = 0; step < iterations; ++step) {
        auto execution = execute_execution_plan(*plan, step_options);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
            return *diagnostic;
        }
        const GraphExecutionResult& result = std::get<GraphExecutionResult>(execution);
        auto objective = require_tensor(result, std::get<std::size_t>(objective_id), "objective runtime value", DiagnosticCode::RuntimeTrainError);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&objective)) {
            return *diagnostic;
        }
        auto backward = compute_parameter_gradients(*plan, result, std::get<std::size_t>(objective_id), &parameters);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&backward)) {
            return *diagnostic;
        }
        const float loss = tensor_sum(**std::get_if<const SimpleTensor*>(&objective));
        if (auto diagnostic = apply_sgd(parameters, std::get<BackwardResult>(backward).param_grads, learning_rate)) {
            return *diagnostic;
        }
        std::cout << "step=" << (step + 1)
                  << " loss=" << std::fixed << std::setprecision(6) << loss
                  << " params=" << parameters.size() << '\n';
    }
    std::cout << "-----------------------\n";
    return std::nullopt;
}
