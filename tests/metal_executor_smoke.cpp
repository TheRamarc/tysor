#include "metal_executor.h"

#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace {

bool unsupported_plan_rejects_before_native_context() {
    PlanModule module;
    module.backend = BackendKind::Metal;

    ExecutionPlan plan;
    plan.backend = BackendKind::Metal;
    plan.name = "model";
    plan.ops.push_back(PlanOp{
        PlanOpKind::LibraryCall,
        0,
        "unsupported_metal_op",
        std::string("unsupported_metal_op"),
        FeBinaryOp::Add,
        FeValue::none(),
        {},
        BackendKind::Metal,
    });
    module.plans.push_back(std::move(plan));

    auto result = execute_metal_plan_module(module, "model", GraphExecutorOptions{});
    const auto* diagnostic = std::get_if<Diagnostic>(&result);
    if (diagnostic == nullptr ||
        diagnostic->code != DiagnosticCode::RuntimeExecutionError ||
        diagnostic->message.find("unsupported_metal_op") == std::string::npos) {
        std::cerr << "metal-executor: expected unsupported op diagnostic before native context\n";
        return false;
    }
    return true;
}

bool metal_device_probe_reports_status() {
    auto result = probe_native_metal_device();
    if (const auto* report = std::get_if<std::string>(&result)) {
        if (report->empty()) {
            std::cerr << "metal-executor: device probe returned an empty report\n";
            return false;
        }
        return true;
    }
    const auto* diagnostic = std::get_if<Diagnostic>(&result);
    if (diagnostic == nullptr || diagnostic->message.empty() || diagnostic->code != DiagnosticCode::RuntimeExecutionError) {
        std::cerr << "metal-executor: device probe returned an invalid diagnostic\n";
        return false;
    }
    return true;
}

bool activation_plan_passes_preflight() {
    PlanModule module;
    module.backend = BackendKind::Metal;

    ExecutionPlan plan;
    plan.backend = BackendKind::Metal;
    plan.name = "model";
    plan.values.push_back(PlanValue{0, "x", FeType::tensor("float32", std::nullopt, 2), true, true, Placement::Host});
    plan.values.push_back(PlanValue{1, "act", FeType::callable(FeType::tensor("float32", std::nullopt, 2)), false, false, Placement::Host});
    plan.values.push_back(PlanValue{2, "", FeType::tensor("float32", std::nullopt, 2), false, false, Placement::Device});
    plan.outputs = {2};
    plan.ops.push_back(PlanOp{PlanOpKind::LibraryCtor, 1, "GELU", std::string("Gelu"), FeBinaryOp::Add, FeValue::none(), {}, BackendKind::Metal});
    plan.ops.push_back(PlanOp{PlanOpKind::Apply, 2, "", std::nullopt, FeBinaryOp::Add, FeValue::none(), {1, 0}, BackendKind::Metal});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, 0, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::UploadToDevice, 0, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, 1, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateDeviceValue, 2, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, 1, 0});
    plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, 2, 1});
    plan.steps.push_back(PlanStep{PlanStepKind::DownloadToHost, 2, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::MaterializeOutput, 2, std::nullopt});
    module.plans.push_back(std::move(plan));

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {2, 3};
    auto result = execute_metal_plan_module(module, "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        if (diagnostic->message.find("does not support") != std::string::npos ||
            diagnostic->message.find("GELU") != std::string::npos) {
            std::cerr << "metal-executor: activation should pass unsupported-op preflight\n";
            return false;
        }
    }
    return true;
}

bool linear_plan_passes_preflight() {
    PlanModule module;
    module.backend = BackendKind::Metal;

    ExecutionPlan plan;
    plan.backend = BackendKind::Metal;
    plan.name = "model";
    plan.values.push_back(PlanValue{0, "x", FeType::tensor("float32", std::nullopt, 2), true, true, Placement::Host});
    plan.values.push_back(PlanValue{1, "", FeType::intType(), false, false, Placement::Host});
    plan.values.push_back(PlanValue{2, "", FeType::intType(), false, false, Placement::Host});
    plan.values.push_back(PlanValue{3, "proj", FeType::callable(FeType::tensor("float32", std::nullopt, 2)), false, false, Placement::Host});
    plan.values.push_back(PlanValue{4, "", FeType::tensor("float32", std::nullopt, 2), false, false, Placement::Device});
    plan.outputs = {4};
    plan.ops.push_back(PlanOp{PlanOpKind::Constant, 1, "", std::nullopt, FeBinaryOp::Add, FeValue::intValue(3), {}, BackendKind::Metal});
    plan.ops.push_back(PlanOp{PlanOpKind::Constant, 2, "", std::nullopt, FeBinaryOp::Add, FeValue::intValue(2), {}, BackendKind::Metal});
    plan.ops.push_back(PlanOp{PlanOpKind::LibraryCtor, 3, "linear", std::string("Linear"), FeBinaryOp::Add, FeValue::none(), {1, 2}, BackendKind::Metal});
    plan.ops.push_back(PlanOp{PlanOpKind::Apply, 4, "", std::nullopt, FeBinaryOp::Add, FeValue::none(), {3, 0}, BackendKind::Metal});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, 0, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::UploadToDevice, 0, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, 1, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, 2, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateHostValue, 3, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::AllocateDeviceValue, 4, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, 1, 0});
    plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, 2, 1});
    plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, 3, 2});
    plan.steps.push_back(PlanStep{PlanStepKind::DispatchDeviceOp, 4, 3});
    plan.steps.push_back(PlanStep{PlanStepKind::DownloadToHost, 4, std::nullopt});
    plan.steps.push_back(PlanStep{PlanStepKind::MaterializeOutput, 4, std::nullopt});
    module.plans.push_back(std::move(plan));

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {2, 3};
    auto result = execute_metal_plan_module(module, "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        if (diagnostic->message.find("does not support") != std::string::npos ||
            diagnostic->message.find("linear") != std::string::npos) {
            std::cerr << "metal-executor: linear should pass unsupported-op preflight\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks{
        metal_device_probe_reports_status(),
        unsupported_plan_rejects_before_native_context(),
        linear_plan_passes_preflight(),
        activation_plan_passes_preflight(),
    };
    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
