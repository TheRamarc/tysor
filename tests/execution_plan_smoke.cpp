#include "graph_ir.h"
#include "frontend_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

GraphNode make_plan_op(GraphNodeKind kind, std::string name) {
    return GraphNode{
        kind,
        0,
        std::move(name),
        std::nullopt,
        FeBinaryOp::Add,
        FeValue::none(),
        {},
        BackendKind::Host,
    };
}

std::variant<Program, Diagnostic> parseProgram(const std::string& source) {
    TokenizeResult tokenized = tokenizeWithDiagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        return *diagnostic;
    }
    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parseProgram();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }
    return std::get<Program>(std::move(parsed));
}

std::variant<GraphModule, Diagnostic> graph_module(const std::string& source) {
    auto parsed = parseProgram(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }

    SemanticAnalyzer analyzer;
    SemanticResult semantic_result = analyzer.analyzeWithInfo(std::get<Program>(parsed));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&semantic_result)) {
        return *diagnostic;
    }

    FrontendLowerer lowerer(std::get<Program>(parsed), std::get<SemanticInfo>(semantic_result));
    FrontendResult frontend_result = lowerer.lower();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        return *diagnostic;
    }
    return buildGraphModule(std::get<LoweredModule>(std::move(frontend_result)));
}

bool local_matmul_relu_plan_ok() {
    auto graphs = graph_module(
        "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
        "  y = matmul(x, w)\n"
        "  return relu(y)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graphs)) {
        std::cerr << "local-plan: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }

    auto module_result = scheduleGraphModule(std::get<GraphModule>(graphs), BackendKind::Host);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "local-plan: plan lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(module_result);
    if (module.layers.size() != 1 || !module.functions.empty() || module.backend != BackendKind::Host || !module.scheduled) {
        std::cerr << "local-plan: unexpected module shape\n";
        return false;
    }
    const Graph& plan = module.layers.front();
    if (plan.values.size() != 4 || plan.nodes.size() != 2 || plan.outputs.size() != 1) {
        std::cerr << "local-plan: unexpected plan shape\n" << graphPlanToString(module) << '\n';
        return false;
    }
    if (plan.nodes[0].kind != GraphNodeKind::PrimitiveCall || plan.nodes[0].op != "matmul" ||
        plan.nodes[1].kind != GraphNodeKind::PrimitiveCall || plan.nodes[1].op != "relu") {
        std::cerr << "local-plan: expected matmul and relu primitive ops\n";
        return false;
    }
    if (plan.nodes[0].resolvedOp != OpId::Matmul || plan.nodes[1].resolvedOp != OpId::Relu) {
        std::cerr << "local-plan: expected resolved primitive op ids\n";
        return false;
    }
    if (plan.steps.size() != plan.values.size() + plan.nodes.size() + plan.outputs.size()) {
        std::cerr << "local-plan: unexpected step count\n";
        return false;
    }
    if (plan.steps[0].kind != GraphStepKind::AllocateHostValue ||
        plan.steps[plan.values.size()].kind != GraphStepKind::ExecuteOp ||
        plan.steps.back().kind != GraphStepKind::MaterializeOutput) {
        std::cerr << "local-plan: unexpected step order\n";
        return false;
    }
    return true;
}

bool output_only_optimizer_fuses_matmul_relu() {
    auto graphs = graph_module(
        "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
        "  y = matmul(x, w)\n"
        "  return relu(y)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graphs)) {
        std::cerr << "optimizer-fusion: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }

    auto module_result = scheduleGraphModule(std::get<GraphModule>(graphs), BackendKind::Host);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "optimizer-fusion: plan lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }

    const Graph& plan = std::get<GraphModule>(module_result).layers.front();
    GraphOptimizationOptions options;
    options.preserveIntermediateValues = false;
    Graph optimized = optimizeGraph(plan, options);
    if (auto diagnostic = validateGraph(optimized)) {
        std::cerr << "optimizer-fusion: optimized plan failed validation: " << diagnostic->toString() << '\n';
        return false;
    }
    if (optimized.nodes.size() != 1 || optimized.nodes.front().kind != GraphNodeKind::PrimitiveCall ||
        optimized.nodes.front().op != "matmul_relu" ||
        optimized.nodes.front().resolvedOp != OpId::MatmulRelu ||
        optimized.nodes.front().output != plan.outputs.front() ||
        optimized.nodes.front().inputs != std::vector<std::size_t>{0, 1}) {
        std::cerr << "optimizer-fusion: expected one fused matmul_relu op\n";
        return false;
    }

    if (checkGraphOpCapability(BackendKind::Host, optimized.nodes.front()).status != CapabilityStatus::Supported ||
        checkGraphOpCapability(BackendKind::Metal, optimized.nodes.front()).status != CapabilityStatus::Unsupported) {
        std::cerr << "optimizer-fusion: unexpected fused op capability result\n";
        return false;
    }

    GraphOptimizationOptions debug_options;
    Graph debug_plan = optimizeGraph(plan, debug_options);
    if (debug_plan.nodes.size() != 2) {
        std::cerr << "optimizer-fusion: default optimization should preserve intermediates\n";
        return false;
    }
    return true;
}

bool metal_plan_uses_device_placement() {
    auto graphs = graph_module(
        "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
        "  return relu(matmul(x, w))\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graphs)) {
        std::cerr << "metal-plan: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }

    auto module_result = scheduleGraphModule(std::get<GraphModule>(graphs), BackendKind::Metal);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "metal-plan: plan lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const Graph& plan = std::get<GraphModule>(module_result).layers.front();
    if (plan.backend != BackendKind::Metal || plan.nodes.empty() || plan.nodes.front().backend != BackendKind::Metal) {
        std::cerr << "metal-plan: backend was not preserved\n";
        return false;
    }
    bool saw_device_value = false;
    bool saw_upload = false;
    bool saw_dispatch = false;
    bool saw_download = false;
    for (const auto& value : plan.values) {
        saw_device_value = saw_device_value || value.placement == Placement::Device;
    }
    for (const auto& step : plan.steps) {
        saw_upload = saw_upload || step.kind == GraphStepKind::UploadToDevice;
        saw_dispatch = saw_dispatch || step.kind == GraphStepKind::DispatchDeviceOp;
        saw_download = saw_download || step.kind == GraphStepKind::DownloadToHost;
    }
    if (!saw_device_value || !saw_upload || !saw_dispatch || !saw_download) {
        std::cerr << "metal-plan: missing expected device scheduling\n";
        return false;
    }
    return true;
}

bool validation_rejects_bad_plan_references() {
    Graph plan;
    plan.name = "bad";
    plan.values.push_back(GraphValue{0, "x", FeType::tensor("float16", std::nullopt, std::nullopt), true, true, false, std::nullopt, Placement::Host});
    plan.outputs = {42};

    auto diagnostic = validateGraph(plan);
    if (!diagnostic || diagnostic->message.find("output 42") == std::string::npos) {
        std::cerr << "bad-plan: expected output reference diagnostic\n";
        return false;
    }
    return true;
}

bool skipped_graphs_propagate_to_plan() {
    auto graphs = graph_module(
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  if true:\n"
        "    return x\n"
        "  return x\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graphs)) {
        std::cerr << "skipped-plan: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    auto module_result = scheduleGraphModule(std::get<GraphModule>(graphs), BackendKind::Host);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "skipped-plan: plan lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(module_result);
    if (!module.functions.empty() || !module.layers.empty() || module.skipped.size() != 1) {
        std::cerr << "skipped-plan: expected skipped function to propagate\n";
        return false;
    }
    return true;
}

bool module_local_function_calls_survive_capability_validation() {
    auto graphs = graph_module(
        "fn helper(x: tensor[float16]): tensor[float16]:\n"
        "  return x -> relu()\n"
        "\n"
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  return helper(x)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graphs)) {
        std::cerr << "function-call-plan: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }

    auto module_result = scheduleGraphModule(std::get<GraphModule>(graphs), BackendKind::Host);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "function-call-plan: plan lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(module_result);
    if (module.functions.size() + module.layers.size() != 2) {
        std::cerr << "function-call-plan: expected helper and model plans\n";
        return false;
    }

    return true;
}

bool capability_summary_exposes_supported_backend_ops() {
    BackendCapabilitySummary metal = backendCapabilitySummary(BackendKind::Metal);
    BackendCapabilitySummary cuda = backendCapabilitySummary(BackendKind::Cuda);
    BackendCapabilitySummary rocm = backendCapabilitySummary(BackendKind::Rocm);

    const auto contains = [](const std::vector<std::string>& values, const std::string& target) {
        return std::find(values.begin(), values.end(), target) != values.end();
    };

    if (metal.backend != BackendKind::Metal || !metal.supportsBinaryOps ||
        !contains(metal.primitiveOps, "matmul") ||
        !contains(metal.libraryOps, "rms_norm") ||
        !contains(metal.constructors, "linear")) {
        std::cerr << "capability-summary: missing expected Metal supported ops\n";
        return false;
    }

    const auto mentions_not_implemented = [](const std::vector<std::string>& notes) {
        return std::any_of(notes.begin(), notes.end(), [](const std::string& note) {
            return note.find("not implemented") != std::string::npos;
        });
    };
    if (!mentions_not_implemented(cuda.notes) || !mentions_not_implemented(rocm.notes)) {
        std::cerr << "capability-summary: expected CUDA/ROCm runtime notes\n";
        return false;
    }

    return true;
}

bool capability_check_rejects_unknown_named_ops() {
    CapabilityCheck check = checkGraphOpCapability(
        BackendKind::PyTorch,
        make_plan_op(GraphNodeKind::LibraryCall, "not_a_real_op")
    );

    if (check.status != CapabilityStatus::Unsupported ||
        !check.reason ||
        check.reason->find("not_a_real_op") == std::string::npos) {
        std::cerr << "capability-check: expected unknown op to be rejected\n";
        return false;
    }

    return true;
}

bool capability_validation_returns_structured_backend_diagnostic() {
    Graph plan;
    plan.backend = BackendKind::Metal;
    plan.nodes.push_back(make_plan_op(GraphNodeKind::LibraryCall, "not_a_real_op"));

    auto diagnostic = validateGraphCapabilities(plan);
    if (!diagnostic) {
        std::cerr << "capability-validation: expected unknown op diagnostic\n";
        return false;
    }
    if (diagnostic->code != DiagnosticCode::BackendError ||
        diagnostic->severity != DiagnosticSeverity::Error ||
        diagnostic->message.find("not_a_real_op") == std::string::npos ||
        !diagnostic->help) {
        std::cerr << "capability-validation: unexpected diagnostic shape\n";
        return false;
    }

    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks{
        local_matmul_relu_plan_ok(),
        output_only_optimizer_fuses_matmul_relu(),
        metal_plan_uses_device_placement(),
        validation_rejects_bad_plan_references(),
        skipped_graphs_propagate_to_plan(),
        module_local_function_calls_survive_capability_validation(),
        capability_summary_exposes_supported_backend_ops(),
        capability_check_rejects_unknown_named_ops(),
        capability_validation_returns_structured_backend_diagnostic(),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
