#include "execution_plan.h"
#include "frontend_ir.h"
#include "graph_executor.h"
#include "graph_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

std::variant<PlanModule, Diagnostic> plan_module(const std::string& source) {
    TokenizeResult tokenized = tokenizeWithDiagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        return *diagnostic;
    }
    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parseProgram();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }

    Program program = std::get<Program>(std::move(parsed));
    SemanticAnalyzer analyzer;
    SemanticResult semantic_result = analyzer.analyzeWithInfo(program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&semantic_result)) {
        return *diagnostic;
    }

    FrontendLowerer lowerer(program, std::get<SemanticInfo>(semantic_result));
    FrontendResult frontend_result = lowerer.lower();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        return *diagnostic;
    }

    auto graph_result = buildGraphModule(std::get<LoweredModule>(std::move(frontend_result)));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        return *diagnostic;
    }

    auto plan_result = compilePlanModule(std::get<GraphModule>(std::move(graph_result)), BackendKind::Local);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&plan_result)) {
        return *diagnostic;
    }
    return std::get<PlanModule>(std::move(plan_result));
}

const SimpleTensor* single_tensor_output(const GraphExecutionResult& result) {
    if (result.outputs.size() != 1) {
        return nullptr;
    }
    return std::get_if<SimpleTensor>(&result.outputs.begin()->second);
}

bool matmul_relu_executes() {
    auto module_result = plan_module(
        "layer model(x: tensor[float32], w: tensor[float32]): tensor[float32]:\n"
        "  return relu(matmul(x, w))\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "executor-matmul: plan failed: " << diagnostic->toString() << '\n';
        return false;
    }

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {8, 1};
    options.tensor_shapes["w"] = {1, 8};
    auto reference_execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&reference_execution)) {
        std::cerr << "executor-matmul: reference execution failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const SimpleTensor* reference_output = single_tensor_output(std::get<GraphExecutionResult>(reference_execution));
    if (reference_output == nullptr) {
        std::cerr << "executor-matmul: missing reference output\n";
        return false;
    }

    RuntimeTensorWorkspace workspace;
    options.collect_intermediate_values = false;
    options.tensor_workspace = &workspace;
    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        std::cerr << "executor-matmul: execution failed: " << diagnostic->toString() << '\n';
        return false;
    }

    const SimpleTensor* output = single_tensor_output(std::get<GraphExecutionResult>(execution));
    if (output == nullptr || output->shape != std::vector<std::int64_t>{8, 8} || output->data.size() != 64) {
        std::cerr << "executor-matmul: unexpected output shape\n";
        return false;
    }
    if (output->data != reference_output->data) {
        std::cerr << "executor-matmul: fused output changed the result\n";
        return false;
    }
    if (!tensor_data_is_aligned(*output)) {
        std::cerr << "executor-matmul: tensor data is not aligned to " << tensor_data_alignment << " bytes\n";
        return false;
    }
    if (workspace.stats().allocations > 3) {
        std::cerr << "executor-matmul: fused execution allocated an intermediate tensor\n";
        return false;
    }
    return output->data[0] > 0.0F;
}

bool runtime_tensor_data_is_aligned() {
    SimpleTensor tensor = make_synthetic_tensor(std::vector<std::int64_t>{2, 3}, "float32");
    if (!tensor_data_is_aligned(tensor)) {
        std::cerr << "executor-alignment: synthetic tensor data is not aligned\n";
        return false;
    }
    auto reshaped = apply_reshape(tensor, std::vector<std::int64_t>{3, 2});
    if (const auto* diagnostic = std::get_if<Diagnostic>(&reshaped)) {
        std::cerr << "executor-alignment: reshape failed: " << diagnostic->toString() << '\n';
        return false;
    }
    SimpleTensor reshaped_tensor = std::get<SimpleTensor>(std::move(reshaped));
    if (!tensor_data_is_aligned(reshaped_tensor) || !tensor_data_shares_storage(tensor, reshaped_tensor)) {
        std::cerr << "executor-alignment: reshape should preserve aligned shared storage\n";
        return false;
    }

    const float original = static_cast<const SimpleTensor&>(tensor).data[0];
    reshaped_tensor.data[0] = -99.0F;
    if (static_cast<const SimpleTensor&>(tensor).data[0] != original ||
        tensor_data_shares_storage(tensor, reshaped_tensor)) {
        std::cerr << "executor-alignment: reshape view did not detach safely on mutation\n";
        return false;
    }

    SimpleTensor source_mutation = make_synthetic_tensor(std::vector<std::int64_t>{2, 3}, "float32");
    auto source_view_result = apply_reshape(source_mutation, std::vector<std::int64_t>{3, 2});
    if (const auto* diagnostic = std::get_if<Diagnostic>(&source_view_result)) {
        std::cerr << "executor-alignment: source reshape failed: " << diagnostic->toString() << '\n';
        return false;
    }
    SimpleTensor source_view = std::get<SimpleTensor>(std::move(source_view_result));
    const float view_original = static_cast<const SimpleTensor&>(source_view).data[0];
    source_mutation.data[0] = -123.0F;
    if (static_cast<const SimpleTensor&>(source_view).data[0] != view_original ||
        tensor_data_shares_storage(source_mutation, source_view)) {
        std::cerr << "executor-alignment: source tensor did not detach safely from reshape view\n";
        return false;
    }

    SimpleTensor heads = make_synthetic_tensor(std::vector<std::int64_t>{2, 3, 4}, "float32");
    auto flattened = apply_flatten_heads(heads);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&flattened)) {
        std::cerr << "executor-alignment: flatten_heads failed: " << diagnostic->toString() << '\n';
        return false;
    }
    SimpleTensor flattened_tensor = std::get<SimpleTensor>(std::move(flattened));
    if (flattened_tensor.shape != std::vector<std::int64_t>{2, 12} ||
        !tensor_data_shares_storage(heads, flattened_tensor)) {
        std::cerr << "executor-alignment: flatten_heads should be a metadata-only view\n";
        return false;
    }
    return true;
}

bool callable_linear_and_tanh_execute() {
    auto module_result = plan_module(
        "layer model(x: tensor[float32]): tensor[float32]:\n"
        "  proj = linear(3, 2, true)\n"
        "  return proj(x) -> Tanh()\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "executor-linear: plan failed: " << diagnostic->toString() << '\n';
        return false;
    }

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {2, 3};
    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        std::cerr << "executor-linear: execution failed: " << diagnostic->toString() << '\n';
        return false;
    }

    const SimpleTensor* output = single_tensor_output(std::get<GraphExecutionResult>(execution));
    if (output == nullptr || output->shape != std::vector<std::int64_t>{2, 2}) {
        std::cerr << "executor-linear: unexpected output shape\n";
        return false;
    }
    return std::all_of(output->data.begin(), output->data.end(), [](float value) {
        return value > -1.0F && value < 1.0F;
    });
}

bool module_local_function_calls_execute() {
    auto module_result = plan_module(
        "fn helper(x: tensor[float32]): tensor[float32]:\n"
        "  return sqrt(x)\n"
        "\n"
        "layer model(x: tensor[float32]): tensor[float32]:\n"
        "  return helper(x)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "executor-helper: plan failed: " << diagnostic->toString() << '\n';
        return false;
    }

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {1, 4};
    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        std::cerr << "executor-helper: execution failed: " << diagnostic->toString() << '\n';
        return false;
    }

    const SimpleTensor* output = single_tensor_output(std::get<GraphExecutionResult>(execution));
    if (output == nullptr || output->shape != std::vector<std::int64_t>{1, 4}) {
        std::cerr << "executor-helper: unexpected output shape\n";
        return false;
    }
    return std::abs(output->data[0] - std::sqrt(0.125F)) < 0.0001F;
}

bool missing_shape_returns_runtime_diagnostic() {
    auto module_result = plan_module(
        "layer model(x: tensor[float32]): tensor[float32]:\n"
        "  return relu(x)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "executor-missing-shape: plan failed: " << diagnostic->toString() << '\n';
        return false;
    }

    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", GraphExecutorOptions{});
    const auto* diagnostic = std::get_if<Diagnostic>(&execution);
    if (diagnostic == nullptr || diagnostic->code != DiagnosticCode::RuntimeError || diagnostic->message.find("--shape") == std::string::npos) {
        std::cerr << "executor-missing-shape: expected runtime shape diagnostic\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks{
        matmul_relu_executes(),
        runtime_tensor_data_is_aligned(),
        callable_linear_and_tanh_execute(),
        module_local_function_calls_execute(),
        missing_shape_returns_runtime_diagnostic(),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
