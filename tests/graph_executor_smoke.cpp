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
    TokenizeResult tokenized = tokenize_with_diagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        return *diagnostic;
    }
    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parse_program();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }

    Program program = std::get<Program>(std::move(parsed));
    SemanticAnalyzer analyzer;
    SemanticResult semantic_result = analyzer.analyze_with_info(program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&semantic_result)) {
        return *diagnostic;
    }

    FrontendLowerer lowerer(program, std::get<SemanticInfo>(semantic_result));
    FrontendResult frontend_result = lowerer.lower();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        return *diagnostic;
    }

    auto graph_result = build_graph_module(std::get<LoweredModule>(std::move(frontend_result)));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        return *diagnostic;
    }

    auto plan_result = compile_plan_module(std::get<GraphModule>(std::move(graph_result)), BackendKind::Local);
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
        std::cerr << "executor-matmul: plan failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {2, 3};
    options.tensor_shapes["w"] = {3, 2};
    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        std::cerr << "executor-matmul: execution failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    const SimpleTensor* output = single_tensor_output(std::get<GraphExecutionResult>(execution));
    if (output == nullptr || output->shape != std::vector<std::int64_t>{2, 2} || output->data.size() != 4) {
        std::cerr << "executor-matmul: unexpected output shape\n";
        return false;
    }
    if (!tensor_data_is_aligned(*output)) {
        std::cerr << "executor-matmul: tensor data is not aligned to " << tensor_data_alignment << " bytes\n";
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
        std::cerr << "executor-alignment: reshape failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    return tensor_data_is_aligned(std::get<SimpleTensor>(reshaped));
}

bool callable_linear_and_tanh_execute() {
    auto module_result = plan_module(
        "layer model(x: tensor[float32]): tensor[float32]:\n"
        "  let proj = linear(3, 2, true)\n"
        "  return proj(x) -> Tanh()\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&module_result)) {
        std::cerr << "executor-linear: plan failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {2, 3};
    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        std::cerr << "executor-linear: execution failed: " << diagnostic->to_string() << '\n';
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
        std::cerr << "executor-helper: plan failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    GraphExecutorOptions options;
    options.tensor_shapes["x"] = {1, 4};
    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        std::cerr << "executor-helper: execution failed: " << diagnostic->to_string() << '\n';
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
        std::cerr << "executor-missing-shape: plan failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    auto execution = execute_plan_module(std::get<PlanModule>(module_result), "model", GraphExecutorOptions{});
    const auto* diagnostic = std::get_if<Diagnostic>(&execution);
    if (diagnostic == nullptr || diagnostic->stage != "runtime" || diagnostic->message.find("--shape") == std::string::npos) {
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
