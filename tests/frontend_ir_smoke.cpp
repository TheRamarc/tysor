#include "frontend_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

std::variant<Program, Diagnostic> parse_program(const std::string& source) {
    TokenizeResult tokenized = tokenize_with_diagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        return *diagnostic;
    }
    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parse_program();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }
    return std::get<Program>(std::move(parsed));
}

std::variant<LoweredModule, Diagnostic> lower_module(const std::string& source) {
    auto parsed = parse_program(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }

    SemanticAnalyzer analyzer;
    SemanticResult semantic_result = analyzer.analyze_with_info(std::get<Program>(parsed));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&semantic_result)) {
        return *diagnostic;
    }

    FrontendLowerer lowerer(std::get<Program>(parsed), std::get<SemanticInfo>(semantic_result));
    FrontendResult frontend_result = lowerer.lower();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        return *diagnostic;
    }
    return std::get<LoweredModule>(std::move(frontend_result));
}

bool lower_ok(const std::string& name, const std::string& source, std::size_t functions, std::size_t trains) {
    auto frontend_result = lower_module(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        std::cerr << name << ": frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    const LoweredModule& module = std::get<LoweredModule>(frontend_result);
    if (module.functions.size() != functions || module.trains.size() != trains) {
        std::cerr << name << ": unexpected lowered shape: "
                  << lowered_module_summary(module) << '\n';
        return false;
    }
    return true;
}

bool contains_ir(const std::string& name, const std::string& source, const std::string& expected) {
    auto frontend_result = lower_module(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        std::cerr << name << ": frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    std::string ir = frontend_ir_to_string(std::get<LoweredModule>(frontend_result));
    if (ir.find(expected) == std::string::npos) {
        std::cerr << name << ": expected IR fragment not found: " << expected << "\n" << ir << '\n';
        return false;
    }
    return true;
}

bool local_objective_ok() {
    auto frontend_result = lower_module(
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  loss = relu(x)\n"
        "  return x\n"
        "\n"
        "config model:\n"
        "  optimizer = adam\n"
        "  objective = loss\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        std::cerr << "local-objective: frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    const auto& module = std::get<LoweredModule>(frontend_result);
    if (!module.execution_plan || module.execution_plan->runs.empty()) {
        std::cerr << "local-objective: missing execution plan\n";
        return false;
    }
    if (module.execution_plan->runs.front().objective_source != ObjectiveSource::Local) {
        std::cerr << "local-objective: expected Local objective source\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks = {
        lower_ok(
            "matmul-relu",
            "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
            "  y = matmul(x, w)\n"
            "  return relu(y)\n",
            1,
            0
        ),
        lower_ok(
            "train-config",
            "layer model(loss: tensor[float16]): tensor[float16]:\n"
            "  return loss\n"
            "\n"
            "config model:\n"
            "  optimizer = adam\n"
            "  lr = 1e-4\n"
            "  objective = loss\n",
            1,
            1
        ),
        local_objective_ok(),
        contains_ir(
            "compound-arrow-tuple",
            "layer model(x: tensor[float16]): (tensor[float16], tensor[float16]):\n"
            "  return x -> (relu(), x)\n",
            "return (relu(x), x)"
        ),
        contains_ir(
            "compound-arrow-list",
            "layer model(x: tensor[float16]): [tensor[float16]]:\n"
            "  return x -> [relu()]\n",
            "return [relu(x)]"
        ),
        contains_ir(
            "compound-arrow-unary",
            "layer model(x: tensor[float16]): tensor[float16]:\n"
            "  return x -> -relu()\n",
            "return (0 - relu(x))"
        ),
        contains_ir(
            "compound-arrow-ternary",
            "layer model(x: tensor[float16], flag: bool): tensor[float16]:\n"
            "  return x -> relu() if flag else x\n",
            "return if flag then relu(x) else x"
        ),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
