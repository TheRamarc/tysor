#include "frontend_ir.h"
#include "graph_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

GraphValue value(std::size_t id, bool is_parameter) {
    return GraphValue{id, "v" + std::to_string(id), FeType::int_type(), is_parameter, false};
}

GraphNode constant_node(std::size_t output) {
    return GraphNode{
        GraphNodeKind::Constant,
        output,
        std::string{},
        std::nullopt,
        FeBinaryOp::Add,
        FeValue::int_value(1),
        {},
    };
}

GraphNode binary_node(std::size_t output, std::vector<std::size_t> inputs) {
    return GraphNode{
        GraphNodeKind::Binary,
        output,
        std::string{},
        std::nullopt,
        FeBinaryOp::Add,
        FeValue::none(),
        std::move(inputs),
    };
}

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

bool graph_ok(const std::string& name, const std::string& source, std::size_t graphs, std::size_t nodes) {
    auto lowered = lower_module(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << name << ": frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    auto graph_result = build_graph_module(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << name << ": graph lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (module.functions.size() != graphs || module.functions.front().nodes.size() != nodes) {
        std::cerr << name << ": unexpected graph shape: " << graph_module_summary(module) << '\n'
                  << graph_ir_to_string(module) << '\n';
        return false;
    }
    return true;
}

bool matmul_relu_graph_ok() {
    auto lowered = lower_module(
        "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
        "  let y = matmul(x, w)\n"
        "  return relu(y)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << "matmul-relu-graph: frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    auto graph_result = build_graph_module(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "matmul-relu-graph: graph lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (module.functions.size() != 1 || module.functions.front().nodes.size() != 2) {
        std::cerr << "matmul-relu-graph: unexpected graph shape\n" << graph_ir_to_string(module) << '\n';
        return false;
    }
    const GraphFunction& graph = module.functions.front();
    if (graph.nodes[0].kind != GraphNodeKind::PrimitiveCall || graph.nodes[0].op != "matmul") {
        std::cerr << "matmul-relu-graph: expected first node to be primitive matmul\n";
        return false;
    }
    if (graph.nodes[1].kind != GraphNodeKind::PrimitiveCall || graph.nodes[1].op != "relu") {
        std::cerr << "matmul-relu-graph: expected second node to be primitive relu\n";
        return false;
    }
    if (graph.outputs.size() != 1 || graph.outputs.front() != graph.nodes[1].output) {
        std::cerr << "matmul-relu-graph: expected relu output as graph output\n";
        return false;
    }
    return true;
}

bool validation_accepts_valid_graph() {
    GraphFunction graph;
    graph.name = "valid";
    graph.values = {value(0, true), value(1, false)};
    graph.nodes = {constant_node(1)};
    graph.outputs = {1};
    graph.named_values["x"] = 0;

    if (auto diagnostic = validate_graph_function(graph)) {
        std::cerr << "validation-valid: unexpected diagnostic: " << diagnostic->to_string() << '\n';
        return false;
    }
    return true;
}

bool validation_rejects_missing_output() {
    GraphFunction graph;
    graph.name = "missing_output";
    graph.values = {value(0, true)};
    graph.outputs = {42};

    auto diagnostic = validate_graph_function(graph);
    if (!diagnostic || diagnostic->stage != "graph_ir" || diagnostic->code != "G0001" ||
        diagnostic->severity != DiagnosticSeverity::Error || !diagnostic->help ||
        diagnostic->message.find("output 42") == std::string::npos) {
        std::cerr << "validation-missing-output: expected structured output diagnostic\n";
        return false;
    }
    return true;
}

bool validation_rejects_read_before_production() {
    GraphFunction graph;
    graph.name = "bad_order";
    graph.values = {value(0, false), value(1, false)};
    graph.nodes = {binary_node(1, {0, 1})};
    graph.outputs = {1};

    auto diagnostic = validate_graph_function(graph);
    if (!diagnostic || diagnostic->message.find("before it is produced") == std::string::npos) {
        std::cerr << "validation-read-before-production: expected ordering diagnostic\n";
        return false;
    }
    return true;
}

bool validation_rejects_bad_binary_shape() {
    GraphFunction graph;
    graph.name = "bad_binary";
    graph.values = {value(0, true), value(1, false)};
    graph.nodes = {binary_node(1, {0})};
    graph.outputs = {1};

    auto diagnostic = validate_graph_function(graph);
    if (!diagnostic || diagnostic->message.find("exactly two inputs") == std::string::npos) {
        std::cerr << "validation-bad-binary: expected binary input-count diagnostic\n";
        return false;
    }
    return true;
}

bool graph_module_skips_non_straight_line() {
    auto lowered = lower_module(
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  if true:\n"
        "    return x\n"
        "  return x\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << "skip-if-graph: frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    auto graph_result = build_graph_module(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "skip-if-graph: graph lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (!module.functions.empty() || module.skipped.size() != 1 ||
        module.skipped.front().reason.find("straight-line") == std::string::npos) {
        std::cerr << "skip-if-graph: expected one skipped graph\n" << graph_ir_to_string(module) << '\n';
        return false;
    }
    return true;
}

bool frontend_only_list_decl_is_skipped() {
    auto lowered = lower_module(
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  let dims = [1, 2]\n"
        "  return x\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << "list-decl-skip: frontend lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    auto graph_result = build_graph_module(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "list-decl-skip: graph lowering failed: " << diagnostic->to_string() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (module.functions.size() != 1 || !module.skipped.empty() ||
        module.functions.front().values.size() != 1 || !module.functions.front().nodes.empty()) {
        std::cerr << "list-decl-skip: expected list declaration to stay frontend-only\n"
                  << graph_ir_to_string(module) << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks = {
        validation_accepts_valid_graph(),
        validation_rejects_missing_output(),
        validation_rejects_read_before_production(),
        validation_rejects_bad_binary_shape(),
        matmul_relu_graph_ok(),
        graph_ok(
            "binary-graph",
            "layer model(x: tensor[float16], y: tensor[float16]): tensor[float16]:\n"
            "  return x + y\n",
            1,
            1
        ),
        graph_ok(
            "apply-layer-ctor-graph",
            "layer model(x: tensor[float16]): tensor[float16]:\n"
            "  return x -> linear(16, 16)\n",
            1,
            4
        ),
        graph_module_skips_non_straight_line(),
        frontend_only_list_decl_is_skipped(),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
