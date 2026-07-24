#include "frontend_ir.h"
#include "execution_plan.h"
#include "graph_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

GraphValue value(std::size_t id, bool isParameter) {
    return GraphValue{id, "v" + std::to_string(id), FeType::intType(), isParameter, false};
}

GraphNode constant_node(std::size_t output) {
    return GraphNode{
        GraphNodeKind::Constant,
        output,
        std::string{},
        std::nullopt,
        FeBinaryOp::Add,
        FeValue::intValue(1),
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

std::variant<LoweredModule, Diagnostic> lowerModule(const std::string& source) {
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
    return std::get<LoweredModule>(std::move(frontend_result));
}

bool graph_ok(const std::string& name, const std::string& source, std::size_t graphs, std::size_t nodes) {
    auto lowered = lowerModule(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << name << ": frontend lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }

    auto graph_result = buildGraphModule(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << name << ": graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    std::size_t total_graphs = module.functions.size() + module.layers.size();
    std::size_t actual_nodes = !module.layers.empty() ? module.layers.front().nodes.size() : (!module.functions.empty() ? module.functions.front().nodes.size() : 0);
    if (total_graphs != graphs || actual_nodes != nodes) {
        std::cerr << name << ": unexpected graph shape: " << graphModuleSummary(module) << '\n'
                  << graphIrToString(module) << '\n';
        return false;
    }
    return true;
}

std::variant<GraphLayer, Diagnostic> graph_from_source(const std::string& source) {
    auto lowered = lowerModule(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        return *diagnostic;
    }
    LoweredModule module = std::get<LoweredModule>(std::move(lowered));
    if (!module.layers.empty()) {
        return buildGraphLayer(module.layers.front());
    }
    if (!module.functions.empty()) {
        auto res = buildGraphFunction(module.functions.front());
        if (const auto* diagnostic = std::get_if<Diagnostic>(&res)) {
            return *diagnostic;
        }
        auto g_fn = std::get<GraphFunction>(res);
        GraphLayer layer;
        layer.name = g_fn.name;
        layer.returnType = g_fn.returnType;
        layer.values = g_fn.values;
        layer.nodes = g_fn.nodes;
        layer.inputs = g_fn.inputs;
        layer.outputs = g_fn.outputs;
        layer.namedValues = g_fn.namedValues;
        return layer;
    }
    return Diagnostic::error(DiagnosticCode::TestError, "expected lowered layer");
}

template <typename GraphT>
const GraphTensorType* named_tensor_type(const GraphT& graph, const std::string& name) {
    auto found = graph.namedValues.find(name);
    if (found == graph.namedValues.end()) {
        return nullptr;
    }
    if (found->second >= graph.values.size()) {
        return nullptr;
    }
    return graph.values[found->second].tensorType ? &*graph.values[found->second].tensorType : nullptr;
}

template <typename GraphT>
bool expect_named_tensor(
    const std::string& test_name,
    const GraphT& graph,
    const std::string& value_name,
    const std::string& expected
) {
    const GraphTensorType* tensor = named_tensor_type(graph, value_name);
    if (tensor == nullptr) {
        std::cerr << test_name << ": missing tensor metadata for '" << value_name << "'\n";
        return false;
    }
    const std::string actual = graphTensorTypeToString(*tensor);
    if (actual != expected) {
        std::cerr << test_name << ": expected '" << value_name << "' tensor " << expected
                  << ", got " << actual << '\n';
        return false;
    }
    return true;
}

template <typename GraphT>
const GraphParameter* find_graph_parameter(
    const GraphT& graph,
    const std::string& role,
    std::size_t ownerValue
) {
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        auto found = std::find_if(graph.parameters.begin(), graph.parameters.end(), [&](const GraphParameter& parameter) {
            return parameter.role == role && parameter.ownerValue == ownerValue;
        });
        return found == graph.parameters.end() ? nullptr : &*found;
    } else {
        return nullptr;
    }
}

const PlanParameter* find_plan_parameter(
    const ExecutionPlan& plan,
    const std::string& role,
    std::size_t ownerValue
) {
    auto found = std::find_if(plan.parameters.begin(), plan.parameters.end(), [&](const PlanParameter& parameter) {
        return parameter.role == role && parameter.ownerValue == ownerValue;
    });
    return found == plan.parameters.end() ? nullptr : &*found;
}

template <typename GraphT>
bool expect_parameter(
    const std::string& test_name,
    const GraphT& graph,
    const std::string& role,
    std::size_t ownerValue,
    const std::string& expected_name,
    const std::string& expected_tensor
) {
    const GraphParameter* parameter = find_graph_parameter(graph, role, ownerValue);
    if (parameter == nullptr) {
        std::cerr << test_name << ": missing graph parameter role=" << role
                  << " owner=%" << ownerValue << '\n';
        return false;
    }
    if (parameter->valueId >= graph.values.size() || !graph.values[parameter->valueId].isModelParameter) {
        std::cerr << test_name << ": parameter " << parameter->name
                  << " does not reference a model parameter value\n";
        return false;
    }
    if (parameter->name != expected_name ||
        graphTensorTypeToString(parameter->tensorType) != expected_tensor ||
        !parameter->trainable) {
        std::cerr << test_name << ": unexpected parameter " << parameter->name
                  << " tensor=" << graphTensorTypeToString(parameter->tensorType) << '\n';
        return false;
    }
    return true;
}

bool matmul_relu_graph_ok() {
    auto lowered = lowerModule(
        "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
        "  y = matmul(x, w)\n"
        "  return relu(y)\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << "matmul-relu-graph: frontend lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    auto graph_result = buildGraphModule(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "matmul-relu-graph: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (module.layers.size() != 1 || module.layers.front().nodes.size() != 2) {
        std::cerr << "matmul-relu-graph: unexpected graph shape\n" << graphIrToString(module) << '\n';
        return false;
    }
    const GraphLayer& graph = module.layers.front();
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

bool graph_infers_linear_symbolic_shape() {
    auto graph_result = graph_from_source(
        "layer model(x: tensor[float32, [batch, 3]]): tensor[float32]:\n"
        "  proj = linear(3, 4, true)\n"
        "  y = x -> proj()\n"
        "  return y\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "linear-shape-inference: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphLayer& graph = std::get<GraphLayer>(graph_result);
    if (!expect_named_tensor("linear-shape-inference", graph, "x", "float32[batch, 3]")) {
        return false;
    }
    if (!expect_named_tensor("linear-shape-inference", graph, "y", "float32[batch, 4]")) {
        return false;
    }
    if (graph.parameters.size() != 2) {
        std::cerr << "linear-shape-inference: expected weight and bias parameters\n"
                  << graphIrToString(makeGraphModule(graph)) << '\n';
        return false;
    }
    const std::size_t owner = graph.namedValues.at("proj");
    const std::string expected_weight = "linear_" + std::to_string(owner) + "_weight";
    const std::string expected_bias = "linear_" + std::to_string(owner) + "_bias";
    if (!expect_parameter("linear-shape-inference", graph, "weight", owner, expected_weight, "float32[3, 4]") ||
        !expect_parameter("linear-shape-inference", graph, "bias", owner, expected_bias, "float32[4]")) {
        return false;
    }
    const auto has_apply = std::any_of(graph.nodes.begin(), graph.nodes.end(), [](const GraphNode& node) {
        return node.kind == GraphNodeKind::Apply;
    });
    const auto has_matmul = std::any_of(graph.nodes.begin(), graph.nodes.end(), [](const GraphNode& node) {
        return node.kind == GraphNodeKind::PrimitiveCall && node.op == "matmul";
    });
    const auto has_bias_add = std::any_of(graph.nodes.begin(), graph.nodes.end(), [](const GraphNode& node) {
        return node.kind == GraphNodeKind::Binary && node.binaryOp == FeBinaryOp::Add;
    });
    if (has_apply || !has_matmul || !has_bias_add) {
        std::cerr << "linear-shape-inference: expected linear apply to lower to matmul plus bias add\n"
                  << graphIrToString(makeGraphModule(graph)) << '\n';
        return false;
    }
    ExecutionPlan plan = compileLocalExecutionPlan(graph);
    const auto y = graph.namedValues.at("y");
    if (!plan.values[y].tensorType || graphTensorTypeToString(*plan.values[y].tensorType) != "float32[batch, 4]") {
        std::cerr << "linear-shape-inference: execution plan did not preserve inferred tensor metadata\n";
        return false;
    }
    const PlanParameter* weight = find_plan_parameter(plan, "weight", owner);
    if (weight == nullptr || weight->name != expected_weight ||
        graphTensorTypeToString(weight->tensorType) != "float32[3, 4]" ||
        weight->valueId >= plan.values.size() ||
        !plan.values[weight->valueId].isModelParameter) {
        std::cerr << "linear-shape-inference: execution plan did not preserve parameter metadata\n";
        return false;
    }
    return true;
}

bool graph_infers_reshape_and_flatten_shapes() {
    auto graph_result = graph_from_source(
        "layer model(x: tensor[float32, [2, 3, 4]]): tensor[float32]:\n"
        "  flat = flatten_heads(x)\n"
        "  reshaped = reshape(flat, 4, 6)\n"
        "  return reshaped\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "reshape-flatten-shape-inference: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphLayer& graph = std::get<GraphLayer>(graph_result);
    return expect_named_tensor("reshape-flatten-shape-inference", graph, "flat", "float32[2, 12]") &&
           expect_named_tensor("reshape-flatten-shape-inference", graph, "reshaped", "float32[4, 6]");
}

bool graph_records_embedding_parameter() {
    auto graph_result = graph_from_source(
        "layer model(idx: tensor[float32, [batch]]): tensor[float32]:\n"
        "  tok = Embedding(32, 8)\n"
        "  y = idx -> tok()\n"
        "  return y\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "embedding-parameter: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphLayer& graph = std::get<GraphLayer>(graph_result);
    if (!expect_named_tensor("embedding-parameter", graph, "y", "float32[batch, 8]")) {
        return false;
    }
    const std::size_t owner = graph.namedValues.at("tok");
    return expect_parameter(
        "embedding-parameter",
        graph,
        "weight",
        owner,
        "embedding_" + std::to_string(owner) + "_weight",
        "float32[32, 8]"
    );
}

bool graph_rejects_matmul_inner_mismatch() {
    auto graph_result = graph_from_source(
        "layer model(x: tensor[float32, [2, 3]], w: tensor[float32, [4, 5]]): tensor[float32]:\n"
        "  return matmul(x, w)\n"
    );
    const auto* diagnostic = std::get_if<Diagnostic>(&graph_result);
    if (diagnostic == nullptr || diagnostic->message.find("matmul inner") == std::string::npos) {
        std::cerr << "matmul-shape-mismatch: expected matmul inner dimension diagnostic\n";
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
    graph.namedValues["x"] = 0;

    if (auto diagnostic = validateGraphFunction(graph)) {
        std::cerr << "validation-valid: unexpected diagnostic: " << diagnostic->toString() << '\n';
        return false;
    }
    return true;
}

bool validation_rejects_missing_output() {
    GraphFunction graph;
    graph.name = "missing_output";
    graph.values = {value(0, true)};
    graph.outputs = {42};

    auto diagnostic = validateGraphFunction(graph);
    if (!diagnostic || diagnostic->code != DiagnosticCode::GraphIrError ||
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

    auto diagnostic = validateGraphFunction(graph);
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

    auto diagnostic = validateGraphFunction(graph);
    if (!diagnostic || diagnostic->message.find("exactly two inputs") == std::string::npos) {
        std::cerr << "validation-bad-binary: expected binary input-count diagnostic\n";
        return false;
    }
    return true;
}

bool graph_module_skips_non_straight_line() {
    auto lowered = lowerModule(
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  if true:\n"
        "    return x\n"
        "  return x\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << "skip-if-graph: frontend lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    auto graph_result = buildGraphModule(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "skip-if-graph: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (!module.layers.empty() || module.skipped.size() != 1 ||
        module.skipped.front().reason.find("straight-line") == std::string::npos) {
        std::cerr << "skip-if-graph: expected one skipped graph\n" << graphIrToString(module) << '\n';
        return false;
    }
    return true;
}

bool frontend_only_list_decl_is_skipped() {
    auto lowered = lowerModule(
        "layer model(x: tensor[float16]): tensor[float16]:\n"
        "  dims = [1, 2]\n"
        "  return x\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
        std::cerr << "list-decl-skip: frontend lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    auto graph_result = buildGraphModule(std::get<LoweredModule>(lowered));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
        std::cerr << "list-decl-skip: graph lowering failed: " << diagnostic->toString() << '\n';
        return false;
    }
    const GraphModule& module = std::get<GraphModule>(graph_result);
    if (module.layers.size() != 1 || !module.skipped.empty() ||
        module.layers.front().values.size() != 1 || !module.layers.front().nodes.empty()) {
        std::cerr << "list-decl-skip: expected list declaration to stay frontend-only\n"
                  << graphIrToString(module) << '\n';
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
        graph_infers_linear_symbolic_shape(),
        graph_infers_reshape_and_flatten_shapes(),
        graph_records_embedding_parameter(),
        graph_rejects_matmul_inner_mismatch(),
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
            5
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
