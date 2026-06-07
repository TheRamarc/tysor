#pragma once

#include "diagnostic.h"
#include "frontend_ir.h"

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

enum class GraphNodeKind {
    Constant,
    Binary,
    PrimitiveCall,
    LibraryCall,
    LibraryCtor,
    Apply,
};

struct GraphValue {
    std::size_t id = 0;
    std::string name;
    FeType type;
    bool is_parameter = false;
    bool requires_grad = false;
};

struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Constant;
    std::size_t output = 0;
    std::string op;
    std::optional<std::string> op_id;
    FeBinaryOp binary_op = FeBinaryOp::Add;
    FeValue constant = FeValue::none();
    std::vector<std::size_t> inputs;
};

struct GraphFunction {
    std::string name;
    bool is_layer = false;
    FeType return_type;
    std::vector<GraphValue> values;
    std::vector<GraphNode> nodes;
    std::vector<std::size_t> outputs;
    std::map<std::string, std::size_t> named_values;
};

struct GraphBuildSkipped {
    std::string function_name;
    std::string reason;
};

struct GraphModule {
    std::vector<GraphFunction> functions;
    std::vector<GraphBuildSkipped> skipped;
};

using GraphFunctionResult = std::variant<GraphFunction, Diagnostic>;

class GraphBuilder {
public:
    explicit GraphBuilder(const FeFunction& function);

    GraphFunctionResult build();

private:
    const FeFunction& function_;

    std::variant<std::size_t, Diagnostic> lower_expr(const FeExprPtr& expr, GraphFunction& graph) const;
    std::variant<std::size_t, Diagnostic> lookup_named_value(const std::string& name, const GraphFunction& graph) const;
};

GraphFunctionResult build_graph_function(const FeFunction& function);
std::variant<GraphModule, Diagnostic> build_graph_module(const LoweredModule& module);
std::optional<Diagnostic> validate_graph_function(const GraphFunction& graph);
std::string graph_module_summary(const GraphModule& module);
std::string graph_ir_to_string(const GraphModule& module);
