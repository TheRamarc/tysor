#include "graph_ir.h"

#include "ops.h"

#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace {

Diagnostic graph_error(std::string message) {
    return Diagnostic::error("graph_ir", "G0001", std::move(message))
        .with_help("Graph IR validation failed before backend planning. This usually means frontend lowering produced an invalid graph boundary.");
}

const char* graph_node_kind_name(GraphNodeKind kind) {
    switch (kind) {
        case GraphNodeKind::Constant:
            return "constant";
        case GraphNodeKind::Binary:
            return "binary";
        case GraphNodeKind::PrimitiveCall:
            return "primitive_call";
        case GraphNodeKind::LibraryCall:
            return "library_call";
        case GraphNodeKind::LibraryCtor:
            return "library_ctor";
        case GraphNodeKind::Apply:
            return "apply";
    }
    return "unknown";
}

std::string fe_type_to_graph_string(const FeType& type) {
    switch (type.kind) {
        case FeTypeKind::Unknown:
            return "unknown";
        case FeTypeKind::Int:
            return type.scalar_dtype.value_or("int");
        case FeTypeKind::Float:
            return type.scalar_dtype.value_or("float");
        case FeTypeKind::Bool:
            return "bool";
        case FeTypeKind::str:
            return "str";
        case FeTypeKind::Tensor:
            if (type.tensor_dtype && type.tensor_shape_expr) {
                return "tensor[" + *type.tensor_dtype + ", " + *type.tensor_shape_expr + "]";
            }
            if (type.tensor_dtype) {
                return "tensor[" + *type.tensor_dtype + "]";
            }
            if (type.tensor_shape_expr) {
                return "tensor[" + *type.tensor_shape_expr + "]";
            }
            return "tensor";
        case FeTypeKind::Tuple: {
            std::ostringstream out;
            out << '(';
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    out << ", ";
                }
                out << fe_type_to_graph_string(type.elements[index]);
            }
            out << ')';
            return out.str();
        }
        case FeTypeKind::List: {
            std::ostringstream out;
            out << '[';
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    out << ", ";
                }
                out << fe_type_to_graph_string(type.elements[index]);
            }
            out << ']';
            return out.str();
        }
        case FeTypeKind::Callable:
            return "callable -> " + (type.callable_return ? fe_type_to_graph_string(*type.callable_return) : "void");
        case FeTypeKind::Void:
            return "void";
        case FeTypeKind::None:
            return "none";
    }
    return "unknown";
}

std::string fe_value_to_graph_string(const FeValue& value) {
    return std::visit(
        [](const auto& inner) -> std::string {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return "None";
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(inner);
            } else if constexpr (std::is_same_v<T, double>) {
                std::ostringstream out;
                out << inner;
                return out.str();
            } else if constexpr (std::is_same_v<T, bool>) {
                return inner ? "true" : "false";
            } else if constexpr (std::is_same_v<T, std::string>) {
                return '"' + inner + '"';
            } else if constexpr (std::is_same_v<T, FeTupleValue>) {
                std::ostringstream out;
                out << '[';
                for (std::size_t index = 0; index < inner.values.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    out << fe_value_to_graph_string(inner.values[index]);
                }
                out << ']';
                return out.str();
            } else if constexpr (std::is_same_v<T, FeListValue>) {
                std::ostringstream out;
                out << '[';
                for (std::size_t index = 0; index < inner.values.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    out << fe_value_to_graph_string(inner.values[index]);
                }
                out << ']';
                return out.str();
            }
        },
        value.value
    );
}

std::string fe_binary_op_to_graph_string(FeBinaryOp op) {
    switch (op) {
        case FeBinaryOp::Add:
            return "+";
        case FeBinaryOp::Sub:
            return "-";
        case FeBinaryOp::Mul:
            return "*";
        case FeBinaryOp::Div:
            return "/";
        case FeBinaryOp::FloorDiv:
            return "//";
        case FeBinaryOp::Eq:
            return "==";
        case FeBinaryOp::NotEq:
            return "!=";
        case FeBinaryOp::Lt:
            return "<";
        case FeBinaryOp::Gt:
            return ">";
        case FeBinaryOp::LtEq:
            return "<=";
        case FeBinaryOp::GtEq:
            return ">=";
        case FeBinaryOp::And:
            return "&&";
        case FeBinaryOp::Or:
            return "||";
        case FeBinaryOp::Not:
            return "!";
    }
    return "?";
}

std::size_t append_value(GraphFunction& graph, std::string name, FeType type, bool is_parameter) {
    const std::size_t id = graph.values.size();
    const bool requires_grad = is_parameter && type.kind == FeTypeKind::Tensor;
    graph.values.push_back(GraphValue{id, name, std::move(type), is_parameter, requires_grad});
    if (!name.empty()) {
        graph.named_values[name] = id;
    }
    return id;
}

std::optional<std::string> graph_op_id_name(const std::string& name) {
    auto id = lookup_op_id(name);
    if (!id) {
        return std::nullopt;
    }
    return std::string(op_id_name(*id));
}

bool is_frontend_only_expr(const FeExprPtr& expr) {
    return expr && std::holds_alternative<FeListExpr>(expr->kind);
}

std::optional<Diagnostic> validate_node_shape(
    const GraphFunction& graph,
    std::size_t index,
    GraphNodeKind kind,
    std::size_t input_count
) {
    if (kind == GraphNodeKind::Constant && input_count != 0) {
        return graph_error("Graph '" + graph.name + "' constant node #" + std::to_string(index) + " must not have inputs");
    }
    if (kind == GraphNodeKind::Binary && input_count != 2) {
        return graph_error("Graph '" + graph.name + "' binary node #" + std::to_string(index) + " must have exactly two inputs");
    }
    if (kind == GraphNodeKind::Apply && input_count < 2) {
        return graph_error("Graph '" + graph.name + "' apply node #" + std::to_string(index) + " must have a callee and an input");
    }
    return std::nullopt;
}

void append_inputs(std::ostringstream& out, const std::vector<std::size_t>& inputs) {
    out << '[';
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << '%' << inputs[index];
    }
    out << ']';
}

} // namespace

GraphBuilder::GraphBuilder(const FeFunction& function) : function_(function) {}

GraphFunctionResult GraphBuilder::build() {
    GraphFunction graph;
    graph.name = function_.name;
    graph.is_layer = function_.is_layer;
    graph.return_type = function_.return_type;

    for (const auto& param : function_.params) {
        append_value(graph, param.first, param.second, true);
    }

    for (const auto& stmt : function_.body) {
        if (const auto* decl = std::get_if<FeVarDeclStmt>(&stmt.kind)) {
            if (is_frontend_only_expr(decl->value)) {
                continue;
            }
            if (!decl->has_value || !decl->value) {
                return graph_error("Graph builder expected value");
            }
            auto value_id = lower_expr(decl->value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&value_id)) {
                return *diagnostic;
            }
            graph.named_values[decl->name] = std::get<std::size_t>(value_id);
            graph.values[std::get<std::size_t>(value_id)].name = decl->name;
        } else if (const auto* assign = std::get_if<FeAssignStmt>(&stmt.kind)) {
            if (is_frontend_only_expr(assign->value)) {
                continue;
            }
            auto value_id = lower_expr(assign->value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&value_id)) {
                return *diagnostic;
            }
            graph.named_values[assign->name] = std::get<std::size_t>(value_id);
            graph.values[std::get<std::size_t>(value_id)].name = assign->name;
        } else if (const auto* ret = std::get_if<FeReturnStmt>(&stmt.kind)) {
            auto output_id = lower_expr(ret->value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&output_id)) {
                return *diagnostic;
            }
            graph.outputs.push_back(std::get<std::size_t>(output_id));
        } else {
            return graph_error("Graph builder currently supports straight-line statements only");
        }
    }

    if (auto diagnostic = validate_graph_function(graph)) {
        return *diagnostic;
    }
    return graph;
}

std::variant<std::size_t, Diagnostic> GraphBuilder::lookup_named_value(
    const std::string& name,
    const GraphFunction& graph
) const {
    auto found = graph.named_values.find(name);
    if (found == graph.named_values.end()) {
        return graph_error("Graph builder could not resolve symbol '" + name + "'");
    }
    return found->second;
}

std::variant<std::size_t, Diagnostic> GraphBuilder::lower_expr(const FeExprPtr& expr, GraphFunction& graph) const {
    if (!expr) {
        return graph_error("Graph builder expected expression");
    }
    if (const auto* constant = std::get_if<FeConstantExpr>(&expr->kind)) {
        const std::size_t output = append_value(graph, std::string{}, expr->type, false);
        graph.nodes.push_back(GraphNode{
            GraphNodeKind::Constant,
            output,
            std::string{},
            std::nullopt,
            FeBinaryOp::Add,
            constant->value,
            {},
        });
        return output;
    }
    if (const auto* var = std::get_if<FeVarExpr>(&expr->kind)) {
        return lookup_named_value(var->symbol, graph);
    }
    if (const auto* binary = std::get_if<FeBinaryExpr>(&expr->kind)) {
        auto lhs = lower_expr(binary->lhs, graph);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
            return *diagnostic;
        }
        auto rhs = lower_expr(binary->rhs, graph);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
            return *diagnostic;
        }
        const std::size_t output = append_value(graph, std::string{}, expr->type, false);
        graph.nodes.push_back(GraphNode{
            GraphNodeKind::Binary,
            output,
            std::string{},
            std::nullopt,
            binary->op,
            FeValue::none(),
            {std::get<std::size_t>(lhs), std::get<std::size_t>(rhs)},
        });
        return output;
    }
    if (const auto* call = std::get_if<FeCallExpr>(&expr->kind)) {
        std::vector<std::size_t> inputs;
        for (const auto& arg : call->args) {
            auto lowered = lower_expr(arg.value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            inputs.push_back(std::get<std::size_t>(lowered));
        }
        const std::size_t output = append_value(graph, std::string{}, expr->type, false);
        graph.nodes.push_back(GraphNode{
            is_primitive_tensor_op(call->callee) ? GraphNodeKind::PrimitiveCall : GraphNodeKind::LibraryCall,
            output,
            call->callee,
            graph_op_id_name(call->callee),
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        });
        return output;
    }
    if (const auto* ctor = std::get_if<FeLayerCtorExpr>(&expr->kind)) {
        std::vector<std::size_t> inputs;
        for (const auto& arg : ctor->args) {
            auto lowered = lower_expr(arg.value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            inputs.push_back(std::get<std::size_t>(lowered));
        }
        const std::size_t output = append_value(graph, std::string{}, expr->type, false);
        graph.nodes.push_back(GraphNode{
            GraphNodeKind::LibraryCtor,
            output,
            ctor->callee,
            graph_op_id_name(ctor->callee),
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        });
        return output;
    }
    if (const auto* apply = std::get_if<FeApplyExpr>(&expr->kind)) {
        std::vector<std::size_t> inputs;
        auto callee = lower_expr(apply->callee, graph);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&callee)) {
            return *diagnostic;
        }
        inputs.push_back(std::get<std::size_t>(callee));
        for (const auto& arg : apply->args) {
            auto lowered = lower_expr(arg.value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            inputs.push_back(std::get<std::size_t>(lowered));
        }
        const std::size_t output = append_value(graph, std::string{}, expr->type, false);
        graph.nodes.push_back(GraphNode{
            GraphNodeKind::Apply,
            output,
            std::string{},
            std::nullopt,
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        });
        return output;
    }
    if (std::holds_alternative<FeListExpr>(expr->kind)) {
        return graph_error("Graph builder does not support list expressions yet");
    }
    return graph_error("Graph builder does not support this FE expression kind yet");
}

GraphFunctionResult build_graph_function(const FeFunction& function) {
    return GraphBuilder(function).build();
}

std::variant<GraphModule, Diagnostic> build_graph_module(const LoweredModule& module) {
    GraphModule graph_module;
    for (const auto& function : module.functions) {
        auto graph = build_graph_function(function);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&graph)) {
            graph_module.skipped.push_back(GraphBuildSkipped{function.name, diagnostic->message});
            continue;
        }
        graph_module.functions.push_back(std::get<GraphFunction>(std::move(graph)));
    }
    return graph_module;
}

std::optional<Diagnostic> validate_graph_function(const GraphFunction& graph) {
    std::set<std::size_t> value_ids;
    for (const auto& value : graph.values) {
        value_ids.insert(value.id);
    }
    if (value_ids.size() != graph.values.size()) {
        return graph_error("Graph '" + graph.name + "' has duplicate value ids");
    }
    for (std::size_t index = 0; index < graph.values.size(); ++index) {
        if (graph.values[index].id != index) {
            return graph_error(
                "Graph '" + graph.name + "' value id " + std::to_string(graph.values[index].id) +
                " is out of order at index " + std::to_string(index)
            );
        }
    }
    if (graph.outputs.empty()) {
        return graph_error("Graph '" + graph.name + "' must have at least one output");
    }
    for (const auto output : graph.outputs) {
        if (value_ids.find(output) == value_ids.end()) {
            return graph_error("Graph '" + graph.name + "' output " + std::to_string(output) + " does not reference a value");
        }
    }
    for (const auto& named : graph.named_values) {
        if (named.first.empty()) {
            return graph_error("Graph '" + graph.name + "' has an empty named value");
        }
        if (value_ids.find(named.second) == value_ids.end()) {
            return graph_error(
                "Graph '" + graph.name + "' named value '" + named.first +
                "' references missing value " + std::to_string(named.second)
            );
        }
    }

    std::set<std::size_t> produced_values;
    for (const auto& value : graph.values) {
        if (value.is_parameter) {
            produced_values.insert(value.id);
        }
    }
    for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
        const auto& node = graph.nodes[index];
        if (value_ids.find(node.output) == value_ids.end()) {
            return graph_error(
                "Graph '" + graph.name + "' node #" + std::to_string(index) +
                " outputs missing value " + std::to_string(node.output)
            );
        }
        if (produced_values.find(node.output) != produced_values.end()) {
            return graph_error(
                "Graph '" + graph.name + "' node #" + std::to_string(index) +
                " writes value " + std::to_string(node.output) + " more than once"
            );
        }
        if (auto diagnostic = validate_node_shape(graph, index, node.kind, node.inputs.size())) {
            return diagnostic;
        }
        for (const auto input : node.inputs) {
            if (value_ids.find(input) == value_ids.end()) {
                return graph_error(
                    "Graph '" + graph.name + "' node #" + std::to_string(index) +
                    " input " + std::to_string(input) + " does not reference a value"
                );
            }
            if (produced_values.find(input) == produced_values.end()) {
                return graph_error(
                    "Graph '" + graph.name + "' node #" + std::to_string(index) +
                    " reads value " + std::to_string(input) + " before it is produced"
                );
            }
        }
        produced_values.insert(node.output);
    }
    for (const auto output : graph.outputs) {
        if (produced_values.find(output) == produced_values.end()) {
            return graph_error("Graph '" + graph.name + "' output " + std::to_string(output) + " is never produced");
        }
    }
    return std::nullopt;
}

std::string graph_module_summary(const GraphModule& module) {
    std::ostringstream out;
    out << "graph=functions:" << module.functions.size() << " skipped:" << module.skipped.size();
    return out.str();
}

std::string graph_ir_to_string(const GraphModule& module) {
    std::ostringstream out;
    out << graph_module_summary(module) << '\n';
    for (const auto& graph : module.functions) {
        out << (graph.is_layer ? "graph layer " : "graph fn ") << graph.name
            << " values=" << graph.values.size()
            << " nodes=" << graph.nodes.size()
            << " outputs=" << graph.outputs.size()
            << " -> " << fe_type_to_graph_string(graph.return_type) << '\n';
        for (const auto& value : graph.values) {
            out << "  %" << value.id;
            if (!value.name.empty()) {
                out << ' ' << value.name;
            }
            out << ": " << fe_type_to_graph_string(value.type);
            if (value.is_parameter) {
                out << " param";
            }
            if (value.requires_grad) {
                out << " requires_grad";
            }
            out << '\n';
        }
        for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
            const auto& node = graph.nodes[index];
            out << "  node #" << index << ' ' << graph_node_kind_name(node.kind)
                << " -> %" << node.output;
            if (!node.op.empty()) {
                out << " op=" << node.op;
            }
            if (node.op_id) {
                out << " op_id=" << *node.op_id;
            }
            if (node.kind == GraphNodeKind::Binary) {
                out << " op=" << fe_binary_op_to_graph_string(node.binary_op);
            }
            if (node.kind == GraphNodeKind::Constant) {
                out << " value=" << fe_value_to_graph_string(node.constant);
            }
            out << " inputs=";
            append_inputs(out, node.inputs);
            out << '\n';
        }
        out << "  outputs=";
        append_inputs(out, graph.outputs);
        out << '\n';
    }
    for (const auto& skipped : module.skipped) {
        out << "// graph function: " << skipped.function_name << "\n"
            << "// unavailable: " << skipped.reason << '\n';
    }
    return out.str();
}
