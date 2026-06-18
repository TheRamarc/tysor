#include "graph_ir.h"

#include "ops.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

GraphDim GraphDim::unknown() {
    return GraphDim{GraphDimKind::Unknown, 0, std::string{}};
}

GraphDim GraphDim::known(std::int64_t dim) {
    return GraphDim{GraphDimKind::Known, dim, std::string{}};
}

GraphDim GraphDim::symbolic(std::string name) {
    return GraphDim{GraphDimKind::Symbolic, 0, std::move(name)};
}

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

std::string trim_copy(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return std::string{};
    }
    return std::string(first, last);
}

std::vector<std::string> split_shape_parts(const std::string& value) {
    std::vector<std::string> parts;
    int bracket_depth = 0;
    std::size_t start = 0;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const char ch = value[index];
        if (ch == '[' || ch == '(') {
            ++bracket_depth;
        } else if (ch == ']' || ch == ')') {
            --bracket_depth;
        } else if (ch == ',' && bracket_depth == 0) {
            parts.push_back(trim_copy(value.substr(start, index - start)));
            start = index + 1;
        }
    }
    parts.push_back(trim_copy(value.substr(start)));
    return parts;
}

std::optional<std::int64_t> parse_int_literal(const std::string& value) {
    if (value.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        return std::nullopt;
    }
    return static_cast<std::int64_t>(parsed);
}

GraphDim parse_shape_dim(const std::string& value) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty() || trimmed == "?" || trimmed == "_") {
        return GraphDim::unknown();
    }
    if (auto parsed = parse_int_literal(trimmed)) {
        return GraphDim::known(*parsed);
    }
    return GraphDim::symbolic(trimmed);
}

std::optional<std::vector<GraphDim>> parse_shape_expr(const std::string& value) {
    std::string expr = trim_copy(value);
    if (expr.empty()) {
        return std::nullopt;
    }
    if (expr.front() == '[' && expr.back() == ']') {
        expr = expr.substr(1, expr.size() - 2);
    }
    const std::vector<std::string> parts = split_shape_parts(expr);
    if (parts.empty()) {
        return std::nullopt;
    }
    std::vector<GraphDim> dims;
    dims.reserve(parts.size());
    for (const auto& part : parts) {
        dims.push_back(parse_shape_dim(part));
    }
    return dims;
}

std::optional<GraphTensorType> tensor_type_from_fe_type(const FeType& type) {
    if (type.kind != FeTypeKind::Tensor) {
        return std::nullopt;
    }
    GraphTensorType tensor;
    tensor.dtype = type.tensor_dtype;
    if (type.tensor_shape_expr) {
        const std::string shape_expr = trim_copy(*type.tensor_shape_expr);
        const bool looks_like_slice = type.tensor_rank && shape_expr.find(':') != std::string::npos;
        if (!looks_like_slice) {
            if (auto parsed_shape = parse_shape_expr(shape_expr)) {
                tensor.shape = std::move(*parsed_shape);
                tensor.has_known_rank = true;
                return tensor;
            }
        }
        if (type.tensor_rank) {
            tensor.shape.assign(*type.tensor_rank, GraphDim::unknown());
            tensor.has_known_rank = true;
            return tensor;
        }
        if (auto parsed_shape = parse_shape_expr(shape_expr)) {
            tensor.shape = std::move(*parsed_shape);
            tensor.has_known_rank = true;
            return tensor;
        }
    }
    if (type.tensor_rank) {
        tensor.shape.assign(*type.tensor_rank, GraphDim::unknown());
        tensor.has_known_rank = true;
    }
    return tensor;
}

GraphTensorType unknown_tensor(std::optional<std::string> dtype = std::nullopt) {
    GraphTensorType tensor;
    tensor.dtype = std::move(dtype);
    return tensor;
}

std::variant<GraphTensorType, Diagnostic> require_tensor_type(
    const GraphFunction& graph,
    std::size_t value_id,
    const std::string& label
) {
    if (value_id >= graph.values.size()) {
        return graph_error(label + " references missing value " + std::to_string(value_id));
    }
    const GraphValue& value = graph.values[value_id];
    if (!value.tensor_type) {
        return graph_error(label + " must be a tensor value");
    }
    return *value.tensor_type;
}

std::optional<std::int64_t> constant_int_value(const GraphFunction& graph, std::size_t value_id) {
    for (const auto& node : graph.nodes) {
        if (node.output != value_id || node.kind != GraphNodeKind::Constant) {
            continue;
        }
        if (const auto* value = std::get_if<std::int64_t>(&node.constant.value)) {
            return *value;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<bool> constant_bool_value(const GraphFunction& graph, std::size_t value_id) {
    for (const auto& node : graph.nodes) {
        if (node.output != value_id || node.kind != GraphNodeKind::Constant) {
            continue;
        }
        if (const auto* value = std::get_if<bool>(&node.constant.value)) {
            return *value;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

const GraphNode* producer_node(const GraphFunction& graph, std::size_t value_id) {
    auto found = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const GraphNode& node) {
        return node.output == value_id;
    });
    return found == graph.nodes.end() ? nullptr : &*found;
}

bool same_dtype_or_unknown(const GraphTensorType& lhs, const GraphTensorType& rhs) {
    return !lhs.dtype || !rhs.dtype || *lhs.dtype == *rhs.dtype;
}

std::optional<Diagnostic> ensure_compatible_dim(
    const GraphDim& lhs,
    const GraphDim& rhs,
    const std::string& context
) {
    if (lhs.kind == GraphDimKind::Unknown || rhs.kind == GraphDimKind::Unknown) {
        return std::nullopt;
    }
    if (lhs.kind == GraphDimKind::Known && rhs.kind == GraphDimKind::Known && lhs.value != rhs.value) {
        return graph_error(context + " dimension mismatch: " + graph_dim_to_string(lhs) + " vs " + graph_dim_to_string(rhs));
    }
    if (lhs.kind == GraphDimKind::Symbolic && rhs.kind == GraphDimKind::Symbolic && lhs.symbol != rhs.symbol) {
        return graph_error(context + " symbolic dimension mismatch: " + lhs.symbol + " vs " + rhs.symbol);
    }
    return std::nullopt;
}

std::optional<Diagnostic> ensure_same_shape(
    const GraphTensorType& lhs,
    const GraphTensorType& rhs,
    const std::string& context
) {
    if (!same_dtype_or_unknown(lhs, rhs)) {
        return graph_error(context + " dtype mismatch: " + *lhs.dtype + " vs " + *rhs.dtype);
    }
    if (!lhs.has_known_rank || !rhs.has_known_rank) {
        return std::nullopt;
    }
    if (lhs.shape.size() != rhs.shape.size()) {
        return graph_error(
            context + " rank mismatch: " + std::to_string(lhs.shape.size()) +
            " vs " + std::to_string(rhs.shape.size())
        );
    }
    for (std::size_t index = 0; index < lhs.shape.size(); ++index) {
        if (auto diagnostic = ensure_compatible_dim(lhs.shape[index], rhs.shape[index], context)) {
            return diagnostic;
        }
    }
    return std::nullopt;
}

GraphDim multiply_dims(const std::vector<GraphDim>& dims) {
    if (dims.empty()) {
        return GraphDim::known(1);
    }
    std::int64_t product = 1;
    std::vector<std::string> symbolic_parts;
    for (const auto& dim : dims) {
        if (dim.kind == GraphDimKind::Unknown) {
            return GraphDim::unknown();
        }
        if (dim.kind == GraphDimKind::Known) {
            product *= dim.value;
            continue;
        }
        symbolic_parts.push_back(dim.symbol);
    }
    if (symbolic_parts.empty()) {
        return GraphDim::known(product);
    }
    std::ostringstream out;
    if (product != 1) {
        out << product << '*';
    }
    for (std::size_t index = 0; index < symbolic_parts.size(); ++index) {
        if (index != 0) {
            out << '*';
        }
        out << symbolic_parts[index];
    }
    return GraphDim::symbolic(out.str());
}

std::optional<std::int64_t> known_element_count(const GraphTensorType& tensor) {
    if (!tensor.has_known_rank) {
        return std::nullopt;
    }
    std::int64_t product = 1;
    for (const auto& dim : tensor.shape) {
        if (dim.kind != GraphDimKind::Known) {
            return std::nullopt;
        }
        product *= dim.value;
    }
    return product;
}

GraphTensorType tensor_with_shape(const GraphTensorType& source, std::vector<GraphDim> shape) {
    GraphTensorType result;
    result.dtype = source.dtype;
    result.shape = std::move(shape);
    result.has_known_rank = true;
    return result;
}

GraphTensorType merge_tensor_metadata(const GraphTensorType& existing, GraphTensorType inferred) {
    if (!inferred.dtype) {
        inferred.dtype = existing.dtype;
    }
    if (!inferred.has_known_rank && existing.has_known_rank) {
        inferred.shape = existing.shape;
        inferred.has_known_rank = true;
    }
    return inferred;
}

std::size_t append_value(GraphFunction& graph, std::string name, FeType type, bool is_parameter) {
    const std::size_t id = graph.values.size();
    const bool requires_grad = is_parameter && type.kind == FeTypeKind::Tensor;
    const auto tensor_type = tensor_type_from_fe_type(type);
    graph.values.push_back(GraphValue{id, name, std::move(type), is_parameter, requires_grad, tensor_type});
    if (!name.empty()) {
        graph.named_values[name] = id;
    }
    return id;
}

std::string graph_linear_weight_name(std::size_t owner_value) {
    return "linear_" + std::to_string(owner_value) + "_weight";
}

std::string graph_linear_bias_name(std::size_t owner_value) {
    return "linear_" + std::to_string(owner_value) + "_bias";
}

std::string graph_embedding_weight_name(std::size_t owner_value) {
    return "embedding_" + std::to_string(owner_value) + "_weight";
}

bool graph_has_parameter(const GraphFunction& graph, const std::string& name) {
    return std::any_of(graph.parameters.begin(), graph.parameters.end(), [&](const GraphParameter& parameter) {
        return parameter.name == name;
    });
}

void append_graph_parameter(
    GraphFunction& graph,
    std::string name,
    std::string role,
    std::size_t owner_value,
    GraphTensorType tensor_type
) {
    if (graph_has_parameter(graph, name)) {
        return;
    }
    graph.parameters.push_back(GraphParameter{
        std::move(name),
        std::move(role),
        owner_value,
        std::move(tensor_type),
        true,
    });
}

std::optional<GraphDim> last_dim(const GraphTensorType& tensor) {
    if (!tensor.has_known_rank || tensor.shape.empty()) {
        return std::nullopt;
    }
    return tensor.shape.back();
}

GraphTensorType parameter_tensor(
    std::optional<std::string> dtype,
    std::vector<GraphDim> shape
) {
    GraphTensorType tensor;
    tensor.dtype = std::move(dtype);
    tensor.shape = std::move(shape);
    tensor.has_known_rank = true;
    return tensor;
}

std::optional<bool> linear_with_bias(const GraphFunction& graph, const GraphNode& ctor) {
    if (ctor.inputs.size() == 1) {
        return true;
    }
    if (ctor.inputs.size() == 2) {
        if (auto with_bias = constant_bool_value(graph, ctor.inputs[1])) {
            return *with_bias;
        }
        return true;
    }
    if (ctor.inputs.size() >= 3) {
        return constant_bool_value(graph, ctor.inputs[2]).value_or(true);
    }
    return true;
}

std::optional<std::int64_t> linear_in_features(const GraphFunction& graph, const GraphNode& ctor) {
    if (ctor.inputs.size() >= 2 && !constant_bool_value(graph, ctor.inputs[1]).has_value()) {
        return constant_int_value(graph, ctor.inputs[0]);
    }
    return std::nullopt;
}

std::optional<std::int64_t> linear_out_features(const GraphFunction& graph, const GraphNode& ctor) {
    if (ctor.inputs.empty()) {
        return std::nullopt;
    }
    if (ctor.inputs.size() == 1) {
        return constant_int_value(graph, ctor.inputs[0]);
    }
    if (constant_bool_value(graph, ctor.inputs[1]).has_value()) {
        return constant_int_value(graph, ctor.inputs[0]);
    }
    return constant_int_value(graph, ctor.inputs[1]);
}

void append_linear_parameters_for_apply(
    GraphFunction& graph,
    const GraphNode& ctor,
    const GraphTensorType& input
) {
    const auto out_features = linear_out_features(graph, ctor);
    if (!out_features) {
        return;
    }

    GraphDim in_dim = GraphDim::unknown();
    if (auto explicit_in = linear_in_features(graph, ctor)) {
        in_dim = GraphDim::known(*explicit_in);
    } else if (auto inferred_in = last_dim(input)) {
        in_dim = *inferred_in;
    }
    const GraphDim out_dim = GraphDim::known(*out_features);
    append_graph_parameter(
        graph,
        graph_linear_weight_name(ctor.output),
        "weight",
        ctor.output,
        parameter_tensor(input.dtype, {in_dim, out_dim})
    );
    if (linear_with_bias(graph, ctor).value_or(true)) {
        append_graph_parameter(
            graph,
            graph_linear_bias_name(ctor.output),
            "bias",
            ctor.output,
            parameter_tensor(input.dtype, {out_dim})
        );
    }
}

void append_embedding_parameters_for_apply(
    GraphFunction& graph,
    const GraphNode& ctor,
    const GraphTensorType& output
) {
    if (ctor.inputs.size() < 2) {
        return;
    }
    auto num_embeddings = constant_int_value(graph, ctor.inputs[0]);
    auto embedding_dim = constant_int_value(graph, ctor.inputs[1]);
    if (!num_embeddings || !embedding_dim) {
        return;
    }
    append_graph_parameter(
        graph,
        graph_embedding_weight_name(ctor.output),
        "weight",
        ctor.output,
        parameter_tensor(output.dtype, {GraphDim::known(*num_embeddings), GraphDim::known(*embedding_dim)})
    );
}

void append_trainable_parameters_for_node(
    GraphFunction& graph,
    const GraphNode& node,
    const std::optional<GraphTensorType>& inferred
) {
    if (node.kind != GraphNodeKind::Apply || node.inputs.size() < 2 || !inferred) {
        return;
    }
    const GraphNode* ctor = producer_node(graph, node.inputs[0]);
    if (ctor == nullptr || ctor->kind != GraphNodeKind::LibraryCtor) {
        return;
    }
    const auto input = require_tensor_type(graph, node.inputs[1], "apply input");
    if (const auto* tensor = std::get_if<GraphTensorType>(&input)) {
        if (ctor->op == "linear") {
            append_linear_parameters_for_apply(graph, *ctor, *tensor);
        }
    }
    if (ctor->op == "Embedding") {
        append_embedding_parameters_for_apply(graph, *ctor, *inferred);
    }
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_binary_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    const auto& lhs = graph.values[node.inputs[0]].tensor_type;
    const auto& rhs = graph.values[node.inputs[1]].tensor_type;
    if (lhs && rhs) {
        if (auto diagnostic = ensure_same_shape(*lhs, *rhs, "binary op")) {
            return *diagnostic;
        }
        return *lhs;
    }
    if (lhs) {
        return *lhs;
    }
    if (rhs) {
        return *rhs;
    }
    return std::optional<GraphTensorType>{};
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_matmul_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto lhs = require_tensor_type(graph, node.inputs[0], "matmul lhs");
    auto rhs = require_tensor_type(graph, node.inputs[1], "matmul rhs");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) return *diagnostic;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) return *diagnostic;
    const GraphTensorType lhs_tensor = std::get<GraphTensorType>(lhs);
    const GraphTensorType rhs_tensor = std::get<GraphTensorType>(rhs);
    if (!same_dtype_or_unknown(lhs_tensor, rhs_tensor)) {
        return graph_error("matmul dtype mismatch: " + *lhs_tensor.dtype + " vs " + *rhs_tensor.dtype);
    }
    if (lhs_tensor.has_known_rank && lhs_tensor.shape.size() != 2) {
        return graph_error("matmul lhs requires rank-2 tensor when rank is known");
    }
    if (rhs_tensor.has_known_rank && rhs_tensor.shape.size() != 2) {
        return graph_error("matmul rhs requires rank-2 tensor when rank is known");
    }
    if (lhs_tensor.has_known_rank && rhs_tensor.has_known_rank) {
        if (auto diagnostic = ensure_compatible_dim(lhs_tensor.shape[1], rhs_tensor.shape[0], "matmul inner")) {
            return *diagnostic;
        }
        return tensor_with_shape(lhs_tensor, {lhs_tensor.shape[0], rhs_tensor.shape[1]});
    }
    return unknown_tensor(lhs_tensor.dtype ? lhs_tensor.dtype : rhs_tensor.dtype);
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_reshape_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], "reshape input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    const GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    std::vector<GraphDim> shape;
    shape.reserve(node.inputs.size() - 1);
    for (std::size_t index = 1; index < node.inputs.size(); ++index) {
        auto dim = constant_int_value(graph, node.inputs[index]);
        if (!dim) {
            return graph_error("reshape dimensions must be compile-time integer constants in Graph IR");
        }
        shape.push_back(GraphDim::known(*dim));
    }
    const auto input_count = known_element_count(input_tensor);
    const auto output_count = known_element_count(tensor_with_shape(input_tensor, shape));
    if (input_count && output_count && *input_count != *output_count) {
        return graph_error(
            "reshape element count mismatch: " + std::to_string(*input_count) +
            " vs " + std::to_string(*output_count)
        );
    }
    return tensor_with_shape(input_tensor, std::move(shape));
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_transpose_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], "transpose input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    GraphTensorType tensor = std::get<GraphTensorType>(input);
    if (!tensor.has_known_rank) {
        return tensor;
    }
    if (tensor.shape.size() != 2) {
        return graph_error("transpose currently requires rank-2 tensor when rank is known");
    }
    std::swap(tensor.shape[0], tensor.shape[1]);
    return tensor;
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_reduce_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], node.op + " input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    const GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    if (node.inputs.size() == 1) {
        return tensor_with_shape(input_tensor, {GraphDim::known(1)});
    }
    auto axis = constant_int_value(graph, node.inputs[1]);
    if (!axis) {
        return graph_error(node.op + " axis must be a compile-time integer constant in Graph IR");
    }
    if (!input_tensor.has_known_rank) {
        return unknown_tensor(input_tensor.dtype);
    }
    if (*axis < 0 || static_cast<std::size_t>(*axis) >= input_tensor.shape.size()) {
        return graph_error(node.op + " axis " + std::to_string(*axis) + " is out of range for rank " + std::to_string(input_tensor.shape.size()));
    }
    std::vector<GraphDim> shape;
    shape.reserve(input_tensor.shape.size() - 1);
    for (std::size_t index = 0; index < input_tensor.shape.size(); ++index) {
        if (index != static_cast<std::size_t>(*axis)) {
            shape.push_back(input_tensor.shape[index]);
        }
    }
    if (shape.empty()) {
        shape.push_back(GraphDim::known(1));
    }
    return tensor_with_shape(input_tensor, std::move(shape));
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_cross_entropy_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto logits = require_tensor_type(graph, node.inputs[0], "cross_entropy logits");
    auto target = require_tensor_type(graph, node.inputs[1], "cross_entropy target");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&logits)) return *diagnostic;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&target)) return *diagnostic;
    const GraphTensorType logits_tensor = std::get<GraphTensorType>(logits);
    const GraphTensorType target_tensor = std::get<GraphTensorType>(target);
    if (auto diagnostic = ensure_same_shape(logits_tensor, target_tensor, "cross_entropy")) {
        return *diagnostic;
    }
    if (!logits_tensor.has_known_rank) {
        return unknown_tensor(logits_tensor.dtype);
    }
    std::vector<GraphDim> prefix;
    if (logits_tensor.shape.size() > 1) {
        prefix.assign(logits_tensor.shape.begin(), logits_tensor.shape.end() - 1);
    }
    return tensor_with_shape(logits_tensor, {multiply_dims(prefix), GraphDim::known(1)});
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_flatten_heads_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], "flatten_heads input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    const GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    if (!input_tensor.has_known_rank || input_tensor.shape.size() < 3) {
        return input_tensor;
    }
    std::vector<GraphDim> shape(input_tensor.shape.begin(), input_tensor.shape.end() - 2);
    shape.push_back(multiply_dims({input_tensor.shape[input_tensor.shape.size() - 2], input_tensor.shape.back()}));
    return tensor_with_shape(input_tensor, std::move(shape));
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_repeat_kv_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], "repeat_kv input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    GraphTensorType tensor = std::get<GraphTensorType>(input);
    auto repeats = constant_int_value(graph, node.inputs[1]);
    if (!repeats) {
        return graph_error("repeat_kv repeats must be a compile-time integer constant in Graph IR");
    }
    if (*repeats <= 0) {
        return graph_error("repeat_kv repeats must be positive");
    }
    if (!tensor.has_known_rank) {
        return tensor;
    }
    if (tensor.shape.size() < 2) {
        return graph_error("repeat_kv expects rank >= 2 when rank is known");
    }
    GraphDim& heads = tensor.shape[1];
    if (heads.kind == GraphDimKind::Known) {
        heads.value *= *repeats;
    } else if (heads.kind == GraphDimKind::Symbolic) {
        heads.symbol += "*" + std::to_string(*repeats);
    }
    return tensor;
}

std::optional<Diagnostic> validate_last_dim(
    const GraphTensorType& tensor,
    std::int64_t expected,
    const std::string& context
) {
    if (!tensor.has_known_rank || tensor.shape.empty()) {
        return std::nullopt;
    }
    const GraphDim& last = tensor.shape.back();
    if (last.kind == GraphDimKind::Known && last.value != expected) {
        return graph_error(context + " expected last dimension " + std::to_string(expected) + " but got " + std::to_string(last.value));
    }
    return std::nullopt;
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_linear_apply_tensor_type(
    const GraphFunction& graph,
    const GraphNode& ctor,
    const GraphTensorType& input
) {
    if (ctor.inputs.empty()) {
        return graph_error("linear constructor requires out_features for graph shape inference");
    }
    std::optional<std::int64_t> in_features;
    std::optional<std::int64_t> out_features;
    if (ctor.inputs.size() == 1) {
        out_features = constant_int_value(graph, ctor.inputs[0]);
    } else {
        const bool second_is_bias = constant_bool_value(graph, ctor.inputs[1]).has_value();
        if (second_is_bias) {
            out_features = constant_int_value(graph, ctor.inputs[0]);
        } else {
            in_features = constant_int_value(graph, ctor.inputs[0]);
            out_features = constant_int_value(graph, ctor.inputs[1]);
        }
    }
    if (!out_features) {
        return graph_error("linear out_features must be a compile-time integer constant in Graph IR");
    }
    if (in_features) {
        if (auto diagnostic = validate_last_dim(input, *in_features, "linear input")) {
            return *diagnostic;
        }
    }
    if (!input.has_known_rank || input.shape.empty()) {
        return unknown_tensor(input.dtype);
    }
    std::vector<GraphDim> shape = input.shape;
    shape.back() = GraphDim::known(*out_features);
    return tensor_with_shape(input, std::move(shape));
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_embedding_apply_tensor_type(
    const GraphFunction& graph,
    const GraphNode& ctor,
    const GraphTensorType& input
) {
    if (ctor.inputs.size() < 2) {
        return graph_error("Embedding constructor requires num_embeddings and embedding_dim for graph shape inference");
    }
    auto embedding_dim = constant_int_value(graph, ctor.inputs[1]);
    if (!embedding_dim) {
        return graph_error("Embedding embedding_dim must be a compile-time integer constant in Graph IR");
    }
    if (!input.has_known_rank) {
        return unknown_tensor(input.dtype);
    }
    std::vector<GraphDim> shape = input.shape;
    shape.push_back(GraphDim::known(*embedding_dim));
    return tensor_with_shape(input, std::move(shape));
}

bool is_shape_preserving_callable(const std::string& op) {
    return op == "SiLU" || op == "GELU" || op == "Tanh" || op == "Sigmoid" ||
           op == "Softmax" || op == "Dropout" || op == "RMSNorm";
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_apply_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[1], "apply input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    const GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    const GraphNode* ctor = producer_node(graph, node.inputs[0]);
    if (ctor == nullptr || ctor->kind != GraphNodeKind::LibraryCtor) {
        return input_tensor;
    }
    if (ctor->op == "linear") {
        return infer_linear_apply_tensor_type(graph, *ctor, input_tensor);
    }
    if (ctor->op == "Embedding") {
        return infer_embedding_apply_tensor_type(graph, *ctor, input_tensor);
    }
    if (is_shape_preserving_callable(ctor->op)) {
        return input_tensor;
    }
    return input_tensor;
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_library_call_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    if (node.op == "reshape") {
        return infer_reshape_tensor_type(graph, node);
    }
    if (node.op == "transpose") {
        return infer_transpose_tensor_type(graph, node);
    }
    if (node.op == "sum" || node.op == "mean") {
        return infer_reduce_tensor_type(graph, node);
    }
    if (node.op == "cross_entropy") {
        return infer_cross_entropy_tensor_type(graph, node);
    }
    if (node.op == "flatten_heads") {
        return infer_flatten_heads_tensor_type(graph, node);
    }
    if (node.op == "repeat_kv") {
        return infer_repeat_kv_tensor_type(graph, node);
    }
    auto input = require_tensor_type(graph, node.inputs[0], node.op + " input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    if (node.op == "rms_norm" && node.inputs.size() > 1) {
        if (auto hidden = constant_int_value(graph, node.inputs[1])) {
            if (auto diagnostic = validate_last_dim(input_tensor, *hidden, "rms_norm input")) {
                return *diagnostic;
            }
        }
    }
    if (node.op == "rope" && node.inputs.size() > 1) {
        if (auto head_dim = constant_int_value(graph, node.inputs[1])) {
            if (auto diagnostic = validate_last_dim(input_tensor, *head_dim, "rope input")) {
                return *diagnostic;
            }
        }
    }
    return input_tensor;
}

std::variant<std::optional<GraphTensorType>, Diagnostic> infer_node_tensor_type(
    const GraphFunction& graph,
    const GraphNode& node
) {
    if (node.kind == GraphNodeKind::Constant || node.kind == GraphNodeKind::LibraryCtor) {
        return std::optional<GraphTensorType>{};
    }
    if (node.kind == GraphNodeKind::Binary) {
        return infer_binary_tensor_type(graph, node);
    }
    if (node.kind == GraphNodeKind::PrimitiveCall) {
        if (node.op == "matmul") {
            return infer_matmul_tensor_type(graph, node);
        }
        auto input = require_tensor_type(graph, node.inputs[0], node.op + " input");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
        return std::get<GraphTensorType>(input);
    }
    if (node.kind == GraphNodeKind::LibraryCall) {
        return infer_library_call_tensor_type(graph, node);
    }
    if (node.kind == GraphNodeKind::Apply) {
        return infer_apply_tensor_type(graph, node);
    }
    return std::optional<GraphTensorType>{};
}

std::optional<Diagnostic> apply_inferred_tensor_type(
    GraphFunction& graph,
    const GraphNode& node,
    const std::optional<GraphTensorType>& inferred
) {
    if (!inferred) {
        return std::nullopt;
    }
    if (node.output >= graph.values.size()) {
        return graph_error("Node output " + std::to_string(node.output) + " does not reference a value");
    }
    GraphValue& output = graph.values[node.output];
    if (output.tensor_type) {
        if (output.tensor_type->dtype && inferred->dtype && *output.tensor_type->dtype != *inferred->dtype) {
            return graph_error(
                "Inferred dtype for value %" + std::to_string(node.output) +
                " conflicts with frontend dtype"
            );
        }
        output.tensor_type = merge_tensor_metadata(*output.tensor_type, *inferred);
    } else {
        output.tensor_type = *inferred;
    }
    return std::nullopt;
}

std::optional<Diagnostic> append_node(GraphFunction& graph, GraphNode node) {
    auto inferred = infer_node_tensor_type(graph, node);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&inferred)) {
        return *diagnostic;
    }
    if (auto diagnostic = apply_inferred_tensor_type(graph, node, std::get<std::optional<GraphTensorType>>(inferred))) {
        return diagnostic;
    }
    append_trainable_parameters_for_node(graph, node, std::get<std::optional<GraphTensorType>>(inferred));
    graph.nodes.push_back(std::move(node));
    return std::nullopt;
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
        GraphNode node{
            GraphNodeKind::Constant,
            output,
            std::string{},
            std::nullopt,
            FeBinaryOp::Add,
            constant->value,
            {},
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
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
        GraphNode node{
            GraphNodeKind::Binary,
            output,
            std::string{},
            std::nullopt,
            binary->op,
            FeValue::none(),
            {std::get<std::size_t>(lhs), std::get<std::size_t>(rhs)},
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
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
        GraphNode node{
            is_primitive_tensor_op(call->callee) ? GraphNodeKind::PrimitiveCall : GraphNodeKind::LibraryCall,
            output,
            call->callee,
            graph_op_id_name(call->callee),
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
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
        GraphNode node{
            GraphNodeKind::LibraryCtor,
            output,
            ctor->callee,
            graph_op_id_name(ctor->callee),
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
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
        GraphNode node{
            GraphNodeKind::Apply,
            output,
            std::string{},
            std::nullopt,
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
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

std::string graph_dim_to_string(const GraphDim& dim) {
    switch (dim.kind) {
        case GraphDimKind::Unknown:
            return "?";
        case GraphDimKind::Known:
            return std::to_string(dim.value);
        case GraphDimKind::Symbolic:
            return dim.symbol;
    }
    return "?";
}

std::string graph_tensor_type_to_string(const GraphTensorType& type) {
    std::ostringstream out;
    out << type.dtype.value_or("tensor");
    if (!type.has_known_rank) {
        out << "[?]";
        return out.str();
    }
    out << '[';
    for (std::size_t index = 0; index < type.shape.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << graph_dim_to_string(type.shape[index]);
    }
    out << ']';
    return out.str();
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
    std::set<std::string> parameter_names;
    for (const auto& parameter : graph.parameters) {
        if (parameter.name.empty()) {
            return graph_error("Graph '" + graph.name + "' has an unnamed parameter");
        }
        if (parameter.role.empty()) {
            return graph_error("Graph '" + graph.name + "' parameter '" + parameter.name + "' has an empty role");
        }
        if (value_ids.find(parameter.owner_value) == value_ids.end()) {
            return graph_error(
                "Graph '" + graph.name + "' parameter '" + parameter.name +
                "' references missing owner value " + std::to_string(parameter.owner_value)
            );
        }
        if (!parameter_names.insert(parameter.name).second) {
            return graph_error("Graph '" + graph.name + "' has duplicate parameter '" + parameter.name + "'");
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
            << " parameters=" << graph.parameters.size()
            << " nodes=" << graph.nodes.size()
            << " outputs=" << graph.outputs.size()
            << " -> " << fe_type_to_graph_string(graph.return_type) << '\n';
        for (const auto& value : graph.values) {
            out << "  %" << value.id;
            if (!value.name.empty()) {
                out << ' ' << value.name;
            }
            out << ": " << fe_type_to_graph_string(value.type);
            if (value.tensor_type) {
                out << " tensor=" << graph_tensor_type_to_string(*value.tensor_type);
            }
            if (value.is_parameter) {
                out << " param";
            }
            if (value.requires_grad) {
                out << " requires_grad";
            }
            out << '\n';
        }
        for (const auto& parameter : graph.parameters) {
            out << "  param " << parameter.name
                << " role=" << parameter.role
                << " owner=%" << parameter.owner_value
                << " tensor=" << graph_tensor_type_to_string(parameter.tensor_type);
            if (parameter.trainable) {
                out << " trainable";
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
