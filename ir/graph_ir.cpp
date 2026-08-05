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
    return Diagnostic::error(DiagnosticCode::GraphIrError, std::move(message))
        .withHelp("Graph IR validation failed before backend planning. This usually means frontend lowering produced an invalid graph boundary.");
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
            return type.scalarDtype.value_or("int");
        case FeTypeKind::Float:
            return type.scalarDtype.value_or("float");
        case FeTypeKind::Bool:
            return "bool";
        case FeTypeKind::Str:
            return "str";
        case FeTypeKind::Tensor:
            if (type.tensorDtype && type.tensorShapeExpr) {
                return "tensor[" + *type.tensorDtype + ", " + *type.tensorShapeExpr + "]";
            }
            if (type.tensorDtype) {
                return "tensor[" + *type.tensorDtype + "]";
            }
            if (type.tensorShapeExpr) {
                return "tensor[" + *type.tensorShapeExpr + "]";
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
            return "callable -> " + (type.callableReturn ? fe_type_to_graph_string(*type.callableReturn) : "void");
        case FeTypeKind::None:
            return "None";
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
    tensor.dtype = type.tensorDtype;
    if (type.tensorShapeExpr) {
        const std::string shape_expr = trim_copy(*type.tensorShapeExpr);
        const bool looks_like_slice = type.tensorRank && shape_expr.find(':') != std::string::npos;
        if (!looks_like_slice) {
            if (auto parsed_shape = parse_shape_expr(shape_expr)) {
                tensor.shape = std::move(*parsed_shape);
                tensor.hasKnownRank = true;
                return tensor;
            }
        }
        if (type.tensorRank) {
            tensor.shape.assign(*type.tensorRank, GraphDim::unknown());
            tensor.hasKnownRank = true;
            return tensor;
        }
        if (auto parsed_shape = parse_shape_expr(shape_expr)) {
            tensor.shape = std::move(*parsed_shape);
            tensor.hasKnownRank = true;
            return tensor;
        }
    }
    if (type.tensorRank) {
        tensor.shape.assign(*type.tensorRank, GraphDim::unknown());
        tensor.hasKnownRank = true;
    }
    return tensor;
}

GraphTensorType unknown_tensor(std::optional<std::string> dtype = std::nullopt) {
    GraphTensorType tensor;
    tensor.dtype = std::move(dtype);
    return tensor;
}

template <typename GraphT>
std::variant<GraphTensorType, Diagnostic> require_tensor_type(
    const GraphT& graph,
    std::size_t valueId,
    const std::string& label
) {
    if (valueId >= graph.values.size()) {
        return graph_error(label + " references missing value " + std::to_string(valueId));
    }
    const GraphValue& value = graph.values[valueId];
    if (!value.tensorType) {
        return graph_error(label + " must be a tensor value");
    }
    return *value.tensorType;
}

template <typename GraphT>
std::optional<std::int64_t> constant_int_value(const GraphT& graph, std::size_t valueId) {
    for (const auto& node : graph.nodes) {
        if (node.output != valueId || node.kind != GraphNodeKind::Constant) {
            continue;
        }
        if (const auto* value = std::get_if<std::int64_t>(&node.constant.value)) {
            return *value;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

template <typename GraphT>
std::optional<bool> constant_bool_value(const GraphT& graph, std::size_t valueId) {
    for (const auto& node : graph.nodes) {
        if (node.output != valueId || node.kind != GraphNodeKind::Constant) {
            continue;
        }
        if (const auto* value = std::get_if<bool>(&node.constant.value)) {
            return *value;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

template <typename GraphT>
const GraphNode* producer_node(const GraphT& graph, std::size_t valueId) {
    auto found = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const GraphNode& node) {
        return node.output == valueId;
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
        return graph_error(context + " dimension mismatch: " + graphDimToString(lhs) + " vs " + graphDimToString(rhs));
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
    // Validates that two tensor types have the exact same shape and compatible
    // dtypes. Returns an error diagnostic if there is a mismatch.
    if (!same_dtype_or_unknown(lhs, rhs)) {
        return graph_error(context + " dtype mismatch: " + *lhs.dtype + " vs " + *rhs.dtype);
    }
    if (!lhs.hasKnownRank || !rhs.hasKnownRank) {
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

std::optional<Diagnostic> ensure_trailing_vector_broadcast(
    const GraphTensorType& tensor,
    const GraphTensorType& vector,
    const std::string& context
) {
    if (!same_dtype_or_unknown(tensor, vector)) {
        return graph_error(context + " dtype mismatch: " + *tensor.dtype + " vs " + *vector.dtype);
    }
    if (!tensor.hasKnownRank || !vector.hasKnownRank) {
        return std::nullopt;
    }
    if (tensor.shape.size() != 2 || vector.shape.size() != 1) {
        return graph_error(
            context + " rank mismatch: " + std::to_string(tensor.shape.size()) +
            " vs " + std::to_string(vector.shape.size())
        );
    }
    return ensure_compatible_dim(tensor.shape.back(), vector.shape.front(), context + " broadcast");
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
    if (!tensor.hasKnownRank) {
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

bool has_known_dim(const GraphTensorType& tensor, std::size_t index) {
    return tensor.hasKnownRank &&
           index < tensor.shape.size() &&
           tensor.shape[index].kind == GraphDimKind::Known;
}

GraphTensorType tensor_with_shape(const GraphTensorType& source, std::vector<GraphDim> shape) {
    GraphTensorType result;
    result.dtype = source.dtype;
    result.shape = std::move(shape);
    result.hasKnownRank = true;
    return result;
}

GraphTensorType merge_tensor_metadata(const GraphTensorType& existing, GraphTensorType inferred) {
    if (!inferred.dtype) {
        inferred.dtype = existing.dtype;
    }
    if (!inferred.hasKnownRank && existing.hasKnownRank) {
        inferred.shape = existing.shape;
        inferred.hasKnownRank = true;
    }
    return inferred;
}

template <typename GraphT>
std::size_t append_value(GraphT& graph, std::string name, FeType type, bool isParameter) {
    const std::size_t id = graph.values.size();
    const bool requiresGrad = isParameter && type.kind == FeTypeKind::Tensor;
    const auto tensorType = tensor_type_from_fe_type(type);
    graph.values.push_back(GraphValue{id, name, std::move(type), isParameter, requiresGrad, false, tensorType});
    if (!name.empty()) {
        graph.namedValues[name] = id;
    }
    return id;
}

std::string shape_expr_from_tensor_type(const GraphTensorType& tensor) {
    if (!tensor.hasKnownRank) {
        return std::string{};
    }
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < tensor.shape.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << graphDimToString(tensor.shape[index]);
    }
    out << ']';
    return out.str();
}

FeType fe_tensor_type_from_graph_tensor(const GraphTensorType& tensor) {
    std::optional<std::string> shape_expr;
    if (tensor.hasKnownRank) {
        shape_expr = shape_expr_from_tensor_type(tensor);
    }
    return FeType::tensor(tensor.dtype, shape_expr, tensor.hasKnownRank ? std::optional<std::size_t>{tensor.shape.size()} : std::nullopt);
}

template <typename GraphT>
std::size_t append_tensor_value(
    GraphT& graph,
    std::string name,
    GraphTensorType tensorType,
    bool isModelParameter
) {
    const std::size_t id = graph.values.size();
    FeType type = fe_tensor_type_from_graph_tensor(tensorType);
    graph.values.push_back(GraphValue{id, name, std::move(type), false, isModelParameter, isModelParameter, tensorType});
    if (!name.empty()) {
        graph.namedValues[name] = id;
    }
    return id;
}

std::string graph_linear_weight_name(std::size_t ownerValue) {
    return "linear_" + std::to_string(ownerValue) + "_weight";
}

std::string graph_linear_bias_name(std::size_t ownerValue) {
    return "linear_" + std::to_string(ownerValue) + "_bias";
}

std::string graph_embedding_weight_name(std::size_t ownerValue) {
    return "embedding_" + std::to_string(ownerValue) + "_weight";
}

template <typename GraphT>
bool graph_has_parameter(const GraphT& graph, const std::string& name) {
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        return std::any_of(graph.parameters.begin(), graph.parameters.end(), [&](const GraphParameter& parameter) {
            return parameter.name == name;
        });
    } else {
        return false;
    }
}

template <typename GraphT>
const GraphParameter* find_graph_parameter(
    const GraphT& graph,
    std::size_t ownerValue,
    const std::string& role
) {
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        auto found = std::find_if(graph.parameters.begin(), graph.parameters.end(), [&](const GraphParameter& parameter) {
            return parameter.ownerValue == ownerValue && parameter.role == role;
        });
        return found == graph.parameters.end() ? nullptr : &*found;
    } else {
        return nullptr;
    }
}

template <typename GraphT>
void append_graph_parameter(
    GraphT& graph,
    std::string name,
    std::string role,
    std::size_t ownerValue,
    std::size_t valueId,
    GraphTensorType tensorType
) {
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        if (graph_has_parameter(graph, name)) {
            return;
        }
        graph.parameters.push_back(GraphParameter{
            std::move(name),
            std::move(role),
            ownerValue,
            valueId,
            std::move(tensorType),
            true
        });
    }
}

std::optional<GraphDim> last_dim(const GraphTensorType& tensor) {
    if (!tensor.hasKnownRank || tensor.shape.empty()) {
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
    tensor.hasKnownRank = true;
    return tensor;
}

template <typename GraphT>
std::optional<bool> linear_with_bias(const GraphT& graph, const GraphNode& ctor) {
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

template <typename GraphT>
std::optional<std::int64_t> linear_in_features(const GraphT& graph, const GraphNode& ctor) {
    if (ctor.inputs.size() >= 2 && !constant_bool_value(graph, ctor.inputs[1]).has_value()) {
        return constant_int_value(graph, ctor.inputs[0]);
    }
    return std::nullopt;
}

template <typename GraphT>
std::optional<std::int64_t> linear_out_features(const GraphT& graph, const GraphNode& ctor) {
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

template <typename GraphT>
void append_linear_parameters_for_apply(
    GraphT& graph,
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
    const GraphTensorType weight_type = parameter_tensor(input.dtype, {in_dim, out_dim});
    const std::string weight_name = graph_linear_weight_name(ctor.output);
    if (find_graph_parameter(graph, ctor.output, "weight") == nullptr) {
        const std::size_t weight_value = append_tensor_value(graph, weight_name, weight_type, true);
        append_graph_parameter(
            graph,
            weight_name,
            "weight",
            ctor.output,
            weight_value,
            weight_type
        );
    }
    if (linear_with_bias(graph, ctor).value_or(true)) {
        const GraphTensorType bias_type = parameter_tensor(input.dtype, {out_dim});
        const std::string bias_name = graph_linear_bias_name(ctor.output);
        if (find_graph_parameter(graph, ctor.output, "bias") == nullptr) {
            const std::size_t bias_value = append_tensor_value(graph, bias_name, bias_type, true);
            append_graph_parameter(
                graph,
                bias_name,
                "bias",
                ctor.output,
                bias_value,
                bias_type
            );
        }
    }
}

template <typename GraphT>
void append_embedding_parameters_for_apply(
    GraphT& graph,
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
    const GraphTensorType weight_type = parameter_tensor(
        output.dtype,
        {GraphDim::known(*num_embeddings), GraphDim::known(*embedding_dim)}
    );
    const std::string weight_name = graph_embedding_weight_name(ctor.output);
    if (find_graph_parameter(graph, ctor.output, "weight") == nullptr) {
        const std::size_t weight_value = append_tensor_value(graph, weight_name, weight_type, true);
        append_graph_parameter(
            graph,
            weight_name,
            "weight",
            ctor.output,
            weight_value,
            weight_type
        );
    }
}

template <typename GraphT>
void append_trainable_parameters_for_node(
    GraphT& graph,
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_binary_tensor_type(
    const GraphT& graph,
    const GraphNode& node
) {
    const auto& lhs = graph.values[node.inputs[0]].tensorType;
    const auto& rhs = graph.values[node.inputs[1]].tensorType;
    if (lhs && rhs) {
        const bool supports_trailing_broadcast = node.binaryOp == FeBinaryOp::Add ||
                                                 node.binaryOp == FeBinaryOp::Sub ||
                                                 node.binaryOp == FeBinaryOp::Mul;
        if (supports_trailing_broadcast &&
            lhs->hasKnownRank && rhs->hasKnownRank && lhs->shape.size() == 2 && rhs->shape.size() == 1) {
            if (auto diagnostic = ensure_trailing_vector_broadcast(*lhs, *rhs, "binary op")) {
                return *diagnostic;
            }
            return *lhs;
        }
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_matmul_tensor_type(
    const GraphT& graph,
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
    if (lhs_tensor.hasKnownRank && lhs_tensor.shape.size() != 2) {
        return graph_error("matmul lhs requires rank-2 tensor when rank is known");
    }
    if (rhs_tensor.hasKnownRank && rhs_tensor.shape.size() != 2) {
        return graph_error("matmul rhs requires rank-2 tensor when rank is known");
    }
    if (lhs_tensor.hasKnownRank && rhs_tensor.hasKnownRank) {
        if (auto diagnostic = ensure_compatible_dim(lhs_tensor.shape[1], rhs_tensor.shape[0], "matmul inner")) {
            return *diagnostic;
        }
        return tensor_with_shape(lhs_tensor, {lhs_tensor.shape[0], rhs_tensor.shape[1]});
    }
    return unknown_tensor(lhs_tensor.dtype ? lhs_tensor.dtype : rhs_tensor.dtype);
}

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_reshape_tensor_type(
    const GraphT& graph,
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_transpose_tensor_type(
    const GraphT& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], "transpose input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    GraphTensorType tensor = std::get<GraphTensorType>(input);
    if (!tensor.hasKnownRank) {
        return tensor;
    }
    if (tensor.shape.size() != 2) {
        return graph_error("transpose currently requires rank-2 tensor when rank is known");
    }
    std::swap(tensor.shape[0], tensor.shape[1]);
    return tensor;
}

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_reduce_tensor_type(
    const GraphT& graph,
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
    if (!input_tensor.hasKnownRank) {
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_cross_entropy_tensor_type(
    const GraphT& graph,
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
    if (!logits_tensor.hasKnownRank) {
        return unknown_tensor(logits_tensor.dtype);
    }
    std::vector<GraphDim> prefix;
    if (logits_tensor.shape.size() > 1) {
        prefix.assign(logits_tensor.shape.begin(), logits_tensor.shape.end() - 1);
    }
    return tensor_with_shape(logits_tensor, {multiply_dims(prefix), GraphDim::known(1)});
}

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_flatten_heads_tensor_type(
    const GraphT& graph,
    const GraphNode& node
) {
    auto input = require_tensor_type(graph, node.inputs[0], "flatten_heads input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) return *diagnostic;
    const GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    if (!input_tensor.hasKnownRank || input_tensor.shape.size() < 3) {
        return input_tensor;
    }
    std::vector<GraphDim> shape(input_tensor.shape.begin(), input_tensor.shape.end() - 2);
    shape.push_back(multiply_dims({input_tensor.shape[input_tensor.shape.size() - 2], input_tensor.shape.back()}));
    return tensor_with_shape(input_tensor, std::move(shape));
}

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_repeat_kv_tensor_type(
    const GraphT& graph,
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
    if (!tensor.hasKnownRank) {
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
    if (!tensor.hasKnownRank || tensor.shape.empty()) {
        return std::nullopt;
    }
    const GraphDim& last = tensor.shape.back();
    if (last.kind == GraphDimKind::Known && last.value != expected) {
        return graph_error(context + " expected last dimension " + std::to_string(expected) + " but got " + std::to_string(last.value));
    }
    return std::nullopt;
}

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_linear_apply_tensor_type(
    const GraphT& graph,
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
    if (!input.hasKnownRank || input.shape.empty()) {
        return unknown_tensor(input.dtype);
    }
    std::vector<GraphDim> shape = input.shape;
    shape.back() = GraphDim::known(*out_features);
    return tensor_with_shape(input, std::move(shape));
}

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_embedding_apply_tensor_type(
    const GraphT& graph,
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
    if (!input.hasKnownRank) {
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_apply_tensor_type(
    const GraphT& graph,
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_library_call_tensor_type(
    const GraphT& graph,
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

template <typename GraphT>
std::variant<std::optional<GraphTensorType>, Diagnostic> infer_node_tensor_type(
    const GraphT& graph,
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
        if (node.op == "print" || (node.opId && *node.opId == "Print")) {
            return std::optional<GraphTensorType>{};
        }
        if (node.inputs.empty()) {
            return std::optional<GraphTensorType>{};
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

template <typename GraphT>
std::optional<Diagnostic> apply_inferred_tensor_type(
    GraphT& graph,
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
    if (output.tensorType) {
        if (output.tensorType->dtype && inferred->dtype && *output.tensorType->dtype != *inferred->dtype) {
            return graph_error(
                "Inferred dtype for value %" + std::to_string(node.output) +
                " conflicts with frontend dtype"
            );
        }
        output.tensorType = merge_tensor_metadata(*output.tensorType, *inferred);
    } else {
        output.tensorType = *inferred;
    }
    return std::nullopt;
}

std::optional<std::string> graph_op_id_name(const std::string& name);

template <typename GraphT>
std::optional<Diagnostic> append_node(GraphT& graph, GraphNode node) {
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

template <typename GraphT>
std::variant<std::optional<std::size_t>, Diagnostic> try_lower_linear_apply(
    GraphT& graph,
    std::size_t callee_id,
    std::size_t input_id,
    const FeType& output_type
) {
    const GraphNode* ctor = producer_node(graph, callee_id);
    if (ctor == nullptr || ctor->kind != GraphNodeKind::LibraryCtor || ctor->op != "linear") {
        return std::optional<std::size_t>{};
    }

    auto input = require_tensor_type(graph, input_id, "linear input");
    if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
        return *diagnostic;
    }
    const GraphTensorType input_tensor = std::get<GraphTensorType>(input);
    auto inferred = infer_linear_apply_tensor_type(graph, *ctor, input_tensor);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&inferred)) {
        return *diagnostic;
    }
    const auto inferred_tensor = std::get<std::optional<GraphTensorType>>(inferred);
    if (!inferred_tensor) {
        return graph_error("linear apply did not infer an output tensor type");
    }

    append_linear_parameters_for_apply(graph, *ctor, input_tensor);
    const GraphParameter* weight = find_graph_parameter(graph, ctor->output, "weight");
    if (weight == nullptr) {
        return graph_error("linear apply could not materialize a weight parameter");
    }
    if (!has_known_dim(weight->tensorType, 0) || !has_known_dim(weight->tensorType, 1)) {
        return std::optional<std::size_t>{};
    }

    const bool with_bias = linear_with_bias(graph, *ctor).value_or(true);
    const GraphParameter* bias = nullptr;
    if (with_bias) {
        bias = find_graph_parameter(graph, ctor->output, "bias");
        if (bias == nullptr) {
            return graph_error("linear apply could not materialize a bias parameter");
        }
        if (!has_known_dim(bias->tensorType, 0)) {
            return std::optional<std::size_t>{};
        }
    }

    const std::size_t matmul_output = with_bias
        ? append_tensor_value(graph, std::string{}, *inferred_tensor, false)
        : append_value(graph, std::string{}, output_type, false);
    GraphNode matmul_node{
        GraphNodeKind::PrimitiveCall,
        matmul_output,
        "matmul",
        graph_op_id_name("matmul"),
        FeBinaryOp::Add,
        FeValue::none(),
        {input_id, weight->valueId},
    };
    if (auto diagnostic = append_node(graph, std::move(matmul_node))) {
        return *diagnostic;
    }
    if (!with_bias) {
        return std::optional<std::size_t>{matmul_output};
    }

    const std::size_t output = append_value(graph, std::string{}, output_type, false);
    GraphNode bias_node{
        GraphNodeKind::Binary,
        output,
        std::string{},
        std::nullopt,
        FeBinaryOp::Add,
        FeValue::none(),
        {matmul_output, bias->valueId},
    };
    if (auto diagnostic = append_node(graph, std::move(bias_node))) {
        return *diagnostic;
    }
    return std::optional<std::size_t>{output};
}

std::optional<std::string> graph_op_id_name(const std::string& name) {
    auto id = lookupOpId(name);
    if (!id) {
        return std::nullopt;
    }
    return std::string(opIdName(*id));
}

bool is_frontend_only_expr(const FeExprPtr& expr) {
    return expr && std::holds_alternative<FeListExpr>(expr->kind);
}

template <typename GraphT>
std::optional<Diagnostic> validate_node_shape(
    const GraphT& graph,
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

template <typename FeCallableT, typename GraphT>
class GraphBuilder {
public:
    explicit GraphBuilder(const FeCallableT& callable) : callable_(callable) {}

    std::variant<GraphT, Diagnostic> build() {
        GraphT graph;
        graph.name = callable_.name;
        graph.returnType = callable_.returnType;

        for (const auto& param : callable_.params) {
            append_value(graph, param.first, param.second, true);
            graph.inputs.push_back(graph.values.back().id);
        }

        for (const auto& stmt : callable_.body) {
            if (const auto* decl = std::get_if<FeVarDeclStmt>(&stmt.kind)) {
                if (is_frontend_only_expr(decl->value)) {
                    continue;
                }
                if (!decl->hasValue || !decl->value) {
                    return graph_error("Graph builder expected value");
                }
                auto valueId = lower_expr(decl->value, graph);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&valueId)) {
                    return *diagnostic;
                }
                graph.namedValues[decl->name] = std::get<std::size_t>(valueId);
                graph.values[std::get<std::size_t>(valueId)].name = decl->name;
            } else if (const auto* assign = std::get_if<FeAssignStmt>(&stmt.kind)) {
                if (is_frontend_only_expr(assign->value)) {
                    continue;
                }
                auto valueId = lower_expr(assign->value, graph);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&valueId)) {
                    return *diagnostic;
                }
                graph.namedValues[assign->name] = std::get<std::size_t>(valueId);
                graph.values[std::get<std::size_t>(valueId)].name = assign->name;
            } else if (const auto* ret = std::get_if<FeReturnStmt>(&stmt.kind)) {
                auto output_id = lower_expr(ret->value, graph);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&output_id)) {
                    return *diagnostic;
                }
                graph.outputs.push_back(std::get<std::size_t>(output_id));
            } else if (const auto* expr_stmt = std::get_if<FeExprStmt>(&stmt.kind)) {
                if (is_frontend_only_expr(expr_stmt->value)) {
                    continue;
                }
                auto valueId = lower_expr(expr_stmt->value, graph);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&valueId)) {
                    return *diagnostic;
                }
            } else {
                return graph_error("Graph builder currently supports straight-line statements only");
            }
        }

        if constexpr (std::is_same_v<GraphT, GraphLayer>) {
            if (auto diagnostic = validateGraphLayer(graph)) {
                return *diagnostic;
            }
        } else {
            if (auto diagnostic = validateGraphFunction(graph)) {
                return *diagnostic;
            }
        }
        return graph;
    }

private:
    const FeCallableT& callable_;

    std::variant<std::size_t, Diagnostic> lowerConstantExpr(const FeConstantExpr& constant, const FeType& type, GraphT& graph) const {
        const std::size_t output = append_value(graph, std::string{}, type, false);
        GraphNode node{
            GraphNodeKind::Constant,
            output,
            std::string{},
            std::nullopt,
            FeBinaryOp::Add,
            constant.value,
            {},
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
        return output;
    }

    std::variant<std::size_t, Diagnostic> lowerVarExpr(const FeVarExpr& var, GraphT& graph) const {
        return lookup_named_value(var.symbol, graph);
    }

    std::variant<std::size_t, Diagnostic> lowerBinaryExpr(const FeBinaryExpr& binary, const FeType& type, GraphT& graph) const {
        auto lhs = lower_expr(binary.lhs, graph);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
            return *diagnostic;
        }
        auto rhs = lower_expr(binary.rhs, graph);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
            return *diagnostic;
        }
        const std::size_t output = append_value(graph, std::string{}, type, false);
        GraphNode node{
            GraphNodeKind::Binary,
            output,
            std::string{},
            std::nullopt,
            binary.op,
            FeValue::none(),
            {std::get<std::size_t>(lhs), std::get<std::size_t>(rhs)},
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
        return output;
    }

    std::variant<std::size_t, Diagnostic> lowerCallExpr(const FeCallExpr& call, const FeType& type, GraphT& graph) const {
        std::vector<std::size_t> inputs;
        for (const auto& arg : call.args) {
            auto lowered = lower_expr(arg.value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            inputs.push_back(std::get<std::size_t>(lowered));
        }
        const std::size_t output = append_value(graph, std::string{}, type, false);
        const bool is_ctor = type.kind == FeTypeKind::Callable || isCallableLibraryOp(call.callee);
        GraphNodeKind kind = is_ctor
            ? GraphNodeKind::LibraryCtor
            : (isPrimitiveTensorOp(call.callee) ? GraphNodeKind::PrimitiveCall : GraphNodeKind::LibraryCall);
        GraphNode node{
            kind,
            output,
            call.callee,
            graph_op_id_name(call.callee),
            FeBinaryOp::Add,
            FeValue::none(),
            std::move(inputs),
        };
        if (auto diagnostic = append_node(graph, std::move(node))) {
            return *diagnostic;
        }
        return output;
    }

    std::variant<std::size_t, Diagnostic> lowerApplyExpr(const FeApplyExpr& apply, const FeType& type, GraphT& graph) const {
        std::vector<std::size_t> inputs;
        auto callee = lower_expr(apply.callee, graph);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&callee)) {
            return *diagnostic;
        }
        inputs.push_back(std::get<std::size_t>(callee));
        for (const auto& arg : apply.args) {
            auto lowered = lower_expr(arg.value, graph);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            inputs.push_back(std::get<std::size_t>(lowered));
        }
        if (inputs.size() == 2) {
            auto lowered_linear = try_lower_linear_apply(graph, inputs[0], inputs[1], type);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_linear)) {
                return *diagnostic;
            }
            if (const auto lowered_output = std::get<std::optional<std::size_t>>(lowered_linear)) {
                return *lowered_output;
            }
        }
        const std::size_t output = append_value(graph, std::string{}, type, false);
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

    std::variant<std::size_t, Diagnostic> lower_expr(const FeExprPtr& expr, GraphT& graph) const {
        if (!expr) {
            return graph_error("Graph builder expected expression");
        }
        if (const auto* constant = std::get_if<FeConstantExpr>(&expr->kind)) {
            return lowerConstantExpr(*constant, expr->type, graph);
        }
        if (const auto* var = std::get_if<FeVarExpr>(&expr->kind)) {
            return lowerVarExpr(*var, graph);
        }
        if (const auto* binary = std::get_if<FeBinaryExpr>(&expr->kind)) {
            return lowerBinaryExpr(*binary, expr->type, graph);
        }
        if (const auto* call = std::get_if<FeCallExpr>(&expr->kind)) {
            return lowerCallExpr(*call, expr->type, graph);
        }
        if (const auto* apply = std::get_if<FeApplyExpr>(&expr->kind)) {
            return lowerApplyExpr(*apply, expr->type, graph);
        }
        if (std::holds_alternative<FeListExpr>(expr->kind)) {
            return graph_error("Graph builder does not support list expressions yet");
        }
        return graph_error("Graph builder does not support this FE expression kind yet");
    }

    std::variant<std::size_t, Diagnostic> lookup_named_value(const std::string& name, const GraphT& graph) const {
        auto found = graph.namedValues.find(name);
        if (found == graph.namedValues.end()) {
            return graph_error("Graph builder could not resolve symbol '" + name + "'");
        }
        return found->second;
    }
};

GraphFunctionResult buildGraphFunction(const FeFunction& function) {
    return GraphBuilder<FeFunction, GraphFunction>(function).build();
}

GraphLayerResult buildGraphLayer(const FeLayer& layer) {
    return GraphBuilder<FeLayer, GraphLayer>(layer).build();
}

std::variant<GraphModule, Diagnostic> buildGraphModule(const LoweredModule& module) {
    GraphModule graph_module;
    for (const auto& layer : module.layers) {
        auto graph = buildGraphLayer(layer);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&graph)) {
            graph_module.skipped.push_back(GraphBuildSkipped{layer.name, diagnostic->message});
            continue;
        }
        graph_module.layers.push_back(std::get<GraphLayer>(std::move(graph)));
    }
    for (const auto& function : module.functions) {
        auto graph = buildGraphFunction(function);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&graph)) {
            graph_module.skipped.push_back(GraphBuildSkipped{function.name, diagnostic->message});
            continue;
        }
        graph_module.functions.push_back(std::get<GraphFunction>(std::move(graph)));
    }
    return graph_module;
}

std::string graphDimToString(const GraphDim& dim) {
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

std::string graphTensorTypeToString(const GraphTensorType& type) {
    std::ostringstream out;
    out << type.dtype.value_or("tensor");
    if (!type.hasKnownRank) {
        out << "[?]";
        return out.str();
    }
    out << '[';
    for (std::size_t index = 0; index < type.shape.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << graphDimToString(type.shape[index]);
    }
    out << ']';
    return out.str();
}

template <typename GraphT>
std::optional<Diagnostic> validate_graph_impl(const GraphT& graph) {
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
    for (const auto& named : graph.namedValues) {
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
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        std::set<std::string> parameter_names;
        for (const auto& parameter : graph.parameters) {
            if (parameter.name.empty()) {
                return graph_error("Graph '" + graph.name + "' has an unnamed parameter");
            }
            if (parameter.role.empty()) {
                return graph_error("Graph '" + graph.name + "' parameter '" + parameter.name + "' has an empty role");
            }
            if (value_ids.find(parameter.ownerValue) == value_ids.end()) {
                return graph_error(
                    "Graph '" + graph.name + "' parameter '" + parameter.name +
                    "' references missing owner value " + std::to_string(parameter.ownerValue)
                );
            }
            if (value_ids.find(parameter.valueId) == value_ids.end()) {
                return graph_error(
                    "Graph '" + graph.name + "' parameter '" + parameter.name +
                    "' references missing value " + std::to_string(parameter.valueId)
                );
            }
            if (!parameter_names.insert(parameter.name).second) {
                return graph_error("Graph '" + graph.name + "' has duplicate parameter '" + parameter.name + "'");
            }
        }
    }

    std::set<std::size_t> produced_values;
    for (const auto& value : graph.values) {
        if (value.isParameter || value.isModelParameter) {
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

std::optional<Diagnostic> validateGraphFunction(const GraphFunction& graph) {
    return validate_graph_impl(graph);
}

std::optional<Diagnostic> validateGraphLayer(const GraphLayer& graph) {
    return validate_graph_impl(graph);
}

std::string graphModuleSummary(const GraphModule& module) {
    std::ostringstream out;
    out << "graph=layers:" << module.layers.size() << " functions:" << module.functions.size() << " skipped:" << module.skipped.size();
    return out.str();
}

template <typename GraphT>
void format_graph_body(std::ostringstream& out, const GraphT& graph, const std::string& prefix) {
    out << prefix << graph.name
        << " values=" << graph.values.size();
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        out << " parameters=" << graph.parameters.size();
    }
    out << " nodes=" << graph.nodes.size()
        << " outputs=" << graph.outputs.size()
        << " -> " << fe_type_to_graph_string(graph.returnType) << '\n';
    for (const auto& value : graph.values) {
        out << "  %" << value.id;
        if (!value.name.empty()) {
            out << ' ' << value.name;
        }
        out << ": " << fe_type_to_graph_string(value.type);
        if (value.tensorType) {
            out << " tensor=" << graphTensorTypeToString(*value.tensorType);
        }
        if (value.isParameter) {
            out << " param";
        }
        if (value.requiresGrad) {
            out << " requiresGrad";
        }
        if (value.isModelParameter) {
            out << " model_param";
        }
        out << '\n';
    }
    if constexpr (std::is_same_v<GraphT, GraphLayer>) {
        for (const auto& parameter : graph.parameters) {
            out << "  param " << parameter.name
                << " role=" << parameter.role
                << " owner=%" << parameter.ownerValue
                << " value=%" << parameter.valueId
                << " tensor=" << graphTensorTypeToString(parameter.tensorType);
            if (parameter.trainable) {
                out << " trainable";
            }
            out << '\n';
        }
    }
    for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
        const auto& node = graph.nodes[index];
        out << "  node #" << index << ' ' << graph_node_kind_name(node.kind)
            << " -> %" << node.output;
        if (!node.op.empty()) {
            out << " op=" << node.op;
        }
        if (node.opId) {
            out << " opId=" << *node.opId;
        }
        if (node.kind == GraphNodeKind::Binary) {
            out << " op=" << fe_binary_op_to_graph_string(node.binaryOp);
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

std::string graphIrToString(const GraphModule& module) {
    std::ostringstream out;
    out << graphModuleSummary(module) << '\n';
    for (const auto& graph : module.layers) {
        format_graph_body(out, graph, "graph layer ");
    }
    for (const auto& graph : module.functions) {
        format_graph_body(out, graph, "graph fn ");
    }
    for (const auto& skipped : module.skipped) {
        out << "// graph function: " << skipped.functionName << "\n"
            << "// unavailable: " << skipped.reason << '\n';
    }
    return out.str();
}
