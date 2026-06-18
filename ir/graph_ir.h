#pragma once

#include "diagnostic.h"
#include "frontend_ir.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Graph IR is a compact dataflow form after frontend lowering. Values are
// referenced by numeric ids, and nodes describe how each value is produced.
enum class GraphNodeKind {
    Constant,
    Binary,
    PrimitiveCall,
    LibraryCall,
    LibraryCtor,
    Apply,
};

enum class GraphDimKind {
    Unknown,
    Known,
    Symbolic,
};

// Graph dimensions are compiler facts, not runtime storage. Known dimensions
// come from source annotations or op arguments; symbolic dimensions keep names
// such as "batch" when the graph can prove equality but not a concrete size.
struct GraphDim {
    GraphDimKind kind = GraphDimKind::Unknown;
    std::int64_t value = 0;
    std::string symbol;

    static GraphDim unknown();
    static GraphDim known(std::int64_t dim);
    static GraphDim symbolic(std::string name);
};

struct GraphTensorType {
    std::optional<std::string> dtype;
    std::vector<GraphDim> shape;
    bool has_known_rank = false;
};

struct GraphParameter {
    std::string name;
    std::string role;
    std::size_t owner_value = 0;
    std::size_t value_id = 0;
    GraphTensorType tensor_type;
    bool trainable = true;
};

// Value ids are expected to be dense and stable. Execution planning validates
// this and the local executor uses it for vector-backed runtime storage.
struct GraphValue {
    std::size_t id = 0;
    std::string name;
    FeType type;
    bool is_parameter = false;
    bool requires_grad = false;
    bool is_model_parameter = false;
    std::optional<GraphTensorType> tensor_type = std::nullopt;
};

// Operation names are kept for diagnostics/backends, while op_id carries the
// resolved builtin identity when the node maps to a known operation.
struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Constant;
    std::size_t output = 0;
    std::string op;
    std::optional<std::string> op_id;
    FeBinaryOp binary_op = FeBinaryOp::Add;
    FeValue constant = FeValue::none();
    std::vector<std::size_t> inputs;
};

// GraphFunction is the executable shape of a function/layer. named_values keeps
// source symbols available for train objectives and user-facing output lookup.
struct GraphFunction {
    std::string name;
    bool is_layer = false;
    FeType return_type;
    std::vector<GraphValue> values;
    std::vector<GraphParameter> parameters;
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
std::string graph_dim_to_string(const GraphDim& dim);
std::string graph_tensor_type_to_string(const GraphTensorType& type);
std::string graph_module_summary(const GraphModule& module);
std::string graph_ir_to_string(const GraphModule& module);
