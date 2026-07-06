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
    // Why it exists: To distinguish between static sizes and symbolic sizes.
    // What it tracks: Whether the dimension is known concretely or symbolically.
    // What mutates it: Initialized on creation; usually immutable.
    GraphDimKind kind = GraphDimKind::Unknown;
    // Why it exists: To hold the actual size when statically known.
    // What it tracks: The numerical size of this tensor dimension.
    // What mutates it: Set for Known dimensions during inference.
    std::int64_t value = 0;
    // Why it exists: To reference variables inside environments or symbol tables.
    // What it tracks: The name of the identifier representing a variable or dimension.
    // What mutates it: Bound during creation of variable expressions or dimensions.
    std::string symbol;

    static GraphDim unknown();
    static GraphDim known(std::int64_t dim);
    static GraphDim symbolic(std::string name);
};

// Describes a multidimensional array type (tensor) within the graph.
// If the rank is known, the shape is populated with GraphDims.
struct GraphTensorType {
    // Why it exists: To specify the element representation (e.g. float32, int8).
    // What it tracks: The scalar type of the tensor elements.
    // What mutates it: Resolved during graph lowering.
    std::optional<std::string> dtype;
    // Why it exists: To describe the topological structure of the tensor.
    // What it tracks: A list of dimension sizes (known or symbolic).
    // What mutates it: Constructed from frontend shape expressions; typically immutable here.
    std::vector<GraphDim> shape;
    // Why it exists: To distinguish between unknown rank and zero-rank (scalar).
    // What it tracks: True if the rank of the tensor is determined.
    // What mutates it: Set to true if shape vector completely describes dimensions.
    bool has_known_rank = false;
};

// Represents a trainable or non-trainable parameter associated with a graph value.
// It links the underlying value_id with parameter metadata such as name and shape.
struct GraphParameter {
    // Why it exists: To declare a new symbol in the local scope.
    // What it tracks: The variable's text identifier.
    // What mutates it: Set during lowering.
    std::string name;
    // Why it exists: To distinguish weights, biases, or auxiliary buffers.
    // What it tracks: The semantic purpose of the parameter (e.g., "weight", "bias").
    // What mutates it: Assigned by the frontend node or layer constructor.
    std::string role;
    // Why it exists: To link a parameter to the structure that owns it.
    // What it tracks: The value ID of the layer or tuple that logically owns this parameter.
    // What mutates it: Assigned during graph compilation when layers are flattened.
    std::size_t owner_value = 0;
    // Why it exists: To connect the parameter metadata to its concrete dataflow node.
    // What it tracks: The graph value ID that produces or represents this parameter.
    // What mutates it: Assigned when the parameter node is added to the graph.
    std::size_t value_id = 0;
    // Why it exists: To enforce shape and type constraints on the parameter.
    // What it tracks: The data type and dimensionality of the parameter buffer.
    // What mutates it: Established during parameter creation.
    GraphTensorType tensor_type;
    // Why it exists: To determine if gradients should be computed and applied to it.
    // What it tracks: Whether this parameter is subject to optimization updates.
    // What mutates it: Set based on frontend configuration; immutable.
    bool trainable = true;
};

// Value ids are expected to be dense and stable. Execution planning validates
// this and the local executor uses it for vector-backed runtime storage.
struct GraphValue {
    // Why it exists: To provide a fast, dense handle for referencing this value.
    // What it tracks: The unique numeric ID assigned to this graph value.
    // What mutates it: Sequentially assigned by the GraphBuilder.
    std::size_t id = 0;
    // Why it exists: To declare a new symbol in the local scope.
    // What it tracks: The variable's text identifier.
    // What mutates it: Set during lowering.
    std::string name;
    // Why it exists: To define the variable's memory layout and constraints.
    // What it tracks: The statically resolved type of the variable.
    // What mutates it: Set during semantic checks.
    FeType type;
    // Why it exists: To differentiate computed intermediates from persistent state.
    // What it tracks: True if this value corresponds to a persistent model parameter.
    // What mutates it: Set to true when the value is generated from a layer constructor or explicit param.
    bool is_parameter = false;
    // Why it exists: To instruct the autodiff engine whether this value needs a gradient.
    // What it tracks: True if a parameter is trainable or if an intermediate depends on a trainable param.
    // What mutates it: Initially set by parameter flags; propagated forward during graph analysis.
    bool requires_grad = false;
    // Why it exists: To identify top-level configuration values passed to the model.
    // What it tracks: True if the value comes directly from the model's signature.
    // What mutates it: Marked true for inputs derived from model function arguments.
    bool is_model_parameter = false;
    // Why it exists: To enforce shape and type constraints on the parameter.
    // What it tracks: The data type and dimensionality of the parameter buffer.
    // What mutates it: Established during parameter creation.
    std::optional<GraphTensorType> tensor_type = std::nullopt;
};

// Operation names are kept for diagnostics/backends, while op_id carries the
// resolved builtin identity when the node maps to a known operation.
struct GraphNode {
    // Why it exists: To specify how to compute the node.
    // What it tracks: The fundamental operation category (e.g., Binary, Apply, Constant).
    // What mutates it: Established at node creation.
    GraphNodeKind kind = GraphNodeKind::Constant;
    // Why it exists: To link the node to the value it produces.
    // What it tracks: The value ID generated by this node's evaluation.
    // What mutates it: Assigned during node addition to the graph.
    std::size_t output = 0;
    // Why it exists: For readable serialization and diagnostic messages.
    // What it tracks: The string name of the operation (e.g., "Add", "Relu").
    // What mutates it: Extracted from frontend IR node.
    std::string op;
    // Why it exists: To bypass string matching for known builtin ops.
    // What it tracks: A stable identifier for resolved primitive operations.
    // What mutates it: Resolved during graph building if a match is found.
    std::optional<std::string> op_id;
    // Why it exists: To define what arithmetic or logic operation is performed.
    // What it tracks: The specific frontend binary operator (e.g., Add, Mul, Eq).
    // What mutates it: Assigned when parsing/lowering a binary operation.
    FeBinaryOp binary_op = FeBinaryOp::Add;
    // Why it exists: To hold a runtime or compile-time evaluated constant.
    // What it tracks: The exact underlying primitive or composite data value (using std::variant).
    // What mutates it: Constructed by factory methods and typically passed by value; mutable by assignment if needed.
    FeValue constant = FeValue::none();
    // Why it exists: To capture dataflow dependencies.
    // What it tracks: The value IDs consumed as arguments by this node.
    // What mutates it: Populated with argument IDs when the node is lowered.
    std::vector<std::size_t> inputs;
};

// GraphFunction is the executable shape of a function/layer. named_values keeps
// source symbols available for train objectives and user-facing output lookup.
struct GraphFunction {
    // Why it exists: To identify the function or layer.
    // What it tracks: The name of the defined callable.
    // What mutates it: Set during parsing/lowering.
    std::string name;
    // Why it exists: To differentiate standard functions from trainable network layers.
    // What it tracks: True if the function represents a model layer with parameters/state.
    // What mutates it: Inferred from the defining AST node (e.g., `layer` keyword vs `def`).
    bool is_layer = false;
    // Why it exists: To enforce type safety on function/layer outputs.
    // What it tracks: The declared or inferred return type of the callable.
    // What mutates it: Resolved during semantic analysis.
    FeType return_type;
    // Why it exists: To serve as the registry of all variables/dataflow edges in the graph.
    // What it tracks: Metadata for every value ID referenced in the function.
    // What mutates it: Appended to sequentially as expressions are lowered.
    std::vector<GraphValue> values;
    // Why it exists: To list the persistent state items required by this graph.
    // What it tracks: Parameter metadata tying specific value IDs to weights/biases.
    // What mutates it: Appended to when layer constuctors are encountered.
    std::vector<GraphParameter> parameters;
    // Why it exists: To form the execution trace of the function.
    // What it tracks: The topologically sorted list of operations to execute.
    // What mutates it: Appended to sequentially as statements/expressions are lowered.
    std::vector<GraphNode> nodes;
    // Why it exists: To identify which values are yielded when the function returns.
    // What it tracks: The value IDs returned by the function.
    // What mutates it: Populated upon encountering return statements.
    std::vector<std::size_t> outputs;
    // Why it exists: To bridge source-level symbols with graph IDs for debugging and objectives.
    // What it tracks: A mapping from original variable names to their graph value IDs.
    // What mutates it: Inserted into whenever a named variable is declared or assigned.
    std::map<std::string, std::size_t> named_values;
};

struct GraphBuildSkipped {
    // Why it exists: To identify which function failed to compile.
    // What it tracks: The original name of the function skipped.
    // What mutates it: Set upon encountering a build failure.
    std::string function_name;
    // Why it exists: To inform the user why graph generation was skipped.
    // What it tracks: The error message or unsupported feature reason.
    // What mutates it: Extracted from diagnostics.
    std::string reason;
};

struct GraphModule {
    // Why it exists: To contain the successfully compiled graph components.
    // What it tracks: The fully lowered graph representations of functions/layers.
    // What mutates it: Accumulated as the builder processes the frontend module.
    std::vector<GraphFunction> functions;
    // Why it exists: To keep a record of what could not be compiled without halting entirely.
    // What it tracks: Information about functions skipped during graph building.
    // What mutates it: Pushed into when function lowering returns a diagnostic.
    std::vector<GraphBuildSkipped> skipped;
};

using GraphFunctionResult = std::variant<GraphFunction, Diagnostic>;

// Lowers Frontend IR functions into Graph IR functions.
// This builder traverses the linear list of frontend statements and converts
// expressions into an ordered series of GraphNode dependencies.
class GraphBuilder {
public:
    explicit GraphBuilder(const FeFunction& function);

    // Orchestrates the graph lowering process.
    GraphFunctionResult build();

private:
    // Why it exists: To provide access to the frontend function AST/IR.
    // What it tracks: The function node currently being built into a graph.
    // What mutates it: Passed via constructor; immutable reference.
    const FeFunction& function_;

    // Lowers a frontend expression into one or more graph nodes and returns the value id.
    std::variant<std::size_t, Diagnostic> lower_expr(const FeExprPtr& expr, GraphFunction& graph) const;
    
    // Looks up the numeric value ID associated with a named symbol in the graph.
    std::variant<std::size_t, Diagnostic> lookup_named_value(const std::string& name, const GraphFunction& graph) const;
};

GraphFunctionResult build_graph_function(const FeFunction& function);
std::variant<GraphModule, Diagnostic> build_graph_module(const LoweredModule& module);
std::optional<Diagnostic> validate_graph_function(const GraphFunction& graph);
std::string graph_dim_to_string(const GraphDim& dim);
std::string graph_tensor_type_to_string(const GraphTensorType& type);
std::string graph_module_summary(const GraphModule& module);
std::string graph_ir_to_string(const GraphModule& module);
