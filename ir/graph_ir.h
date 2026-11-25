#pragma once

#include "diagnostic.h"
#include "frontend_ir.h"
#include "ops.h"

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
    bool hasKnownRank = false;
};

// Represents a trainable or non-trainable parameter associated with a graph value.
// It links the underlying valueId with parameter metadata such as name and shape.
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
    std::size_t ownerValue = 0;
    // Why it exists: To connect the parameter metadata to its concrete dataflow node.
    // What it tracks: The graph value ID that produces or represents this parameter.
    // What mutates it: Assigned when the parameter node is added to the graph.
    std::size_t valueId = 0;
    // Why it exists: To enforce shape and type constraints on the parameter.
    // What it tracks: The data type and dimensionality of the parameter buffer.
    // What mutates it: Established during parameter creation.
    GraphTensorType tensorType;
    // Why it exists: To determine if gradients should be computed and applied to it.
    // What it tracks: Whether this parameter is subject to optimization updates.
    // What mutates it: Set based on frontend configuration; immutable.
    bool trainable = true;
};

enum class Placement {
    Host,
    Device,
};

// Value ids are expected to be dense and stable. Execution planning validates
// this and the local executor uses it for vector-backed runtime storage.
struct GraphValue {
    std::size_t id = 0;
    std::string name;
    FeType type;
    bool isParameter = false;
    bool requiresGrad = false;
    bool isModelParameter = false;
    std::optional<GraphTensorType> tensorType = std::nullopt;
    Placement placement = Placement::Host;
};

// Operation names are kept for diagnostics/backends, while opId carries the
// resolved builtin identity when the node maps to a known operation.
struct GraphNode {
    GraphNode() = default;

    GraphNode(GraphNodeKind node_kind, std::size_t output_id, std::string op_name,
              std::optional<std::string> op_identifier, FeBinaryOp binary,
              FeValue constant_value, std::vector<std::size_t> input_ids,
              BackendKind target_backend = BackendKind::Host,
              std::optional<OpId> resolved = std::nullopt)
        : kind(node_kind), output(output_id), op(std::move(op_name)),
          opId(std::move(op_identifier)), binaryOp(binary),
          constant(std::move(constant_value)), inputs(std::move(input_ids)),
          backend(target_backend), resolvedOp(resolved) {}

    GraphNodeKind kind = GraphNodeKind::Constant;
    std::size_t output = 0;
    std::string op;
    std::optional<std::string> opId;
    FeBinaryOp binaryOp = FeBinaryOp::Add;
    FeValue constant = FeValue::none();
    std::vector<std::size_t> inputs;
    BackendKind backend = BackendKind::Host;
    std::optional<OpId> resolvedOp;
};

inline bool planOpIs(const GraphNode& op, OpId id) {
    if (op.resolvedOp) {
        return *op.resolvedOp == id;
    }
    return identifiesOp(op.op, op.opId, id);
}

enum class GraphStepKind {
    AllocateHostValue,
    AllocateDeviceValue,
    ExecuteOp,
    MaterializeOutput,
    UploadToDevice,
    DispatchDeviceOp,
    DownloadToHost,
};

enum class CapabilityStatus {
    Supported,
    Unsupported,
};

struct CapabilityCheck {
    CapabilityStatus status = CapabilityStatus::Supported;
    std::optional<std::string> reason;
};

struct BackendCapabilitySummary {
    BackendKind backend = BackendKind::Host;
    std::vector<std::string> primitiveOps;
    std::vector<std::string> libraryOps;
    std::vector<std::string> constructors;
    bool supportsBinaryOps = true;
    std::vector<std::string> notes;
};

struct GraphStep {
    GraphStepKind kind = GraphStepKind::AllocateHostValue;
    std::size_t valueId = 0;
    std::optional<std::size_t> opIndex;
};

// Graph is the unified middle-end IR for both stateless functions and stateful layers.
// During building it holds SSA-like dataflow values/nodes/parameters; scheduling
// populates value placement, node backend/resolvedOp, and ordered execution steps.
struct Graph {
    std::string name;
    FeType returnType;
    std::vector<GraphValue> values;
    std::vector<GraphParameter> parameters;
    std::vector<GraphNode> nodes;
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    std::map<std::string, std::size_t> namedValues;
    BackendKind backend = BackendKind::Host;
    std::vector<GraphStep> steps;
    bool isLayer = false;
};

using GraphFunction = Graph;
using GraphLayer = Graph;

struct GraphBuildSkipped {
    std::string functionName;
    std::string reason;
};

struct GraphModule {
    BackendKind backend = BackendKind::Host;
    bool scheduled = false;
    std::vector<Graph> functions;
    std::vector<Graph> layers;
    std::vector<GraphBuildSkipped> skipped;
};

struct GraphOptimizationOptions {
    bool preserveIntermediateValues = true;
    bool enableOperatorFusion = true;
};

using GraphFunctionResult = std::variant<Graph, Diagnostic>;
using GraphLayerResult = std::variant<Graph, Diagnostic>;
using GraphModuleResult = std::variant<GraphModule, Diagnostic>;

inline GraphModule makeGraphModule(Graph graph) {
    GraphModule module;
    if (graph.isLayer) {
        module.layers.push_back(std::move(graph));
    } else {
        module.functions.push_back(std::move(graph));
    }
    return module;
}

GraphFunctionResult buildGraphFunction(const FeFunction& function);
GraphLayerResult buildGraphLayer(const FeLayer& layer);
GraphModuleResult buildGraphModule(const LoweredModule& module);
std::optional<Diagnostic> validateGraph(const Graph& graph);
std::optional<Diagnostic> validateGraphFunction(const Graph& graph);
std::optional<Diagnostic> validateGraphLayer(const Graph& graph);
std::string graphDimToString(const GraphDim& dim);
std::string graphTensorTypeToString(const GraphTensorType& type);
std::string graphModuleSummary(const GraphModule& module);
std::string graphIrToString(const GraphModule& module);

Graph scheduleGraph(Graph graph, BackendKind backend);
GraphModuleResult scheduleGraphModule(GraphModule module, BackendKind backend);
Graph optimizeGraph(Graph graph, const GraphOptimizationOptions& options);
GraphModule optimizeGraphModule(GraphModule module, const GraphOptimizationOptions& options);
BackendCapabilitySummary backendCapabilitySummary(BackendKind backend);
CapabilityCheck checkGraphOpCapability(BackendKind backend, const GraphNode& op);
std::optional<Diagnostic> validateGraphCapabilities(const Graph& graph);
std::string graphPlanSummary(const GraphModule& module);
std::string graphPlanToString(const GraphModule& module);
