#pragma once

#include "arena.h"
#include "diagnostic.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Frontend IR is the checked, compiler-friendly form of the program.
//
// Parser AST answers: "what syntax did the user write?"
// Frontend IR answers: "what did that syntax mean after semantic analysis?"
//
// This layer keeps resolved types, resolved call shapes, evaluated config
// constants, and desugared expressions before Graph IR turns them into runtime
// operations.

// Type categories used by lowered expressions, declarations, functions, configs,
// and execution metadata.
enum class FeTypeKind {
    Unknown,
    Int,
    Float,
    Bool,
    Tensor,
    Tuple,
    List,
    Callable,
    None,
    Str,
};

// Lowered type. This is similar to parser Type, but belongs to Frontend IR so
// later compiler stages do not need to keep consulting AST nodes. It captures
// the exact evaluated type for variables, expressions, and functions.
struct FeType {
    // Why it exists: To allow fast switching on type semantics without string matching or subclass checks.
    // What it tracks: The broad category of the type (e.g., Integer, Tensor, Callable).
    // What mutates it: Set during construction by factory methods like FeType::intType() and remains immutable.
    FeTypeKind kind = FeTypeKind::None;
    // Why it exists: To enable recursive type checking and structural typing.
    // What it tracks: The types of nested elements, if this type is a composite (e.g., Tuple, List).
    // What mutates it: Populated by factories like FeType::tuple() and generally not mutated afterwards.
    std::vector<FeType> elements;
    // Why it exists: So the type checker can know what type results from invoking a callable value.
    // What it tracks: The return type if this FeType represents a Callable (e.g., a function or layer).
    // What mutates it: Initialized by FeType::callable() and immutable thereafter.
    std::shared_ptr<FeType> callableReturn;
    // Why it exists: To distinguish exact scalar types beyond just "Int" or "Float".
    // What it tracks: The concrete data type name (e.g., "int32", "float64") for primitive scalars.
    // What mutates it: Assigned during specific type creation and not mutated.
    std::optional<std::string> scalarDtype;
    // Why it exists: To ensure tensor operations only occur between compatible element types.
    // What it tracks: The underlying element data type name (e.g., "float32") for tensors.
    // What mutates it: Assigned by FeType::tensor() and read-only subsequently.
    std::optional<std::string> tensorDtype;
    // Why it exists: To allow symbolic or deferred shape evaluation before graph execution.
    // What it tracks: A string representation of the tensor shape expression, if provided.
    // What mutates it: Set during tensor type parsing and remains unchanged.
    std::optional<std::string> tensorShapeExpr;
    // Why it exists: To enforce dimensionality constraints during semantic analysis.
    // What it tracks: The number of dimensions of the tensor, if statically known.
    // What mutates it: Inferred or provided explicitly during tensor type instantiation; immutable.
    std::optional<std::size_t> tensorRank;

    // Factory methods for creating common FeType instances.
    static FeType unknown();
    static FeType intType();
    static FeType int16();
    static FeType int32();
    static FeType int64();
    static FeType floatType();
    static FeType float16();
    static FeType float32();
    static FeType float64();
    static FeType boolType();
    static FeType strType();
    static FeType tensor(
        std::optional<std::string> dtype,
        std::optional<std::string> shape_expr,
        std::optional<std::size_t> rank
    );
    static FeType tuple(std::vector<FeType> elements);
    static FeType list(std::vector<FeType> elements);
    static FeType callable(FeType returnType);
    static FeType voidType();
    static FeType none();
    static FeType str();
};

// Compile-time value used for constants and evaluated config fields.
// Example: a config field `hidden = 128` becomes FeValue::intValue(128).
struct FeValue;
struct FeTupleValue {
    // Why it exists: To aggregate multiple values within a tuple.
    // What it tracks: The sequence of FeValue instances in this tuple.
    // What mutates it: Initialized on creation, typically read-only afterward.
    std::vector<FeValue> values;
};
struct FeListValue {
    // Why it exists: To aggregate multiple values within a list.
    // What it tracks: The sequence of FeValue instances in this list.
    // What mutates it: Initialized on creation, typically read-only afterward.
    std::vector<FeValue> values;
};

struct FeValue {
    using Storage = std::variant<std::monostate, std::int64_t, double, bool, std::string, FeTupleValue, FeListValue>;
    // Why it exists: To hold a runtime or compile-time evaluated constant.
    // What it tracks: The exact underlying primitive or composite data value (using std::variant).
    // What mutates it: Constructed by factory methods and typically passed by value; mutable by assignment if needed.
    Storage value;

    // Factory methods to create values of different types.
    static FeValue none();
    static FeValue intValue(std::int64_t value);
    static FeValue floatValue(double value);
    static FeValue boolValue(bool value);
    static FeValue stringValue(std::string value);
    static FeValue tupleValue(std::vector<FeValue> values);
    static FeValue listValue(std::vector<FeValue> values);
};

// Compiler-level binary operators. These replace raw parser TokenType operators
// like Plus, Minus, EqualEqual, etc.
// this is used by graph node, why this enum also include lt, gt.
enum class FeBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Not,
};

struct FeExpr;
using FeExprPtr = FeExpr*;

// Lowered call argument. `name` preserves keyword arguments; `value` is already
// lowered and typed.
struct FeCallArg {
    // Why it exists: To support keyword arguments during function/layer invocation.
    // What it tracks: The formal parameter name this argument binds to, if named.
    // What mutates it: Set during parser lowering; unmodified thereafter.
    std::optional<std::string> name;
    // Why it exists: To hold the actual expression being evaluated or passed.
    // What it tracks: A pointer to the underlying evaluated Frontend IR expression.
    // What mutates it: Constructed during lowering and attached to AST nodes; rarely mutated.
    FeExprPtr value;
};

struct FeConstantExpr {
    // Why it exists: To wrap an evaluated static constant in expression semantics.
    // What it tracks: The literal compile-time value.
    // What mutates it: Assigned on initialization.
    FeValue value;
};

// Resolved variable reference. The expression wrapper carries the FeType, so the
// variable node only needs the symbol name.
struct FeVarExpr {
    // Why it exists: To reference variables inside environments or symbol tables.
    // What it tracks: The name of the identifier representing a variable.
    // What mutates it: Bound during creation of variable expressions.
    std::string symbol;
};

// Normal call that produces a value immediately, like relu(x) or sqrt(x).
struct FeCallExpr {
    // Why it exists: To identify the target of a call before resolution to a concrete function.
    // What it tracks: The name of the function, primitive, or layer being invoked.
    // What mutates it: Extracted from syntax and stored as immutable.
    std::string callee;
    // Why it exists: To provide the necessary inputs for a function call.
    // What it tracks: The sequence of lowered arguments (both positional and keyword).
    // What mutates it: Built during expression lowering.
    std::vector<FeCallArg> args;
};

// Constructor-style library call that creates a callable object, like
// linear(...), Embedding(...), or SiLU().
struct FeLayerCtorExpr {
    // Why it exists: To identify the target of a layer construction call.
    // What it tracks: The name of the layer being instantiated.
    // What mutates it: Extracted from syntax and stored as immutable.
    std::string callee;
    // Why it exists: To provide the necessary inputs for a constructor call.
    // What it tracks: The sequence of lowered arguments.
    // What mutates it: Built during expression lowering.
    std::vector<FeCallArg> args;
};

// Applying a callable value to arguments. Example: if dense is a linear layer,
// dense(x) lowers to FeApplyExpr.
struct FeApplyExpr {
    // Why it exists: Because the target of an apply operation might be a complex expression itself (e.g., a lambda or loaded layer).
    // What it tracks: The lowered expression that yields a callable object.
    // What mutates it: Bound during apply expression construction.
    FeExprPtr callee;
    // Why it exists: To provide inputs to the callable.
    // What it tracks: The sequence of lowered arguments.
    // What mutates it: Built during expression lowering.
    std::vector<FeCallArg> args;
};

struct FeTupleExpr {
    // Why it exists: To structure multi-element tuple expressions.
    // What it tracks: The sub-expressions comprising a tuple.
    // What mutates it: Populated during tuple expression lowering.
    std::vector<FeExprPtr> elements;
};

struct FeListExpr {
    // Why it exists: To structure multi-element list expressions.
    // What it tracks: The sub-expressions comprising a list.
    // What mutates it: Populated during list expression lowering.
    std::vector<FeExprPtr> elements;
};

struct FeBinaryExpr {
    // Why it exists: To define what arithmetic or logic operation is performed.
    // What it tracks: The specific frontend binary operator (e.g., Add, Mul, Eq).
    // What mutates it: Assigned when parsing/lowering a binary operation.
    FeBinaryOp op = FeBinaryOp::Add;
    // Why it exists: To specify the left operand of a binary expression.
    // What it tracks: The lowered expression on the left side of the operator.
    // What mutates it: Set during expression tree construction.
    FeExprPtr lhs;
    // Why it exists: To specify the right operand of a binary expression.
    // What it tracks: The lowered expression on the right side of the operator.
    // What mutates it: Set during expression tree construction.
    FeExprPtr rhs;
};

struct FeIfThenElseExpr {
    // Why it exists: To dictate control flow branching.
    // What it tracks: The boolean condition expression for if/else blocks.
    // What mutates it: Evaluated and assigned during parsing/lowering.
    FeExprPtr condition;
    // Why it exists: To provide the result if the condition is true.
    // What it tracks: The expression evaluated in the true-branch.
    // What mutates it: Set during node creation.
    FeExprPtr thenExpr;
    // Why it exists: To provide the result if the condition is false.
    // What it tracks: The expression evaluated in the false-branch.
    // What mutates it: Set during node creation.
    FeExprPtr elseExpr;
};

using FeExprKind = std::variant<
    FeConstantExpr,
    FeVarExpr,
    FeCallExpr,
    FeLayerCtorExpr,
    FeApplyExpr,
    FeTupleExpr,
    FeListExpr,
    FeBinaryExpr,
    FeIfThenElseExpr
>;

// A lowered expression. Compared with AST Expr, this adds the resolved type and
// uses meaning-level expression nodes instead of syntax-only nodes. This structure
// holds the evaluated result type alongside the expression variant.
struct FeExpr {
    // Why it exists: To cache the statically determined type of the expression.
    // What it tracks: The resolved FeType representing what this construct evaluates to.
    // What mutates it: Determined by the semantic analyzer and set during construction.
    FeType type;
    // Why it exists: To define the variant form of the expression.
    // What it tracks: The specific expression subclass (e.g., constant, call, binary).
    // What mutates it: Set at creation time.
    FeExprKind kind;

    // Factory methods for creating specific FeExpr nodes.
    static FeExprPtr constant(Arena& arena, FeValue value, FeType type);
    static FeExprPtr var(Arena& arena, std::string symbol, FeType type);
    static FeExprPtr call(Arena& arena, std::string callee, std::vector<FeCallArg> args, FeType type);
    static FeExprPtr layerCtor(Arena& arena, std::string callee, std::vector<FeCallArg> args, FeType type);
    static FeExprPtr apply(Arena& arena, FeExprPtr callee, std::vector<FeCallArg> args, FeType type);
    static FeExprPtr tuple(Arena& arena, std::vector<FeExprPtr> elements, FeType type);
    static FeExprPtr list(Arena& arena, std::vector<FeExprPtr> elements, FeType type);
    static FeExprPtr binary(Arena& arena, FeBinaryOp op, FeExprPtr lhs, FeExprPtr rhs, FeType type);
    static FeExprPtr ifThenElse(Arena& arena, FeExprPtr condition, FeExprPtr thenExpr, FeExprPtr elseExpr, FeType type);
};

struct FeStmt;

// Lowered variable declaration. If hasValue is false, value may be null.
// Graph-local declarations usually need a value because graph lowering is
// expression-driven.
struct FeVarDeclStmt {
    // Why it exists: To declare a new symbol in the local scope.
    // What it tracks: The variable's text identifier.
    // What mutates it: Set during lowering.
    std::string name;
    // Why it exists: To define the variable's memory layout and constraints.
    // What it tracks: The statically resolved type of the variable.
    // What mutates it: Set during semantic checks.
    FeType type;
    // Why it exists: To bind the initial data to the variable.
    // What it tracks: The evaluated expression forming the initial state.
    // What mutates it: Assigned if an initializer is present.
    FeExprPtr value;
    // Why it exists: To distinguish between an initialized variable and a purely declared one.
    // What it tracks: True if the variable declaration has an initial value.
    // What mutates it: Set based on the presence of an assignment in the AST.
    bool hasValue = false;
};

struct FeAssignStmt {
    // Why it exists: To mutate an existing variable.
    // What it tracks: The name of the symbol being assigned.
    // What mutates it: Set during lowering.
    std::string name;
    // Why it exists: To dictate the new data for the variable.
    // What it tracks: The expression yielding the new value.
    // What mutates it: Set during lowering.
    FeExprPtr value;
};

struct FeReturnStmt {
    // Why it exists: To return a value from a block/function.
    // What it tracks: The expression being returned.
    // What mutates it: Set during parsing.
    FeExprPtr value;
};

struct FeExprStmt {
    // Why it exists: To execute an expression purely for side effects or execution flow.
    // What it tracks: The expression to evaluate.
    // What mutates it: Set during parsing.
    FeExprPtr value;
};

struct FeElifBody {
    // Why it exists: To handle a single elif branch.
    // What it tracks: The branch condition.
    // What mutates it: Evaluated and assigned during parsing.
    FeExprPtr condition;
    // Why it exists: To specify the statements executing if this elif branch is taken.
    // What it tracks: The sequence of statements in the body.
    // What mutates it: Populated during block lowering.
    std::vector<FeStmt> body;
};

struct FeIfStmt {
    // Why it exists: To form the primary if condition.
    // What it tracks: The boolean condition expression.
    // What mutates it: Assigned during parsing.
    FeExprPtr condition;
    // Why it exists: To group statements executed when the `if` condition holds.
    // What it tracks: The sequence of lowered statements inside the block.
    // What mutates it: Populated during scope lowering.
    std::vector<FeStmt> thenBody;
    // Why it exists: To handle chained conditional logic without deep nesting.
    // What it tracks: A sequence of `elif` condition-and-body pairs.
    // What mutates it: Constructed from chained `elif` syntax.
    std::vector<FeElifBody> elifBodies;
    // Why it exists: To provide fallback execution when all preceding conditions fail.
    // What it tracks: The lowered statements in the `else` block.
    // What mutates it: Populated during scope lowering.
    std::vector<FeStmt> elseBody;
};

using FeStmtKind = std::variant<FeVarDeclStmt, FeAssignStmt, FeReturnStmt, FeExprStmt, FeIfStmt>;

// Lowered statement. AST ScopeStmt has already been expanded into vectors of
// FeStmt inside functions, layers, and branches.
struct FeStmt {
    // Why it exists: To identify the type of statement.
    // What it tracks: The underlying statement variant (e.g., assign, return, expr).
    // What mutates it: Set during statement initialization.
    FeStmtKind kind;
};

// Lowered function. Represents a pure, stateless callable.
struct FeFunction {
    // Why it exists: To identify the function.
    // What it tracks: The name of the defined callable.
    // What mutates it: Set during parsing/lowering.
    std::string name;
    // Why it exists: To enforce type safety on function outputs.
    // What it tracks: The declared or inferred return type of the callable.
    // What mutates it: Resolved during semantic analysis.
    FeType returnType;
    // Why it exists: To define the input signature of the function.
    // What it tracks: An ordered list of parameter names and their resolved types.
    // What mutates it: Populated by lowering function arguments.
    std::vector<std::pair<std::string, FeType>> params;
    // Why it exists: To support multiple or named return values.
    // What it tracks: Names and types of output bindings.
    // What mutates it: Extracted from function signatures.
    std::vector<std::pair<std::string, FeType>> namedOutputs;
    // Why it exists: To hold the executable logic of a function.
    // What it tracks: The ordered list of statements comprising the block.
    // What mutates it: Built during block traversal and scope lowering.
    std::vector<FeStmt> body;
};

// Lowered layer. Represents a stateful neural network module.
struct FeLayer {
    // Why it exists: To identify the layer.
    // What it tracks: The name of the defined layer module.
    // What mutates it: Set during parsing/lowering.
    std::string name;
    // Why it exists: To enforce type safety on layer outputs.
    // What it tracks: The declared or inferred return type of the layer.
    // What mutates it: Resolved during semantic analysis.
    FeType returnType;
    // Why it exists: To define the input signature of the layer forward pass.
    // What it tracks: An ordered list of parameter names and their resolved types.
    // What mutates it: Populated by lowering layer arguments.
    std::vector<std::pair<std::string, FeType>> params;
    // Why it exists: To support multiple or named return values from the layer.
    // What it tracks: Names and types of output bindings.
    // What mutates it: Extracted from layer signatures.
    std::vector<std::pair<std::string, FeType>> namedOutputs;
    // Why it exists: To hold the forward pass executable logic of the layer.
    // What it tracks: The ordered list of statements comprising the block.
    // What mutates it: Built during block traversal and scope lowering.
    std::vector<FeStmt> body;
};

// Lowered config. Field initializer expressions have already been evaluated into
// FeValue constants.
struct FeConfig {
    // Why it exists: To name the configuration block.
    // What it tracks: The identifier of the config.
    // What mutates it: Set during parsing.
    std::string name;
    // Why it exists: To store configuration parameters in a structured manner.
    // What it tracks: Key-value pairs where values are compile-time evaluated FeValues.
    // What mutates it: Populated as config fields are resolved by the lowerer.
    std::map<std::string, FeValue> fields;
};

// Lowered training config. These vectors are the frontend's normalized view of
// train options before execution-plan expansion.
struct FeTrain {
    // Why it exists: To identify the training run specification.
    // What it tracks: The train block identifier.
    // What mutates it: Set during parsing.
    std::string name;
    // Why it exists: To specify target execution environments (e.g., CPU, Metal).
    // What it tracks: Evaluated constant values denoting backend choices.
    // What mutates it: Extracted from the train config block.
    std::vector<FeValue> backends;
    // Why it exists: To configure the training update algorithms.
    // What it tracks: Constant values specifying optimizers like SGD or Adam.
    // What mutates it: Loaded from the train config fields.
    std::vector<FeValue> optimizers;
    // Why it exists: To test different learning rate hyperparameters.
    // What it tracks: Evaluated constants for step sizes.
    // What mutates it: Loaded from the train config fields.
    std::vector<FeValue> learningRates;
    // Why it exists: To denote which variable/output is being optimized (e.g., the loss).
    // What it tracks: Variable names designated as training objectives.
    // What mutates it: Collected from train config declarations.
    std::vector<std::string> objectiveSymbols;
    // Why it exists: To dictate the duration of training runs.
    // What it tracks: Constant integer values representing step counts or epochs.
    // What mutates it: Populated from train config.
    std::vector<FeValue> iterations;
    // Why it exists: To support hyperparameter search over Cartesian products.
    // What it tracks: The total number of unique training configurations derived from lists.
    // What mutates it: Computed by multiplying lengths of hyperparameter arrays.
    std::size_t variantCount = 1;
    // Why it exists: To hold arbitrary additional training metadata.
    // What it tracks: Unstructured or custom configuration fields.
    // What mutates it: Inserted during config block evaluation.
    std::map<std::string, FeValue> extraProperties;
};

// Where a train objective symbol came from after resolving it against model
// params, outputs, or locals.
enum class ObjectiveSource {
    Param,
    Output,
    Local,
    Unknown,
};

// One concrete run derived from a model + train config combination.
struct FeExecutionRun {
    // Why it exists: To uniquely identify this specific execution instance.
    // What it tracks: A synthesized name typically based on model and train config.
    // What mutates it: Generated during execution plan creation.
    std::string runName;
    // Why it exists: To link the run to its primary model definition.
    // What it tracks: The identifier of the layer/function being executed.
    // What mutates it: Copied from the execution plan entry.
    std::string modelName;
    // Why it exists: To link the run to its training configuration.
    // What it tracks: The identifier of the associated `train` block.
    // What mutates it: Set by the execution planner.
    std::string trainName;
    // Why it exists: To specify where this specific run executes.
    // What it tracks: The resolved backend choice for this run instance.
    // What mutates it: Plucked from the combinatorial expansion of the train config.
    std::optional<FeValue> backend;
    // Why it exists: To assign a specific optimizer to this run.
    // What it tracks: The chosen optimizer configuration.
    // What mutates it: Assigned during run generation.
    std::optional<FeValue> optimizer;
    // Why it exists: To bind a concrete learning rate for this execution.
    // What it tracks: The chosen learning rate value.
    // What mutates it: Assigned during run generation.
    std::optional<FeValue> learningRate;
    // Why it exists: To clarify what is being minimized.
    // What it tracks: The name of the loss or objective variable for this run.
    // What mutates it: Picked from the train config options.
    std::optional<std::string> objectiveSymbol;
    // Why it exists: To instruct the backward pass where to fetch the gradient seed.
    // What it tracks: Whether the objective is a parameter, output, or local intermediate.
    // What mutates it: Resolved by checking the symbol against model scope.
    ObjectiveSource objectiveSource = ObjectiveSource::Unknown;
    // Why it exists: To ensure the objective is mathematically valid for optimization (e.g., a scalar).
    // What it tracks: The resolved type of the objective symbol.
    // What mutates it: Looked up during objective resolution.
    FeType objectiveType;
    // Why it exists: To bound the length of this specific training run.
    // What it tracks: The concrete number of training steps.
    // What mutates it: Plucked from the train config variants.
    std::optional<FeValue> iteration;
};

// Frontend-level execution plan. Later planning stages translate this into
// backend/runtime-specific work.
struct FeExecutionPlan {
    // Why it exists: To identify the main entry point for graph building.
    // What it tracks: The name of the top-level layer or function to execute.
    // What mutates it: Set when an execution plan is formulated from the module.
    std::string modelEntry;
    // Why it exists: To manage a batch of tasks (like hyperparameter sweeps).
    // What it tracks: A collection of distinct execution runs to be performed.
    // What mutates it: Accumulated by the planner expanding config variants.
    std::vector<FeExecutionRun> runs;
};

// Full lowered module produced by FrontendLowerer.
struct LoweredModule {
    // Arena containing all Frontend IR nodes
    std::unique_ptr<Arena> arena;
    // Why it exists: To store global configurations for use by the runtime or model.
    // What it tracks: Lowered configuration blocks.
    // What mutates it: Appended to as the module is built.
    std::vector<FeConfig> configs;
    // Why it exists: To define how to train the models in the module.
    // What it tracks: Lowered training blocks.
    // What mutates it: Appended to as the module is built.
    std::vector<FeTrain> trains;
    // Why it exists: To hold all executable function definitions.
    // What it tracks: Lowered functions available in this module.
    // What mutates it: Collected during module parsing and lowering.
    std::vector<FeFunction> functions;
    // Why it exists: To hold all stateful layer definitions.
    // What it tracks: Lowered layers available in this module.
    // What mutates it: Collected during module parsing and lowering.
    std::vector<FeLayer> layers;
    // Why it exists: To handle module-level variable declarations or assignments.
    // What it tracks: Stmt forms of global variable initializations.
    // What mutates it: Inserted during module lowering.
    std::vector<FeStmt> globals;
    // Why it exists: To provide a recipe for executing the program.
    // What it tracks: The synthesized execution strategy tying models to runs.
    // What mutates it: Generated as a final step in frontend lowering.
    std::optional<FeExecutionPlan> executionPlan;
};

using FrontendResult = std::variant<LoweredModule, Diagnostic>;

// Converts parser AST + SemanticInfo into Frontend IR.
// This class drives the lowering process, expanding config values, verifying types,
// and resolving call targets into unambiguous Frontend IR nodes.
class FrontendLowerer {
public:
    FrontendLowerer(const Program& program, const SemanticInfo& semantic_info);

    // Performs the lowering pass and returns the resulting module or a diagnostic.
    FrontendResult lower();

    // Retrieves and clears the most recent diagnostic error encountered.
    [[nodiscard]] std::optional<Diagnostic> takeLastDiagnostic();

private:
    struct EvaluatedConfigField {
        // Why it exists: To detect circular dependencies in config field evaluation.
        // What it tracks: True while a config field is actively being computed.
        // What mutates it: Set to true when evaluation starts, false when it ends.
        bool inProgress = false;
        // Why it exists: To memoize evaluated config fields.
        // What it tracks: True if the field has been successfully evaluated.
        // What mutates it: Set to true once evaluation concludes successfully.
        bool computed = false;
        // Why it exists: To hold a runtime or compile-time evaluated constant.
        // What it tracks: The exact underlying primitive or composite data value (using std::variant).
        // What mutates it: Constructed by factory methods and typically passed by value; mutable by assignment if needed.
        FeValue value = FeValue::none();
    };

    // Why it exists: To provide access to the raw AST.
    // What it tracks: The parsed source code representation.
    // What mutates it: Passed via constructor; immutable reference.
    const Program& program_;
    // Why it exists: To access previously analyzed type/scope information.
    // What it tracks: Name resolution and type metadata from the semantic phase.
    // What mutates it: Passed via constructor; immutable reference.
    const SemanticInfo& semanticInfo_;
    // Why it exists: To allow quick lookup of config AST nodes during evaluation.
    // What it tracks: A mapping from config block names to their AST definitions.
    // What mutates it: Populated during initialization of the lowerer.
    std::map<std::string, const Config*> configDefs_;
    // Arena for all allocated FeExprs during lowering
    std::unique_ptr<Arena> arena_;
    // Why it exists: To store memoized field results and prevent redundant work.
    // What it tracks: Evaluated states for each field in each config block.
    // What mutates it: Updated as each config field is evaluated.
    std::map<std::string, std::map<std::string, EvaluatedConfigField>> configFieldCache_;
    // Why it exists: To resolve names at the module level.
    // What it tracks: Types of globally declared symbols.
    // What mutates it: Modified when global variables or functions are encountered.
    std::map<std::string, FeType> globalSymbols_;
    // Why it exists: To resolve names within the local scope.
    // What it tracks: Types of symbols active in the current function or block.
    // What mutates it: Pushed/popped or mutated as scopes are entered and exited.
    std::map<std::string, FeType> currentSymbols_;
    // Why it exists: To associate context (like layer state) with current evaluation.
    // What it tracks: The name of the layer or function currently being lowered.
    // What mutates it: Set when entering a function/layer; cleared on exit.
    std::optional<std::string> currentOwner_;
    // Why it exists: To propagate errors gracefully.
    // What it tracks: The most recent error encountered during lowering.
    // What mutates it: Assigned when a diagnostic is emitted; consumed via takeLastDiagnostic().
    std::optional<Diagnostic> lastDiagnostic_;

    std::variant<FeExprPtr, Diagnostic> lowerExpr(const Expr& expr);
    std::variant<FeExprPtr, Diagnostic> lowerArrowExpr(const Expr& expr);
    std::variant<FeExprPtr, Diagnostic> lowerArrowStageExpr(const Expr& expr, FeExprPtr current);
    std::variant<FeExprPtr, Diagnostic> lowerArrowCallStage(
        const std::string& callee,
        const std::vector<CallArgument>& args,
        const SourceSpan& span,
        FeExprPtr current
    );
    std::variant<FeExprPtr, Diagnostic> lowerSemanticArrowCallStage(
        const SemanticCallInfo& call,
        const std::string& callee,
        const std::vector<CallArgument>& args,
        FeExprPtr current
    );
    std::variant<std::vector<FeStmt>, Diagnostic> lowerScope(const Stmt& stmt);
    std::variant<FeStmt, Diagnostic> lowerStmt(const Stmt& stmt);
    std::variant<FeFunction, Diagnostic> lowerFunction(const Function& function);
    std::variant<FeLayer, Diagnostic> lowerLayer(const Layer& layer);
    std::variant<FeConfig, Diagnostic> lowerConfig(const Config& config);
    std::variant<FeTrain, Diagnostic> lowerTrainConfig(const Config& config);
    std::variant<FeExecutionPlan, Diagnostic> buildExecutionPlan(const LoweredModule& module);
    bool resolveObjectiveStmt(const FeStmt& stmt, const std::string& objectiveSymbol, FeExecutionRun& run) const;
    std::variant<FeValue, Diagnostic> evalConstantExpr(const Expr& expr);
    std::variant<std::vector<FeValue>, Diagnostic> evalConstantFieldValues(const Expr& expr);
    std::variant<FeValue, Diagnostic> evalConfigField(
        const std::string& configName,
        const std::string& fieldName,
        const SourceSpan& span
    );
    std::variant<FeValue, Diagnostic> evalBinary(TokenType op, const FeValue& lhs, const FeValue& rhs, const SourceSpan& span);
    std::variant<FeValue, Diagnostic> evalUnary(TokenType op, const FeValue& operand, const SourceSpan& span);
    std::optional<SemanticCallInfo> semanticCallForExpr(const Expr& expr, const std::string& callee) const;
    std::optional<SemanticCallInfo> semanticCallForArrowStage(const std::string& callee, const SourceSpan& span) const;
    std::optional<SemanticIdentifierInfo> semanticIdentifierForExpr(const Expr& expr, const std::string& name) const;
    std::optional<SemanticAssignmentInfo> semanticAssignmentForStmt(const Stmt& stmt, const std::string& name) const;
    std::optional<SemanticConfigFieldAccessInfo> semanticConfigFieldAccessForExpr(const Expr& expr) const;
    std::optional<SemanticDeclarationInfo> semanticDeclarationForStmt(const Stmt& stmt, const std::string& name) const;
    std::optional<FeType> semanticTypeForExpr(const Expr& expr) const;
    std::variant<FeType, Diagnostic> requiredSemanticTypeForExpr(const Expr& expr, const std::string& context);
    void bindSymbol(const std::string& name, FeType type);
    std::optional<FeType> findSymbol(const std::string& name) const;
    Diagnostic error(const std::string& message);
    Diagnostic errorSpan(const SourceSpan& span, const std::string& message);
};

FeType lowerType(const Type& type);
std::variant<FeBinaryOp, Diagnostic> lowerBinaryOp(TokenType token, const SourceSpan& span);
std::string loweredModuleSummary(const LoweredModule& module);
std::string frontendIrToString(const LoweredModule& module);
