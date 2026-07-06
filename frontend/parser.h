#pragma once

#include "diagnostic.h"
#include "lexer.h"
#include "source.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

/**
 * @brief Base kinds for the type system.
 * 
 * Parser types preserve source-level shape and dtype spelling. Semantic analysis
 * later decides whether those annotations are valid and compatible.
 */
enum class TypeBase {
    Unknown,
    Int,
    None,
    Bool,
    Float,
    Str,
    Tensor,
    Tuple,
    List,
    Callable,
};

/**
 * @brief Represents a parsed type annotation.
 * 
 * Includes optional tensor shape properties (dtype, shape expression, rank),
 * tuple/list elements, or callable return types depending on its base kind.
 */
struct Type {
    /**
     * @brief The core category of the type (e.g. Int, Tensor).
     * 
     * Why it exists: Determines how the type behaves and how it is validated.
     * What it tracks: The TypeBase enum.
     * What mutates/updates it: Initialized during parsing and immutable afterwards.
     */
    TypeBase base = TypeBase::None;

    /**
     * @brief Sub-types for collections (Tuple, List).
     * 
     * Why it exists: Captures the types of inner elements in a collection.
     * What it tracks: A list of nested Type objects.
     * What mutates/updates it: Populated during parsing of collection types.
     */
    std::vector<Type> elements;

    /**
     * @brief Return type if this is a Callable.
     * 
     * Why it exists: Specifies what a callable type yields when invoked.
     * What it tracks: A unique pointer to the return Type.
     * What mutates/updates it: Allocated during parsing of callable annotations.
     */
    std::unique_ptr<Type> callable_return;

    /**
     * @brief Data type string for scalar annotations.
     * 
     * Why it exists: Captures explicitly written scalar types (e.g. `f32`).
     * What it tracks: The string representation of the dtype.
     * What mutates/updates it: Set during parsing if an explicit type is provided.
     */
    std::optional<std::string> scalar_dtype;

    /**
     * @brief Data type string for tensor annotations.
     * 
     * Why it exists: Captures the element type of a tensor (e.g. `tensor<f32>`).
     * What it tracks: The dtype string.
     * What mutates/updates it: Set during parsing of tensor types.
     */
    std::optional<std::string> tensor_dtype;

    /**
     * @brief Shape expression for tensor annotations (e.g. "[B, C, H, W]").
     * 
     * Why it exists: Used to validate tensor dimension compatibility.
     * What it tracks: The raw shape string.
     * What mutates/updates it: Extracted during parsing of tensor types.
     */
    std::optional<std::string> tensor_shape_expr;

    /**
     * @brief Resolved rank of the tensor, if deducible.
     * 
     * Why it exists: Assists in structural type-checking for operations dependent on tensor rank.
     * What it tracks: The number of dimensions.
     * What mutates/updates it: Inferred from the parsed shape expression.
     */
    std::optional<std::size_t> tensor_rank;

    Type() = default;
    Type(const Type& other);
    Type& operator=(const Type& other);
    Type(Type&&) noexcept = default;
    Type& operator=(Type&&) noexcept = default;

    // Factory methods for constructing specific types:
    static Type unknown();
    static Type int_type();
    static Type int16();
    static Type int32();
    static Type int64();
    static Type float_type();
    static Type float16();
    static Type float32();
    static Type float64();
    static Type bool_type();
    static Type str_type();
    static Type None_type();
    static Type tensor(
        std::optional<std::string> dtype,
        std::optional<std::string> shape_expr,
        std::optional<std::size_t> rank
    );
    static Type tuple(std::vector<Type> elements);
    static Type list(std::vector<Type> elements);
    static Type callable(std::optional<Type> return_type);
};

struct Expr;
struct Stmt;

using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

/**
 * @brief An argument passed in a function/callable invocation.
 * Contains an optional name (for keyword args) and the expression value.
 */
struct CallArgument {
    /**
     * @brief Optional name of the argument.
     * 
     * Why it exists: Supports keyword arguments in calls.
     * What it tracks: The identifier string if it's a keyword argument.
     * What mutates/updates it: Populated during parsing; empty for positional arguments.
     */
    std::optional<std::string> name;

    /**
     * @brief The expression being passed as the argument.
     * 
     * Why it exists: Represents the value to evaluate and bind to the parameter.
     * What it tracks: A pointer to the parsed Expr.
     * What mutates/updates it: Set when the argument expression is parsed.
     */
    ExprPtr value;
};

// --- AST Node Structures for Expressions ---

struct IntLiteral {
    /**
     * @brief The parsed integer value.
     * 
     * Why it exists: Stores the actual constant to be evaluated.
     * What it tracks: An int64_t representation of the token text.
     * What mutates/updates it: Populated from the lexer's TokenValue.
     */
    std::int64_t value = 0;
};

struct FloatLiteral {
    /**
     * @brief The parsed float value.
     * 
     * Why it exists: Stores the actual floating-point constant.
     * What it tracks: A double representation of the token text.
     * What mutates/updates it: Populated from the lexer's TokenValue.
     */
    double value = 0.0;
};

struct BoolLiteral {
    /**
     * @brief The parsed boolean value.
     * 
     * Why it exists: Stores the literal true/false.
     * What it tracks: A boolean primitive.
     * What mutates/updates it: Evaluated based on whether the token was 'true' or 'false'.
     */
    bool value = false;
};

struct StringLiteral {
    /**
     * @brief The parsed string content.
     * 
     * Why it exists: Stores the textual constant (e.g. for prints or file paths).
     * What it tracks: The inner string characters (without quotes).
     * What mutates/updates it: Populated from the lexer's TokenValue.
     */
    std::string value;
};

struct IdentifierExpr {
    /**
     * @brief The name of the referenced variable or symbol.
     * 
     * Why it exists: Used to look up the symbol in the current scope.
     * What it tracks: The raw identifier string.
     * What mutates/updates it: Populated from the lexer's TokenValue.
     */
    std::string name;
};

struct CallExpr {
    /**
     * @brief The name of the function or layer being called.
     * 
     * Why it exists: Identifies the target of the invocation.
     * What it tracks: The callee identifier text.
     * What mutates/updates it: Populated during parsing.
     */
    std::string callee;

    /**
     * @brief The arguments passed to the call.
     * 
     * Why it exists: Provides the inputs for the invocation.
     * What it tracks: A list of CallArgument elements.
     * What mutates/updates it: Populated during parsing of the argument list.
     */
    std::vector<CallArgument> args;
};

/**
 * @brief Represents a repeated stage execution (e.g., `stage()[N]`).
 */
struct RepeatExpr {
    /**
     * @brief The block or stage to repeat.
     * 
     * Why it exists: Defines the operation being executed multiple times.
     * What it tracks: A pointer to the stage Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr stage;

    /**
     * @brief The number of times to repeat the stage.
     * 
     * Why it exists: Controls the iteration count.
     * What it tracks: A pointer to the count Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr count;
};

struct UnaryExpr {
    /**
     * @brief The operand expression.
     * 
     * Why it exists: The subject of the unary operation.
     * What it tracks: A pointer to the inner Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr operand;

    /**
     * @brief The unary operator.
     * 
     * Why it exists: Dictates the specific operation (e.g. negation, not).
     * What it tracks: The TokenType of the operator.
     * What mutates/updates it: Populated from the operator token.
     */
    TokenType op = TokenType::Eof;
};

struct BinaryExpr {
    /**
     * @brief The left-hand side operand.
     * 
     * Why it exists: Provides the first input to the binary operator.
     * What it tracks: A pointer to the LHS Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr lhs;

    /**
     * @brief The right-hand side operand.
     * 
     * Why it exists: Provides the second input to the binary operator.
     * What it tracks: A pointer to the RHS Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr rhs;

    /**
     * @brief The binary operator.
     * 
     * Why it exists: Dictates the operation (e.g. addition, multiplication).
     * What it tracks: The TokenType of the operator.
     * What mutates/updates it: Populated from the operator token.
     */
    TokenType op = TokenType::Eof;
};

struct TernaryExpr {
    /**
     * @brief The expression to evaluate if the condition is true.
     * 
     * Why it exists: The positive branch of the ternary.
     * What it tracks: A pointer to the 'then' Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr then_expr;

    /**
     * @brief The boolean condition expression.
     * 
     * Why it exists: Determines which branch to take.
     * What it tracks: A pointer to the conditional Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr condition;

    /**
     * @brief The expression to evaluate if the condition is false.
     * 
     * Why it exists: The negative branch of the ternary.
     * What it tracks: A pointer to the 'else' Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr else_expr;
};

struct TupleExpr {
    /**
     * @brief The elements of the tuple.
     * 
     * Why it exists: Holds the individual values of the compound tuple type.
     * What it tracks: A list of pointers to the inner Exprs.
     * What mutates/updates it: Populated during parsing.
     */
    std::vector<ExprPtr> elements;
};

struct ListExpr {
    /**
     * @brief The elements of the list.
     * 
     * Why it exists: Holds the individual values of the sequence.
     * What it tracks: A list of pointers to the inner Exprs.
     * What mutates/updates it: Populated during parsing.
     */
    std::vector<ExprPtr> elements;
};

/**
 * @brief Represents a sequential pipeline of execution steps connected by '->'.
 */
struct ArrowExpr {
    /**
     * @brief The initial input source to the pipeline.
     * 
     * Why it exists: Defines the starting data that flows through the stages.
     * What it tracks: A pointer to the source Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr source;

    /**
     * @brief The sequence of pipeline stages.
     * 
     * Why it exists: Represents each transformation applied to the data.
     * What it tracks: A list of stage Expr pointers.
     * What mutates/updates it: Populated by chaining `->` expressions during parsing.
     */
    std::vector<ExprPtr> stages;
};

/**
 * @brief A discriminated union of all possible expression types.
 * 
 * The AST is intentionally syntax-shaped. Later IR stages desugar calls,
 * pipelines, config constants, and callable application into meaning-level IR.
 */
using ExprKind = std::variant<
    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    StringLiteral,
    IdentifierExpr,
    CallExpr,
    RepeatExpr,
    UnaryExpr,
    BinaryExpr,
    TernaryExpr,
    TupleExpr,
    ListExpr,
    ArrowExpr
>;

/**
 * @brief A generic Expression AST node wrapping an ExprKind with source location.
 */
struct Expr {
    /**
     * @brief The source location of this expression.
     * 
     * Why it exists: Crucial for emitting accurate error diagnostics during semantic analysis.
     * What it tracks: The span of tokens making up this expression.
     * What mutates/updates it: Assigned when the expression node is constructed by the parser.
     */
    SourceSpan span{};

    /**
     * @brief The specific AST node payload.
     * 
     * Why it exists: Holds the structural data for the specific expression type.
     * What it tracks: The ExprKind variant (e.g., BinaryExpr, CallExpr).
     * What mutates/updates it: Assigned at construction time.
     */
    ExprKind kind;
};

// --- AST Node Structures for Statements ---

/**
 * @brief Represents a variable declaration (e.g., `let x: int = 5`).
 */
struct VarDecl {
    /**
     * @brief The name of the new variable.
     * 
     * Why it exists: Used to register the variable in the current scope.
     * What it tracks: The identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief The explicitly annotated type (or Unknown if inferred).
     * 
     * Why it exists: Used to validate the initializer and type-check the variable.
     * What it tracks: The parsed Type.
     * What mutates/updates it: Populated during parsing.
     */
    Type type;

    /**
     * @brief The initializer expression.
     * 
     * Why it exists: Provides the initial value of the variable.
     * What it tracks: A pointer to the initialization Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr init;

    /**
     * @brief Optional array size for fixed-size declarations.
     * 
     * Why it exists: Used for array allocation if supported.
     * What it tracks: The parsed integer size.
     * What mutates/updates it: Populated during parsing of array types.
     */
    std::optional<std::size_t> array_size;
};

/**
 * @brief Represents variable assignment (e.g., `x = 5`).
 */
struct AssignStmt {
    /**
     * @brief The name of the target variable.
     * 
     * Why it exists: Identifies where the value should be stored.
     * What it tracks: The identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief The new value to assign.
     * 
     * Why it exists: Provides the RHS of the assignment.
     * What it tracks: A pointer to the value Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr value;
};

/**
 * @brief A block of statements enclosed in a new scope.
 */
struct ScopeStmt {
    /**
     * @brief The sequence of statements in the block.
     * 
     * Why it exists: Holds the execution body of the scope.
     * What it tracks: A list of Stmt objects.
     * What mutates/updates it: Appended to while parsing the block.
     */
    std::vector<Stmt> statements;
};

struct IfBranch {
    /**
     * @brief The condition for this branch.
     * 
     * Why it exists: Determines if this branch should execute.
     * What it tracks: A pointer to the condition Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr condition;

    /**
     * @brief The body to execute if the condition is true.
     * 
     * Why it exists: The execution payload of the branch.
     * What it tracks: A pointer to the body Stmt (usually a ScopeStmt).
     * What mutates/updates it: Populated during parsing.
     */
    StmtPtr body;
};

struct IfStmt {
    /**
     * @brief The main condition for the if statement.
     * 
     * Why it exists: Evaluated first to determine branch flow.
     * What it tracks: A pointer to the main condition Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr condition;

    /**
     * @brief The body of the initial if branch.
     * 
     * Why it exists: Executed if the main condition is true.
     * What it tracks: A pointer to the 'then' Stmt.
     * What mutates/updates it: Populated during parsing.
     */
    StmtPtr then_stmt;

    /**
     * @brief Subsequent elif branches.
     * 
     * Why it exists: Handles chained conditional logic.
     * What it tracks: A list of IfBranch objects.
     * What mutates/updates it: Appended to while parsing `elif` blocks.
     */
    std::vector<IfBranch> elifs;

    /**
     * @brief The final else body (if present).
     * 
     * Why it exists: Executed if no prior conditions match.
     * What it tracks: A pointer to the 'else' Stmt.
     * What mutates/updates it: Populated during parsing of the `else` block.
     */
    StmtPtr else_stmt;
};

struct ReturnStmt {
    /**
     * @brief The value to return.
     * 
     * Why it exists: The payload sent back to the caller.
     * What it tracks: A pointer to the return Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr value;
};

struct ExprStmt {
    /**
     * @brief The expression to evaluate.
     * 
     * Why it exists: Allows expressions (like calls) to be used as statements.
     * What it tracks: A pointer to the Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr value;
};

/**
 * @brief A discriminated union of all possible statement types.
 */
using StmtKind = std::variant<ReturnStmt, ExprStmt, VarDecl, AssignStmt, ScopeStmt, IfStmt>;

/**
 * @brief A generic Statement AST node wrapping a StmtKind with source location.
 */
struct Stmt {
    /**
     * @brief The source location of this statement.
     * 
     * Why it exists: Used for reporting syntax or semantic errors.
     * What it tracks: The span of tokens making up this statement.
     * What mutates/updates it: Assigned when constructed by the parser.
     */
    SourceSpan span{};

    /**
     * @brief The specific AST node payload.
     * 
     * Why it exists: Holds the structural data for the specific statement type.
     * What it tracks: The StmtKind variant (e.g., IfStmt, AssignStmt).
     * What mutates/updates it: Assigned at construction time.
     */
    StmtKind kind;
};

/**
 * @brief Function or layer argument definition.
 */
struct Arg {
    /**
     * @brief Name of the parameter.
     * 
     * Why it exists: Used to bind argument values to the local scope.
     * What it tracks: The parameter identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief Type annotation of the parameter.
     * 
     * Why it exists: Enforces type checking at call sites.
     * What it tracks: The parsed Type.
     * What mutates/updates it: Populated during parsing.
     */
    Type type;

    /**
     * @brief Optional default value.
     * 
     * Why it exists: Allows omitting arguments during a call.
     * What it tracks: A pointer to the parsed default Expr.
     * What mutates/updates it: Populated during parsing if an '=' is present.
     */
    ExprPtr default_value;
};

/**
 * @brief Configuration struct field definition.
 */
struct Field {
    /**
     * @brief Name of the config field.
     * 
     * Why it exists: Associates a value with a specific config attribute.
     * What it tracks: The field identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief Type annotation of the field.
     * 
     * Why it exists: Validates the initializer and subsequent assignments.
     * What it tracks: The parsed Type.
     * What mutates/updates it: Populated during parsing.
     */
    Type type;

    /**
     * @brief Initializer expression.
     * 
     * Why it exists: Provides the initial (or only) value of the config field.
     * What it tracks: A pointer to the parsed initialization Expr.
     * What mutates/updates it: Populated during parsing.
     */
    ExprPtr init;
};

/**
 * @brief Represents a standard function definition (`fn`).
 */
struct Function {
    /**
     * @brief The source location of the function definition.
     * 
     * Why it exists: Useful for error reporting on the function signature.
     * What it tracks: The SourceSpan of the definition.
     * What mutates/updates it: Populated during parsing.
     */
    SourceSpan span{};

    /**
     * @brief The name of the function.
     * 
     * Why it exists: Used to call the function and register it in the symbol table.
     * What it tracks: The function identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief The function arguments.
     * 
     * Why it exists: Defines the parameters the function accepts.
     * What it tracks: A list of Arg objects.
     * What mutates/updates it: Populated during parsing.
     */
    std::vector<Arg> args;

    /**
     * @brief The return type of the function.
     * 
     * Why it exists: Used to validate return statements inside the function body.
     * What it tracks: The parsed return Type.
     * What mutates/updates it: Populated during parsing.
     */
    Type return_type;

    /**
     * @brief The function body block.
     * 
     * Why it exists: Contains the actual executable statements.
     * What it tracks: A Stmt (usually a ScopeStmt variant).
     * What mutates/updates it: Populated during parsing.
     */
    Stmt body;
};

/**
 * @brief Represents a neural network layer definition (`layer`).
 */
struct Layer {
    /**
     * @brief The source location of the layer definition.
     * 
     * Why it exists: Useful for error reporting.
     * What it tracks: The SourceSpan of the definition.
     * What mutates/updates it: Populated during parsing.
     */
    SourceSpan span{};

    /**
     * @brief The name of the layer.
     * 
     * Why it exists: Used to instantiate or call the layer.
     * What it tracks: The layer identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief The layer arguments.
     * 
     * Why it exists: Defines initialization or invocation parameters.
     * What it tracks: A list of Arg objects.
     * What mutates/updates it: Populated during parsing.
     */
    std::vector<Arg> args;

    /**
     * @brief The return type of the layer.
     * 
     * Why it exists: Validates the output of the layer.
     * What it tracks: The parsed return Type.
     * What mutates/updates it: Populated during parsing.
     */
    Type return_type;

    /**
     * @brief The layer body block.
     * 
     * Why it exists: Contains the logic to build the layer subgraph.
     * What it tracks: A Stmt (usually a ScopeStmt variant).
     * What mutates/updates it: Populated during parsing.
     */
    Stmt body;
};

/**
 * @brief Represents a configuration block (`config`).
 */
struct Config {
    /**
     * @brief The source location of the config definition.
     * 
     * Why it exists: Useful for error reporting.
     * What it tracks: The SourceSpan of the definition.
     * What mutates/updates it: Populated during parsing.
     */
    SourceSpan span{};

    /**
     * @brief The name of the configuration.
     * 
     * Why it exists: Used to reference the configuration block.
     * What it tracks: The config identifier string.
     * What mutates/updates it: Populated during parsing.
     */
    std::string name;

    /**
     * @brief The configuration fields.
     * 
     * Why it exists: Holds the key-value attributes.
     * What it tracks: A list of Field objects.
     * What mutates/updates it: Populated during parsing.
     */
    std::vector<Field> fields;
};

// Top-level module form. Configs may become train configs during lowering,
// while layers/functions are candidates for graph construction.
struct Program {
    /**
     * @brief Configuration blocks defined in the program.
     * 
     * Why it exists: Stores hyperparameter/training config blocks.
     * What it tracks: A list of Config nodes.
     * What mutates/updates it: Appended to during parsing of top-level items.
     */
    std::vector<Config> configs;

    /**
     * @brief Layer definitions.
     * 
     * Why it exists: Stores custom neural network modules.
     * What it tracks: A list of Layer nodes.
     * What mutates/updates it: Appended to during parsing of top-level items.
     */
    std::vector<Layer> layers;

    /**
     * @brief Function definitions.
     * 
     * Why it exists: Stores standard callable blocks.
     * What it tracks: A list of Function nodes.
     * What mutates/updates it: Appended to during parsing of top-level items.
     */
    std::vector<Function> functions;

    /**
     * @brief Global statements outside any function or layer.
     * 
     * Why it exists: Handles top-level scripting (e.g. global variable definitions).
     * What it tracks: A list of Stmt nodes.
     * What mutates/updates it: Appended to during parsing of top-level items.
     */
    std::vector<Stmt> globals;
};

using ParseResult = std::variant<Program, Diagnostic>;

// Recursive-descent parser over an already-tokenized source stream. Parse
// routines return owned AST nodes so the source text can be discarded.
class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    ParseResult parse_program();
    [[nodiscard]] std::optional<Diagnostic> take_last_diagnostic();

private:
    std::vector<Token> tokens_;
    std::size_t index_ = 0;
    std::optional<Diagnostic> last_diagnostic_;

    ExprPtr parse_expression();
    ExprPtr parse_tuple_expression();
    ExprPtr parse_conditional_expression();
    ExprPtr parse_pipeline_expression();
    ExprPtr parse_logical_or_expression();
    ExprPtr parse_logical_and_expression();
    ExprPtr parse_comparison_expression();
    ExprPtr parse_additive_expression();
    ExprPtr parse_multiplicative_expression();
    ExprPtr parse_unary_expression();
    ExprPtr parse_member_access_expression();
    ExprPtr parse_primary_expression();
    ExprPtr parse_list_expression();

    Stmt parse_stmt();
    Stmt parse_scope();
    Stmt parse_let_var_decl();

    Type parse_type();
    Type parse_list_type_after_open(const std::string& open_error);
    Type parse_tuple_type_after_open(
        TokenType open,
        TokenType close,
        const std::string& open_error,
        const std::string& close_error
    );
    std::vector<Type> parse_type_list(TokenType terminator);
    std::string parse_type_token_sequence(const std::vector<TokenType>& terminators);

    Layer parse_layer();
    Function parse_function();
    Config parse_config();
    std::vector<Arg> parse_callable_args();
    Type parse_callable_return_type(const std::string& colon_error);

    const Token* peek(std::size_t offset) const;
    std::optional<TokenType> peek_kind(std::size_t offset) const;
    Token consume();
    Token expect(TokenType kind, const std::string& message);
    std::string consume_ident(const std::string& message);
    void consume_terminator();

    [[noreturn]] void fail_here(const std::string& message);
    [[noreturn]] void fail_token(const std::string& message, const Token& token);
    void record(Diagnostic diagnostic);
};

std::string type_to_string(const Type& type);
std::string program_summary(const Program& program);
std::string ast_to_string(const Program& program);
