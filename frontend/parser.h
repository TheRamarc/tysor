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
    TypeBase base = TypeBase::None;
    /// Sub-types for collections (Tuple, List).
    std::vector<Type> elements;
    /// Return type if this is a Callable.
    std::unique_ptr<Type> callable_return;
    /// Data type string for scalar annotations.
    std::optional<std::string> scalar_dtype;
    /// Data type string for tensor annotations.
    std::optional<std::string> tensor_dtype;
    /// Shape expression for tensor annotations (e.g. "[B, C, H, W]").
    std::optional<std::string> tensor_shape_expr;
    /// Resolved rank of the tensor, if deducible.
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
    std::optional<std::string> name;
    ExprPtr value;
};

// --- AST Node Structures for Expressions ---

struct IntLiteral {
    std::int64_t value = 0;
};

struct FloatLiteral {
    double value = 0.0;
};

struct BoolLiteral {
    bool value = false;
};

struct StringLiteral {
    std::string value;
};

struct IdentifierExpr {
    std::string name;
};

struct CallExpr {
    std::string callee;
    std::vector<CallArgument> args;
};

/**
 * @brief Represents a repeated stage execution (e.g., `stage()[N]`).
 */
struct RepeatExpr {
    ExprPtr stage;
    ExprPtr count;
};

struct UnaryExpr {
    ExprPtr operand;
    TokenType op = TokenType::Eof;
};

struct BinaryExpr {
    ExprPtr lhs;
    ExprPtr rhs;
    TokenType op = TokenType::Eof;
};

struct TernaryExpr {
    ExprPtr then_expr;
    ExprPtr condition;
    ExprPtr else_expr;
};

struct TupleExpr {
    std::vector<ExprPtr> elements;
};

struct ListExpr {
    std::vector<ExprPtr> elements;
};

/**
 * @brief Represents a sequential pipeline of execution steps connected by '->'.
 */
struct ArrowExpr {
    ExprPtr source;
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
    SourceSpan span{};
    ExprKind kind;
};

// --- AST Node Structures for Statements ---

/**
 * @brief Represents a variable declaration (e.g., `let x: int = 5`).
 */
struct VarDecl {
    std::string name;
    Type type;
    ExprPtr init;
    std::optional<std::size_t> array_size;
};

/**
 * @brief Represents variable assignment (e.g., `x = 5`).
 */
struct AssignStmt {
    std::string name;
    ExprPtr value;
};

/**
 * @brief A block of statements enclosed in a new scope.
 */
struct ScopeStmt {
    std::vector<Stmt> statements;
};

struct IfBranch {
    ExprPtr condition;
    StmtPtr body;
};

struct IfStmt {
    ExprPtr condition;
    StmtPtr then_stmt;
    std::vector<IfBranch> elifs;
    StmtPtr else_stmt;
};

struct ReturnStmt {
    ExprPtr value;
};

struct ExprStmt {
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
    SourceSpan span{};
    StmtKind kind;
};

/**
 * @brief Function or layer argument definition.
 */
struct Arg {
    std::string name;
    Type type;
    ExprPtr default_value;
};

/**
 * @brief Configuration struct field definition.
 */
struct Field {
    std::string name;
    Type type;
    ExprPtr init;
};

/**
 * @brief Represents a standard function definition (`fn`).
 */
struct Function {
    SourceSpan span{};
    std::string name;
    std::vector<Arg> args;
    Type return_type;
    Stmt body;
};

/**
 * @brief Represents a neural network layer definition (`layer`).
 */
struct Layer {
    SourceSpan span{};
    std::string name;
    std::vector<Arg> args;
    Type return_type;
    Stmt body;
};

/**
 * @brief Represents a configuration block (`config`).
 */
struct Config {
    SourceSpan span{};
    std::string name;
    std::vector<Field> fields;
};

// Top-level module form. Configs may become train configs during lowering,
// while layers/functions are candidates for graph construction.
struct Program {
    std::vector<Config> configs;
    std::vector<Layer> layers;
    std::vector<Function> functions;
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
