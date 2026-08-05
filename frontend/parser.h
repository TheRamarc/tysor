#pragma once

#include "arena.h"
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
    std::vector<Type> elements;
    std::unique_ptr<Type> callableReturn;
    std::optional<std::string> scalarDtype;
    std::optional<std::string> tensorDtype;
    std::optional<std::string> tensorShapeExpr;
    std::optional<std::size_t> tensorRank;

    Type() = default;
    Type(const Type& other);
    Type& operator=(const Type& other);
    Type(Type&&) noexcept = default;
    Type& operator=(Type&&) noexcept = default;

    // Factory methods for constructing specific types:
    static Type unknown();
    static Type intType();
    static Type int16();
    static Type int32();
    static Type int64();
    static Type floatType();
    static Type float16();
    static Type float32();
    static Type float64();
    static Type boolType();
    static Type strType();
    static Type noneType();
    static Type tensor(
        std::optional<std::string> dtype,
        std::optional<std::string> shapeExpr,
        std::optional<std::size_t> rank
    );
    static Type tuple(std::vector<Type> elements);
    static Type list(std::vector<Type> elements);
    static Type callable(std::optional<Type> returnType);
};

struct Expr;
struct Stmt;

using ExprPtr = Expr*;
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
    ExprPtr thenExpr;
    ExprPtr condition;
    ExprPtr elseExpr;
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
    std::uint32_t id = 0;
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
    std::string name;
    Type type;
    ExprPtr init;
    std::optional<std::size_t> arraySize;
};

struct AssignStmt {
    std::string name;
    ExprPtr value;
};

struct ScopeStmt {
    std::vector<Stmt> statements;
};

struct IfBranch {
    ExprPtr condition;
    StmtPtr body;
};

struct IfStmt {
    ExprPtr condition;
    StmtPtr thenStmt;
    std::vector<IfBranch> elifs;
    StmtPtr elseStmt;
};

struct ReturnStmt {
    ExprPtr value;
};

struct ExprStmt {
    ExprPtr value;
};

using StmtKind = std::variant<ReturnStmt, ExprStmt, VarDecl, AssignStmt, ScopeStmt, IfStmt>;

struct Stmt {
    std::uint32_t id = 0;
    SourceSpan span{};
    StmtKind kind;
};

struct Arg {
    std::string name;
    Type type;
    ExprPtr defaultValue;
};

struct Field {
    std::string name;
    Type type;
    ExprPtr init;
};

struct Function {
    SourceSpan span{};
    std::string name;
    std::vector<Arg> args;
    Type returnType;
    Stmt body;
};

struct Layer {
    SourceSpan span{};
    std::string name;
    std::vector<Arg> args;
    Type returnType;
    Stmt body;
};

struct Config {
    SourceSpan span{};
    std::string name;
    std::vector<Field> fields;
};

struct Program {
    std::unique_ptr<Arena> arena;
    std::vector<Config> configs;
    std::vector<Layer> layers;
    std::vector<Function> functions;
    std::vector<Stmt> globals;
};

using ParseResult = std::variant<Program, Diagnostic>;

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    std::variant<Program, Diagnostic> parseProgram();
    [[nodiscard]] std::optional<Diagnostic> takeLastDiagnostic();

private:
    std::vector<Token> tokens_;
    std::size_t index_ = 0;
    std::unique_ptr<Arena> arena_;
    std::optional<Diagnostic> lastDiagnostic_;

    ExprPtr parseExpression();
    ExprPtr parseTupleExpression();
    ExprPtr parseConditionalExpression();
    ExprPtr parsePipelineExpression();
    ExprPtr parseLogicalOrExpression();
    ExprPtr parseLogicalAndExpression();
    ExprPtr parseComparisonExpression();
    ExprPtr parseAdditiveExpression();
    ExprPtr parseMultiplicativeExpression();
    ExprPtr parseUnaryExpression();
    ExprPtr parseMemberAccessExpression();
    ExprPtr parsePrimaryExpression();
    ExprPtr parseListExpression();
    ExprPtr makeExpr(SourceSpan span, ExprKind kind);
    Stmt makeStmt(SourceSpan span, StmtKind kind);

    Stmt parseStmt();
    Stmt parseScope();
    Stmt parseLetVarDecl();

    Type parseType();
    Type parseListTypeAfterOpen(const std::string& openError);
    Type parseTupleTypeAfterOpen(
        TokenType open,
        TokenType close,
        const std::string& openError,
        const std::string& closeError
    );
    std::vector<Type> parseTypeList(TokenType terminator);
    std::string parseTypeTokenSequence(const std::vector<TokenType>& terminators);

    Layer parseLayer();
    Function parseFunction();
    Config parseConfig();
    std::vector<Arg> parseCallableArgs();
    Type parseCallableReturnType(const std::string& colonError);

    const Token* peek(std::size_t offset) const;
    std::optional<TokenType> peekKind(std::size_t offset) const;
    Token consume();
    Token expect(TokenType kind, const std::string& message);
    std::string consumeIdent(const std::string& message);
    void consumeTerminator();

    [[noreturn]] void failHere(const std::string& message);
    [[noreturn]] void failToken(const std::string& message, const Token& token);
    void record(Diagnostic diagnostic);

    std::uint32_t nextNodeId_ = 1;
};

std::string typeToString(const Type& type);
std::string programSummary(const Program& program);
std::string astToString(const Program& program);
