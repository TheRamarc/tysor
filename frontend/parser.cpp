#include "parser.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <iostream>

namespace {

struct ParseFailure final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

SourceSpan span_of(const Token& token) {
    return SourceSpan{token.line, token.column};
}

Stmt make_stmt(SourceSpan span, StmtKind kind) {
    Stmt stmt;
    stmt.span = span;
    stmt.kind = std::move(kind);
    return stmt;
}

bool is_ident_like(TokenType kind) {
    switch (kind) {
        case TokenType::Ident:
        case TokenType::Config:
        case TokenType::True:
        case TokenType::False:
        case TokenType::Int:
        case TokenType::Float:
        case TokenType::Bool:
        case TokenType::Layer:
        case TokenType::Fn:
        case TokenType::If:
        case TokenType::Else:
        case TokenType::Elif:
        case TokenType::Return:
            return true;
        default:
            return false;
    }
}

std::string expect_string(const Token& token) {
    if (const auto* value = std::get_if<std::string>(&token.value)) {
        return *value;
    }
    throw ParseFailure("Expected string token at " + std::to_string(token.line) + ":" +
                       std::to_string(token.column));
}

std::int64_t expect_int(const Token& token) {
    if (const auto* value = std::get_if<std::int64_t>(&token.value)) {
        return *value;
    }
    throw ParseFailure("Expected int literal at " + std::to_string(token.line) + ":" +
                       std::to_string(token.column));
}

double expect_float(const Token& token) {
    if (const auto* value = std::get_if<double>(&token.value)) {
        return *value;
    }
    throw ParseFailure("Expected float literal at " + std::to_string(token.line) + ":" +
                       std::to_string(token.column));
}

bool contains_token(const std::vector<TokenType>& values, TokenType kind) {
    return std::find(values.begin(), values.end(), kind) != values.end();
}

std::optional<std::size_t> derive_rank_from_shape_expr(const std::string& shape_expr) {
    const auto colon = shape_expr.find(':');
    const auto close = shape_expr.rfind(']');
    const auto open = shape_expr.rfind('[');
    if (colon == std::string::npos || close == std::string::npos || open == std::string::npos) {
        return std::nullopt;
    }
    if (colon < open || colon > close) {
        return std::nullopt;
    }

    const std::string start = shape_expr.substr(open + 1, colon - open - 1);
    const std::string end = shape_expr.substr(colon + 1, close - colon - 1);
    if (end.empty()) {
        return std::nullopt;
    }

    try {
        const int start_value = start.empty() ? 0 : std::stoi(start);
        const int end_value = std::stoi(end);
        if (end_value < start_value) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(end_value - start_value);
    } catch (...) {
        return std::nullopt;
    }
}

int count_stage_sites(const Expr& expr);

int count_expr_list(const std::vector<ExprPtr>& exprs) {
    int count = 0;
    for (const auto& expr : exprs) {
        count += count_stage_sites(*expr);
    }
    return count;
}

int count_stage_sites(const Expr& expr) {
    return std::visit(
        [](const auto& value) -> int {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, CallExpr>) {
                return 1;
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                return count_stage_sites(*value.stage) + count_stage_sites(*value.count);
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                return count_stage_sites(*value.operand);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                return count_stage_sites(*value.lhs) + count_stage_sites(*value.rhs);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                return count_stage_sites(*value.then_expr) + count_stage_sites(*value.condition) +
                       count_stage_sites(*value.else_expr);
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                return count_expr_list(value.elements);
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                return count_expr_list(value.elements);
            } else if constexpr (std::is_same_v<T, ArrowExpr>) {
                return count_stage_sites(*value.source) + count_expr_list(value.stages);
            } else {
                return 0;
            }
        },
        expr.kind
    );
}

std::string indent(std::size_t count) {
    return std::string(count, ' ');
}

void append_expr_summary(std::ostringstream& out, const Expr& expr);
void append_stmt_summary(std::ostringstream& out, const Stmt& stmt, std::size_t level);

const char* operator_text(TokenType kind) {
    switch (kind) {
        case TokenType::Plus:
            return "+";
        case TokenType::Minus:
            return "-";
        case TokenType::Star:
            return "*";
        case TokenType::Slash:
            return "/";
        case TokenType::DoubleSlash:
            return "//";
        case TokenType::EqEq:
            return "==";
        case TokenType::Neq:
            return "!=";
        case TokenType::Lt:
            return "<";
        case TokenType::Gt:
            return ">";
        case TokenType::LtEq:
            return "<=";
        case TokenType::GtEq:
            return ">=";
        case TokenType::AmpAmp:
            return "&&";
        case TokenType::PipePipe:
            return "||";
        case TokenType::Bang:
            return "!";
        case TokenType::Dot:
            return ".";
        default:
            return token_type_name(kind);
    }
}

void append_joined_exprs(std::ostringstream& out, const std::vector<ExprPtr>& exprs) {
    for (std::size_t index = 0; index < exprs.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        append_expr_summary(out, *exprs[index]);
    }
}

void append_args(std::ostringstream& out, const std::vector<Arg>& args) {
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        const auto& arg = args[index];
        // if (arg.is_mutable) {
        //     out << "mut ";
        // }
        out << arg.name;
        if (arg.type.base != TypeBase::Unknown) {
            out << ": " << type_to_string(arg.type);
        }
        if (arg.default_value) {
            out << " = ";
            append_expr_summary(out, *arg.default_value);
        }
    }
}

void append_expr_summary(std::ostringstream& out, const Expr& expr) {
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IntLiteral>) {
                out << value.value;
            } else if constexpr (std::is_same_v<T, FloatLiteral>) {
                out << value.value;
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                out << (value.value ? "true" : "false");
            } else if constexpr (std::is_same_v<T, StringLiteral>) {
                out << '"' << value.value << '"';
            } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
                out << value.name;
            } else if constexpr (std::is_same_v<T, CallExpr>) {
                out << value.callee << '(';
                for (std::size_t index = 0; index < value.args.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    const auto& arg = value.args[index];
                    if (arg.name) {
                        out << *arg.name << ": ";
                    }
                    append_expr_summary(out, *arg.value);
                }
                out << ')';
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                append_expr_summary(out, *value.stage);
                out << '[';
                append_expr_summary(out, *value.count);
                out << ']';
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                out << operator_text(value.op);
                append_expr_summary(out, *value.operand);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                out << '(';
                append_expr_summary(out, *value.lhs);
                out << ' ' << operator_text(value.op) << ' ';
                append_expr_summary(out, *value.rhs);
                out << ')';
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                out << '(';
                append_expr_summary(out, *value.then_expr);
                out << " if ";
                append_expr_summary(out, *value.condition);
                out << " else ";
                append_expr_summary(out, *value.else_expr);
                out << ')';
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                out << '(';
                append_joined_exprs(out, value.elements);
                out << ')';
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                out << '[';
                append_joined_exprs(out, value.elements);
                out << ']';
            } else if constexpr (std::is_same_v<T, ArrowExpr>) {
                append_expr_summary(out, *value.source);
                for (const auto& stage : value.stages) {
                    out << " -> ";
                    append_expr_summary(out, *stage);
                }
            }
        },
        expr.kind
    );
}

void append_stmt_summary(std::ostringstream& out, const Stmt& stmt, std::size_t level) {
    out << indent(level);
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, ReturnStmt>) {
                out << "return ";
                append_expr_summary(out, *value.value);
                out << '\n';
            } else if constexpr (std::is_same_v<T, ExprStmt>) {
                out << "expr ";
                append_expr_summary(out, *value.value);
                out << '\n';
            } else if constexpr (std::is_same_v<T, VarDecl>) {
                out  << value.name << ": "
                    << type_to_string(value.type);
                if (value.init) {
                    out << " = ";
                    append_expr_summary(out, *value.init);
                }
                out << '\n';
            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                out << "assign " << value.name << " = ";
                append_expr_summary(out, *value.value);
                out << '\n';
            } else if constexpr (std::is_same_v<T, ScopeStmt>) {
                out << "scope statements=" << value.statements.size() << '\n';
                for (const auto& child : value.statements) {
                    append_stmt_summary(out, child, level + 2);
                }
            } else if constexpr (std::is_same_v<T, IfStmt>) {
                out << "if ";
                append_expr_summary(out, *value.condition);
                out << '\n';
                append_stmt_summary(out, *value.then_stmt, level + 2);
                for (const auto& branch : value.elifs) {
                    out << indent(level) << "elif ";
                    append_expr_summary(out, *branch.condition);
                    out << '\n';
                    append_stmt_summary(out, *branch.body, level + 2);
                }
                if (value.else_stmt) {
                    out << indent(level) << "else\n";
                    append_stmt_summary(out, *value.else_stmt, level + 2);
                }
            }
        },
        stmt.kind
    );
}

} // namespace

Type::Type(const Type& other)
    : base(other.base),
      elements(other.elements),
      scalar_dtype(other.scalar_dtype),
      tensor_dtype(other.tensor_dtype),
      tensor_shape_expr(other.tensor_shape_expr),
      tensor_rank(other.tensor_rank) {
    if (other.callable_return) {
        callable_return = std::make_unique<Type>(*other.callable_return);
    }
}

Type& Type::operator=(const Type& other) {
    if (this == &other) {
        return *this;
    }
    base = other.base;
    elements = other.elements;
    scalar_dtype = other.scalar_dtype;
    tensor_dtype = other.tensor_dtype;
    tensor_shape_expr = other.tensor_shape_expr;
    tensor_rank = other.tensor_rank;
    callable_return = other.callable_return ? std::make_unique<Type>(*other.callable_return) : nullptr;
    return *this;
}

Type Type::unknown() {
    Type type;
    type.base = TypeBase::Unknown;
    return type;
}

Type Type::int_type() {
    Type type;
    type.base = TypeBase::Int;
    return type;
}

Type Type::int16() {
    Type type = int_type();
    type.scalar_dtype = "int16";
    return type;
}

Type Type::int32() {
    Type type = int_type();
    type.scalar_dtype = "int32";
    return type;
}

Type Type::int64() {
    Type type = int_type();
    type.scalar_dtype = "int64";
    return type;
}

Type Type::float_type() {
    Type type;
    type.base = TypeBase::Float;
    return type;
}

Type Type::float16() {
    Type type = float_type();
    type.scalar_dtype = "float16";
    return type;
}

Type Type::float32() {
    Type type = float_type();
    type.scalar_dtype = "float32";
    return type;
}

Type Type::float64() {
    Type type = float_type();
    type.scalar_dtype = "float64";
    return type;
}

Type Type::bool_type() {
    Type type;
    type.base = TypeBase::Bool;
    return type;
}

Type Type::str_type() {
    Type type;
    type.base = TypeBase::Str;
    return type;
}

Type Type::None_type() {
    return Type{};
}

Type Type::tensor(
    std::optional<std::string> dtype,
    std::optional<std::string> shape_expr,
    std::optional<std::size_t> rank
) {
    Type type;
    type.base = TypeBase::Tensor;
    type.tensor_dtype = std::move(dtype);
    type.tensor_shape_expr = std::move(shape_expr);
    type.tensor_rank = rank;
    return type;
}

Type Type::tuple(std::vector<Type> elements) {
    Type type;
    type.base = TypeBase::Tuple;
    type.elements = std::move(elements);
    return type;
}

Type Type::list(std::vector<Type> elements) {
    Type type;
    type.base = TypeBase::List;
    type.elements = std::move(elements);
    return type;
}
// needed attention here.
Type Type::callable(std::optional<Type> return_type) {
    Type type;
    if (return_type.has_value()){
        type.callable_return = std::make_unique<Type>(std::move(return_type.value()));
    }
    type.base = TypeBase::Callable;
    return type;
}

Parser::Parser(std::vector<Token> tokens)
    : tokens_(std::move(tokens)), arena_(std::make_unique<Arena>()) {}

ExprPtr Parser::make_expr(SourceSpan span, ExprKind kind) {
    return arena_->allocate<Expr>(Expr{span, std::move(kind)});
}

std::optional<Diagnostic> Parser::take_last_diagnostic() {
    auto diagnostic = last_diagnostic_;
    last_diagnostic_.reset();
    return diagnostic;
}

/**
 * @brief Parses the entire token stream into a Program AST.
 * 
 * Top-level definitions can be layers, functions, configs, or statements.
 * @return A ParseResult containing the Program or a Diagnostic on failure.
 */
ParseResult Parser::parse_program() {
    try {
        Program program;
        while (const Token* token = peek(0)) {
            switch (token->kind) {
                case TokenType::Newline:
                case TokenType::Dedent:
                    consume();
                    break;
                case TokenType::Eof:
                    program.arena = std::move(arena_);
                    return program;
                case TokenType::Layer:
                    program.layers.push_back(parse_layer());
                    break;
                case TokenType::Fn:
                    program.functions.push_back(parse_function());
                    break;
                case TokenType::Config:
                    program.configs.push_back(parse_config());
                    break;
                case TokenType::Ident:
                    if (peek_kind(1) == TokenType::Eq) {
                        program.globals.push_back(parse_stmt());
                    } else if (peek_kind(1) == TokenType::Colon) {
                        fail_token("Top-level variable declarations must start with 'let'", *token);
                    } else {
                        fail_here("Unexpected token at top level");
                    }
                    break;
                default:
                    fail_here("Unexpected token at top level");
            }
        }
        program.arena = std::move(arena_);
        return program;
    } catch (const ParseFailure&) {
        return last_diagnostic_.value_or(Diagnostic::error("parser", "P0001", "Parser error"));
    }
}

/**
 * @brief Parses an expression, which handles comma-separated tuple expressions at the top level.
 */
ExprPtr Parser::parse_expression() {
    return parse_tuple_expression();
}

/**
 * @brief Parses a tuple expression if a comma is present.
 */
ExprPtr Parser::parse_tuple_expression() {
    auto first = parse_conditional_expression();
    if (peek_kind(0) != TokenType::Comma) {
        return first;
    }
    const SourceSpan span = first->span;
    std::vector<ExprPtr> elements;
    elements.push_back(std::move(first));
    while (peek_kind(0) == TokenType::Comma) {
        consume();
        elements.push_back(parse_conditional_expression());
    }
    return make_expr(span, TupleExpr{std::move(elements)});
}

/**
 * @brief Parses ternary conditional expressions (`A if B else C`).
 */
ExprPtr Parser::parse_conditional_expression() {
    auto expr = parse_pipeline_expression();
    if (peek_kind(0) != TokenType::If) {
        return expr;
    }
    Token if_token = consume();
    auto condition = parse_pipeline_expression();
    expect(TokenType::Else, "Expected 'else' in ternary expression");
    auto else_expr = parse_conditional_expression();
    return make_expr(
        span_of(if_token),
        TernaryExpr{std::move(expr), std::move(condition), std::move(else_expr)}
    );
}

/**
 * @brief Parses pipeline expressions (`A -> B -> C`).
 */
ExprPtr Parser::parse_pipeline_expression() {
    auto lhs = parse_logical_or_expression();
    if (peek_kind(0) != TokenType::Arrow) {
        return lhs;
    }
    const SourceSpan span = peek(0) ? span_of(*peek(0)) : SourceSpan{};
    std::vector<ExprPtr> stages;
    while (peek_kind(0) == TokenType::Arrow) {
        Token arrow = consume();
        auto rhs = parse_logical_or_expression();
        if (count_stage_sites(*rhs) != 1) {
            fail_token(
                "RHS of '->' must contain exactly one callable stage site. Use 'stage()[n]' for repetition.",
                arrow
            );
        }
        stages.push_back(std::move(rhs));
    }
    return make_expr(span, ArrowExpr{std::move(lhs), std::move(stages)});
}

ExprPtr Parser::parse_logical_or_expression() {
    auto lhs = parse_logical_and_expression();
    while (peek_kind(0) == TokenType::PipePipe) {
        Token op = consume();
        auto rhs = parse_logical_and_expression();
        lhs = make_expr(span_of(op), BinaryExpr{std::move(lhs), std::move(rhs), op.kind});
    }
    return lhs;
}

ExprPtr Parser::parse_logical_and_expression() {
    auto lhs = parse_comparison_expression();
    while (peek_kind(0) == TokenType::AmpAmp) {
        Token op = consume();
        auto rhs = parse_comparison_expression();
        lhs = make_expr(span_of(op), BinaryExpr{std::move(lhs), std::move(rhs), op.kind});
    }
    return lhs;
}

ExprPtr Parser::parse_comparison_expression() {
    auto lhs = parse_additive_expression();
    while (true) {
        auto kind = peek_kind(0);
        if (!(kind == TokenType::EqEq || kind == TokenType::Neq || kind == TokenType::Lt ||
              kind == TokenType::Gt || kind == TokenType::LtEq || kind == TokenType::GtEq)) {
            return lhs;
        }
        Token op = consume();
        auto rhs = parse_additive_expression();
        lhs = make_expr(span_of(op), BinaryExpr{std::move(lhs), std::move(rhs), op.kind});
    }
}

ExprPtr Parser::parse_additive_expression() {
    auto lhs = parse_multiplicative_expression();
    while (peek_kind(0) == TokenType::Plus || peek_kind(0) == TokenType::Minus) {
        Token op = consume();
        auto rhs = parse_multiplicative_expression();
        lhs = make_expr(span_of(op), BinaryExpr{std::move(lhs), std::move(rhs), op.kind});
    }
    return lhs;
}

ExprPtr Parser::parse_multiplicative_expression() {
    auto lhs = parse_unary_expression();
    while (peek_kind(0) == TokenType::Star || peek_kind(0) == TokenType::Slash ||
           peek_kind(0) == TokenType::DoubleSlash) {
        Token op = consume();
        auto rhs = parse_unary_expression();
        lhs = make_expr(span_of(op), BinaryExpr{std::move(lhs), std::move(rhs), op.kind});
    }
    return lhs;
}

ExprPtr Parser::parse_unary_expression() {
    if (peek_kind(0) == TokenType::Bang || peek_kind(0) == TokenType::Minus) {
        Token op = consume();
        auto operand = parse_unary_expression();
        return make_expr(span_of(op), UnaryExpr{std::move(operand), op.kind});
    }
    return parse_member_access_expression();
}

ExprPtr Parser::parse_member_access_expression() {
    auto lhs = parse_primary_expression();
    while (peek_kind(0) == TokenType::Dot) {
        Token op = consume();
        const Token* ident = peek(0);
        if (ident == nullptr) {
            fail_here("Expected identifier after '.'");
        }
        std::string name = consume_ident("Expected identifier after '.'");
        auto rhs = make_expr(span_of(*ident), IdentifierExpr{std::move(name)});
        lhs = make_expr(span_of(op), BinaryExpr{std::move(lhs), std::move(rhs), op.kind});
    }
    return lhs;
}

ExprPtr Parser::parse_primary_expression() {
    auto kind = peek_kind(0);
    if (!kind.has_value()) {
        fail_here("Expected expression");
    }

    ExprPtr result;
    switch (*kind) {
        case TokenType::IntLit: {
            Token token = consume();
            result = make_expr(span_of(token), IntLiteral{expect_int(token)});
            break;
        }
        case TokenType::FloatLit: {
            Token token = consume();
            result = make_expr(span_of(token), FloatLiteral{expect_float(token)});
            break;
        }
        case TokenType::True: {
            Token token = consume();
            result = make_expr(span_of(token), BoolLiteral{true});
            break;
        }
        case TokenType::False: {
            Token token = consume();
            result = make_expr(span_of(token), BoolLiteral{false});
            break;
        }
        case TokenType::StringLit: {
            Token token = consume();
            result = make_expr(span_of(token), StringLiteral{expect_string(token)});
            break;
        }
        case TokenType::Ident:
        case TokenType::Config:
        case TokenType::Int:
        case TokenType::Float:
        case TokenType::Bool:
        case TokenType::Layer:
        case TokenType::Fn: {
            const Token token = *peek(0);
            if (peek_kind(1) == TokenType::OpenParen) {
                std::string name = consume_ident("Expected callee");
                expect(TokenType::OpenParen, "Expected '('");
                std::vector<CallArgument> args;
                if (peek_kind(0) != TokenType::CloseParen) {
                    while (true) {
                        std::optional<std::string> arg_name;
                        if (peek_kind(1) == TokenType::Colon && peek_kind(0).has_value() &&
                            is_ident_like(*peek_kind(0))) {
                            arg_name = consume_ident("Expected named argument");
                            expect(TokenType::Colon, "Expected ':' after named argument");
                        }
                        args.push_back(CallArgument{std::move(arg_name), parse_conditional_expression()});
                        if (peek_kind(0) == TokenType::Comma) {
                            consume();
                        } else {
                            break;
                        }
                    }
                }
                expect(TokenType::CloseParen, "Expected ')' after function call arguments");
                result = make_expr(span_of(token), CallExpr{std::move(name), std::move(args)});
            } else {
                // here is we are parse the variable assignment
                std::string name = consume_ident("Expected identifier");
                result = make_expr(span_of(token), IdentifierExpr{std::move(name)});
            }
            break;
        }
        case TokenType::OpenParen: {
            consume();
            result = parse_expression();
            expect(TokenType::CloseParen, "Expected ')'");
            break;
        }
        case TokenType::OpenBracket:
            result = parse_list_expression();
            break;
        default:
            fail_here("Expected expression");
    }

    while (peek_kind(0) == TokenType::OpenBracket) {
        if (!std::holds_alternative<CallExpr>(result->kind)) {
            fail_token("Repeat suffix '[n]' must follow a call expression", *peek(0));
        }
        Token bracket = consume();
        auto repeat_count = parse_conditional_expression();
        expect(TokenType::CloseBracket, "Expected ']' after repetition count");
        result = make_expr(
            span_of(bracket),
            RepeatExpr{std::move(result), std::move(repeat_count)}
        );
    }

    return result;
}

ExprPtr Parser::parse_list_expression() {
    Token start = expect(TokenType::OpenBracket, "Expected '['");
    std::vector<ExprPtr> elements;
    if (peek_kind(0) != TokenType::CloseBracket) {
        while (true) {
            elements.push_back(parse_conditional_expression());
            if (peek_kind(0) == TokenType::Comma) {
                consume();
                if (peek_kind(0) == TokenType::CloseBracket) {
                    break;
                }
            } else {
                break;
            }
        }
    }
    expect(TokenType::CloseBracket, "Expected ']' after list literal");
    return make_expr(span_of(start), ListExpr{std::move(elements)});
}

Stmt Parser::parse_scope() {
    while (peek_kind(0) == TokenType::Newline) {
        consume();
    }
    Token indent_token = expect(TokenType::Indent, "Expected indentation for block");
    std::vector<Stmt> statements;
    while (peek_kind(0) != TokenType::Dedent) {
        if (peek_kind(0) == TokenType::Newline) {
            consume();
            continue;
        }
        if (peek_kind(0) == TokenType::Eof) {
            fail_here("Expected DEDENT");
        }
        statements.push_back(parse_stmt());
    }
    expect(TokenType::Dedent, "Expected DEDENT");
    return make_stmt(span_of(indent_token), ScopeStmt{std::move(statements)});
}

Type Parser::parse_type() {
    const Token* token = peek(0);
    if (token == nullptr) {
        fail_here("Expected type");
    }

    switch (token->kind) {
        case TokenType::Int:
            fail_token("Use an explicit integer width such as int16, int32, or int64", *token);
        case TokenType::Bool:
            consume();
            return Type::bool_type();
        case TokenType::Float:
            fail_token("Use an explicit float width such as float16, float32, or float64", *token);
        case TokenType::Str:
            consume();
            return Type::str_type();
        case TokenType::Callable:
            consume();
            return Type::callable(std::nullopt);
        case TokenType::Ident: {
            std::string name = expect_string(*token);
            consume();
            if (name == "int16") {
                return Type::int16();
            }
            if (name == "int32") {
                return Type::int32();
            }
            if (name == "int64") {
                return Type::int64();
            }
            if (name == "float16") {
                return Type::float16();
            }
            if (name == "float32") {
                return Type::float32();
            }
            if (name == "float64") {
                return Type::float64();
            }
            fail_token("Expected type", *token);
        }
        case TokenType::Tensor: {
            consume();
            if (peek_kind(0) != TokenType::OpenBracket) {
                return Type::tensor(std::nullopt, std::nullopt, std::nullopt);
            }
            consume();
            std::string first =
                parse_type_token_sequence({TokenType::Comma, TokenType::CloseBracket});
            std::optional<std::string> dtype;
            std::optional<std::string> shape_expr;
            if (peek_kind(0) == TokenType::Comma) {
                if (!first.empty()) {
                    dtype = first;
                }
                consume();
                shape_expr = parse_type_token_sequence({TokenType::CloseBracket});
            } else if (!first.empty()) {
                dtype = first;
            }
            expect(TokenType::CloseBracket, "Expected ']' after tensor type annotation");
            return Type::tensor(dtype, shape_expr, shape_expr ? derive_rank_from_shape_expr(*shape_expr) : std::nullopt);
        }
        case TokenType::List:
            consume();
            return parse_list_type_after_open("Expected '[' after list type");
        case TokenType::Tuple:
            consume();
            return parse_tuple_type_after_open(
                TokenType::OpenBracket,
                TokenType::CloseBracket,
                "Expected '[' after tuple type",
                "Expected ']' after tuple type"
            );
        case TokenType::OpenParen:
            return parse_tuple_type_after_open(
                TokenType::OpenParen,
                TokenType::CloseParen,
                "Expected '(' before tuple type",
                "Expected ')' after tuple type"
            );
        case TokenType::OpenBracket:
            return parse_list_type_after_open("Expected '[' before list type");
        default:
            fail_token("Expected type", *token);
    }
}

Type Parser::parse_list_type_after_open(const std::string& open_error) {
    expect(TokenType::OpenBracket, open_error);
    Type element_type = parse_type();
    if (peek_kind(0) == TokenType::Comma) {
        fail_here("List types take exactly one element type");
    }
    expect(TokenType::CloseBracket, "Expected ']' after list element type");
    std::vector<Type> elements;
    elements.push_back(std::move(element_type));
    return Type::list(std::move(elements));
}

Type Parser::parse_tuple_type_after_open(
    TokenType open,
    TokenType close,
    const std::string& open_error,
    const std::string& close_error
) {
    expect(open, open_error);
    std::vector<Type> elements = parse_type_list(close);
    expect(close, close_error);
    return Type::tuple(std::move(elements));
}

std::vector<Type> Parser::parse_type_list(TokenType terminator) {
    if (peek_kind(0) == terminator) {
        fail_here("Expected at least one type");
    }
    std::vector<Type> elements;
    elements.push_back(parse_type());
    while (peek_kind(0) == TokenType::Comma) {
        consume();
        if (peek_kind(0) == terminator) {
            break;
        }
        elements.push_back(parse_type());
    }
    return elements;
}

std::string Parser::parse_type_token_sequence(const std::vector<TokenType>& terminators) {
    std::string result;
    int bracket_depth = 0;
    while (const Token* token = peek(0)) {
        if (bracket_depth == 0 && contains_token(terminators, token->kind)) {
            break;
        }
        Token consumed = consume();
        switch (consumed.kind) {
            case TokenType::Ident:
            case TokenType::StringLit:
                result += expect_string(consumed);
                break;
            case TokenType::IntLit:
                result += std::to_string(expect_int(consumed));
                break;
            case TokenType::FloatLit: {
                std::ostringstream out;
                out << expect_float(consumed);
                result += out.str();
                break;
            }
            case TokenType::Int:
                result += "int";
                break;
            case TokenType::Float:
                result += "float";
                break;
            case TokenType::Bool:
                result += "bool";
                break;
            case TokenType::Tensor:
                result += "tensor";
                break;
            case TokenType::Colon:
                result += ':';
                break;
            case TokenType::Comma:
                result += ", ";
                break;
            case TokenType::Dot:
                result += '.';
                break;
            case TokenType::Plus:
                result += '+';
                break;
            case TokenType::Minus:
                result += '-';
                break;
            case TokenType::Star:
                result += '*';
                break;
            case TokenType::Slash:
                result += '/';
                break;
            case TokenType::DoubleSlash:
                result += "//";
                break;
            case TokenType::OpenBracket:
                result += '[';
                bracket_depth += 1;
                break;
            case TokenType::CloseBracket:
                result += ']';
                bracket_depth -= 1;
                break;
            default:
                fail_token("Unsupported token in tensor type annotation", consumed);
        }
    }
    return result;
}

Stmt Parser::parse_stmt() {
    const Token* token = peek(0);
    if (token == nullptr) {
        fail_here("Expected statement");
    }

    switch (token->kind) {
        case TokenType::Return: {
            Token start = consume();
            auto expr = parse_expression();
            consume_terminator();
            return make_stmt(span_of(start), ReturnStmt{std::move(expr)});
        }
        // case TokenType::Let:
        //     return parse_let_var_decl();
        case TokenType::Ident:
            if (peek_kind(1) == TokenType::Colon) {
                fail_token("Variable declarations must start with 'let'", *token);
            } else if (peek_kind(1) == TokenType::Eq) {
                Token start = *token;
                std::string name = consume_ident("Expected assignment target");
                consume();
                auto value = parse_expression();
                consume_terminator();
                return make_stmt(span_of(start), AssignStmt{std::move(name), std::move(value)});
            } else if (peek_kind(1) == TokenType::OpenParen) {
                Token start = *token;
                auto expr = parse_expression();
                consume_terminator();
                return make_stmt(span_of(start), ExprStmt{std::move(expr)});
            }
            fail_token("Unexpected identifier or missing assignment.", *token);
        // case TokenType::Mut:
        //     fail_token("Mutable declarations must start with 'let mut'", *token);
        case TokenType::Indent:
            return parse_scope();
        case TokenType::If: {
            Token start = consume();
            auto condition = parse_expression();
            expect(TokenType::Colon, "Expected ':' after if condition");
            auto then_stmt = std::make_unique<Stmt>(parse_scope());
            std::vector<IfBranch> elifs;
            while (peek_kind(0) == TokenType::Elif) {
                consume();
                auto elif_condition = parse_expression();
                expect(TokenType::Colon, "Expected ':' after elif condition");
                auto elif_body = std::make_unique<Stmt>(parse_scope());
                elifs.push_back(IfBranch{std::move(elif_condition), std::move(elif_body)});
            }
            StmtPtr else_stmt;
            if (peek_kind(0) == TokenType::Else) {
                consume();
                if (peek_kind(0) == TokenType::Colon) {
                    consume();
                }
                else_stmt = std::make_unique<Stmt>(parse_scope());
            }
            return make_stmt(
                span_of(start),
                IfStmt{
                    std::move(condition),
                    std::move(then_stmt),
                    std::move(elifs),
                    std::move(else_stmt),
                }
            );
        }
        default:
            fail_token("Unexpected token in statement", *token);
    }
}
// this fn parseing the let keyword and assign wheather it mutable or not.
// Stmt Parser::parse_let_var_decl() {
//     Token start = expect(TokenType::Let, "Expected 'let'");
//     bool is_mutable = false;
//     if (peek_kind(0) == TokenType::Mut) {
//         consume();
//         is_mutable = true;
//     }
//     std::string name = consume_ident("Expected variable name");
//     Type type = Type::unknown();
//     if (peek_kind(0) == TokenType::Colon) {
//         consume();
//         type = parse_type();
//     }
//     ExprPtr init;
//     if (peek_kind(0) == TokenType::Eq) {
//         consume();
//         init = parse_expression();
//     }
//     if (type.base == TypeBase::Unknown && !init) {
//         fail_token("Variable declaration needs a type annotation or initializer", start);
//     }
//     consume_terminator();
//     return make_stmt(
//         span_of(start),
//         VarDecl{std::move(name), std::move(type), std::move(init), std::nullopt, is_mutable}
//     );
// }

Layer Parser::parse_layer() {
    Token start = expect(TokenType::Layer, "Expected 'layer' keyword");
    std::string name = consume_ident("Expected layer name");
    auto args = parse_callable_args();
    Type return_type = parse_callable_return_type("Expected ':' after layer signature");
    Stmt body = parse_scope();
    return Layer{span_of(start), std::move(name), std::move(args), std::move(return_type), std::move(body)};
}

Function Parser::parse_function() {
    Token start = expect(TokenType::Fn, "Expected 'fn' keyword");
    std::string name = consume_ident("Expected function name");
    auto args = parse_callable_args();
    Type return_type = parse_callable_return_type("Expected ':' after function signature");
    Stmt body = parse_scope();
    return Function{span_of(start), std::move(name), std::move(args), std::move(return_type), std::move(body)};
}

Config Parser::parse_config() {
    Token start = expect(TokenType::Config, "Expected 'config' keyword");
    std::string name = consume_ident("Expected config name");
    expect(TokenType::Colon, "Expected ':' after config name");
    while (peek_kind(0) == TokenType::Newline) {
        consume();
    }

    std::vector<Field> fields;
    if (peek_kind(0) == TokenType::Indent) {
        consume();
        while (peek_kind(0) != TokenType::Dedent) {
            if (peek_kind(0) == TokenType::Newline) {
                consume();
                continue;
            }
            std::string field_name = consume_ident("Expected field name");
            Type type = Type::unknown();
            if (peek_kind(0) == TokenType::Colon) {
                consume();
                type = parse_type();
            }
            ExprPtr init = nullptr;
            if (peek_kind(0) == TokenType::Eq) {
                consume();
                init = parse_expression();
            }
            fields.push_back(Field{std::move(field_name), std::move(type), std::move(init)});
            consume_terminator();
        }
        expect(TokenType::Dedent, "Expected DEDENT");
    }
    return Config{span_of(start), std::move(name), std::move(fields)};
}

std::vector<Arg> Parser::parse_callable_args() {
    expect(TokenType::OpenParen, "Expected '('");
    std::vector<Arg> args;
    if (peek_kind(0) != TokenType::CloseParen) {
        while (true) {
            std::string arg_name = consume_ident("Expected argument name");
            Type type = Type::unknown();
            if (peek_kind(0) == TokenType::Colon) {
                consume();
                type = parse_type();
            }
            ExprPtr default_value = nullptr;
            if (peek_kind(0) == TokenType::Eq) {
                consume();
                default_value = parse_expression();
            }
            args.push_back(Arg{std::move(arg_name), std::move(type), std::move(default_value)});
            if (peek_kind(0) == TokenType::Comma) {
                consume();
            } else {
                break;
            }
        }
    }
    expect(TokenType::CloseParen, "Expected ')'");
    return args;
}

Type Parser::parse_callable_return_type(const std::string& colon_error) {
    expect(TokenType::Colon, colon_error);
    if (peek_kind(0) == TokenType::Newline || peek_kind(0) == TokenType::Indent) {
        return Type::unknown();
    }
    Type return_type = parse_type();
    expect(TokenType::Colon, "Expected ':' after return type");
    return return_type;
}

const Token* Parser::peek(std::size_t offset) const {
    const std::size_t position = index_ + offset;
    if (position >= tokens_.size()) {
        return nullptr;
    }
    return &tokens_[position];
}

std::optional<TokenType> Parser::peek_kind(std::size_t offset) const {
    const Token* token = peek(offset);
    if (token == nullptr) {
        return std::nullopt;
    }
    return token->kind;
}

Token Parser::consume() {
    if (index_ >= tokens_.size()) {
        fail_here("Unexpected end of tokens");
    }
    return tokens_[index_++];
}

Token Parser::expect(TokenType kind, const std::string& message) {
    const Token* token = peek(0);
    if (token == nullptr) {
        fail_here(message);
    }
    if (token->kind != kind) {
        fail_token(message, *token);
    }
    return consume();
}

std::string Parser::consume_ident(const std::string& message) {
    Token token = consume();
    switch (token.kind) {
        case TokenType::Ident:
            return expect_string(token);
        case TokenType::Config:
            return "config";
        case TokenType::True:
            return "true";
        case TokenType::False:
            return "false";
        case TokenType::Int:
            return "int";
        case TokenType::Float:
            return "float";
        case TokenType::Bool:
            return "bool";
        case TokenType::Layer:
            return "layer";
        case TokenType::Fn:
            return "fn";
        case TokenType::If:
            return "if";
        case TokenType::Else:
            return "else";
        case TokenType::Elif:
            return "elif";
        case TokenType::Return:
            return "return";
        default:
            fail_token(message, token);
    }
}

void Parser::consume_terminator() {
    bool found = false;
    while (peek_kind(0) == TokenType::Semi || peek_kind(0) == TokenType::Newline) {
        consume();
        found = true;
    }
    if (!found && !(peek_kind(0) == TokenType::Dedent || peek_kind(0) == TokenType::Eof)) {
        fail_here("Expected ';' or newline at end of statement");
    }
}

void Parser::fail_here(const std::string& message) {
    if (const Token* token = peek(0)) {
        fail_token(message, *token);
    }
    if (!tokens_.empty()) {
        const Token& last = tokens_.back();
        record(
            Diagnostic::error("parser", "P0001", message)
                .with_span(last.line, last.column)
                .with_help(
                    "Reached end of file after " + std::to_string(last.line) + ":" +
                    std::to_string(last.column)
                )
        );
    } else {
        record(
            Diagnostic::error("parser", "P0001", message)
                .with_help("The parser did not receive any tokens")
        );
    }
    throw ParseFailure(message);
}

void Parser::fail_token(const std::string& message, const Token& token) {
    record(Diagnostic::error("parser", "P0001", message).with_span(token.line, token.column));
    throw ParseFailure(message);
}

void Parser::record(Diagnostic diagnostic) {
    last_diagnostic_ = std::move(diagnostic);
}

std::string type_to_string(const Type& type) {
    switch (type.base) {
        case TypeBase::Unknown:
            return "unknown";
        case TypeBase::Bool:
            return "bool";
        case TypeBase::Int:
            return type.scalar_dtype.value_or("int");
        case TypeBase::Str:
            return "str";
        case TypeBase::Float:
            return type.scalar_dtype.value_or("float");
        case TypeBase::Tensor:
            if (type.tensor_dtype || type.tensor_shape_expr) {
                std::string result = "tensor[";
                if (type.tensor_dtype) {
                    result += *type.tensor_dtype;
                }
                if (type.tensor_shape_expr) {
                    result += ", " + *type.tensor_shape_expr;
                }
                result += ']';
                return result;
            }
            return "tensor";
        case TypeBase::Tuple: {
            std::string result = "tuple[";
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    result += ", ";
                }
                result += type_to_string(type.elements[index]);
            }
            result += ']';
            return result;
        }
        case TypeBase::List:
            return type.elements.empty() ? "list[unknown]" : "list[" + type_to_string(type.elements[0]) + "]";
        case TypeBase::Callable:
            return type.callable_return ? "callable -> " + type_to_string(*type.callable_return)
                                        : "callable";
        case TypeBase::None: // need attention
            return "None";
    }
    return "unknown";
}

std::string program_summary(const Program& program) {
    std::ostringstream out;
    out << "configs=" << program.configs.size() << '\n';
    out << "layers=" << program.layers.size() << '\n';
    out << "functions=" << program.functions.size() << '\n';
    out << "globals=" << program.globals.size();
    return out.str();
}

std::string ast_to_string(const Program& program) {
    std::ostringstream out;
    out << program_summary(program) << '\n';
    for (const auto& config : program.configs) {
        out << "config " << config.name << " fields=" << config.fields.size() << '\n';
        for (const auto& field : config.fields) {
            out << "  field " << field.name << ": " << type_to_string(field.type);
            if (field.init) {
                out << " = ";
                append_expr_summary(out, *field.init);
            }
            out << '\n';
        }
    }
    for (const auto& layer : program.layers) {
        out << "layer " << layer.name << '(';
        append_args(out, layer.args);
        out << ") -> " << type_to_string(layer.return_type) << '\n';
        append_stmt_summary(out, layer.body, 2);
    }
    for (const auto& function : program.functions) {
        out << "fn " << function.name << '(';
        append_args(out, function.args);
        out << ") -> " << type_to_string(function.return_type) << '\n';
        append_stmt_summary(out, function.body, 2);
    }
    for (const auto& global : program.globals) {
        append_stmt_summary(out, global, 0);
    }
    return out.str();
}
