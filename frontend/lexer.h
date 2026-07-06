#pragma once

#include "diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

/**
 * @brief Represents the kind of token produced by the lexer.
 * 
 * The lexer emits both language tokens (keywords, literals, operators) and 
 * layout tokens (Indent, Dedent, Newline). This allows the parser to treat 
 * significant whitespace similarly to regular syntax (like braces in C++).
 */
enum class TokenType {
    // Keywords
    Return,         // "return"
    Int,            // "int"
    Float,          // "float"
    Str,            // "str"
    Bool,           // "bool"
    Tensor,         // "tensor"
    Tuple,          // "tuple"
    List,           // "list"
    True,           // "true"
    False,          // "false"
    Callable,       // "callable"
    Layer,          // "layer"
    Fn,             // "fn"
    Config,         // "config"
    If,             // "if"
    Elif,           // "elif"
    Else,           // "else"
    While,          // "while"
    For,            // "for"
    
    // Literals and Identifiers
    IntLit,         // Integer literals, e.g., 42
    FloatLit,       // Floating point literals, e.g., 3.14
    StringLit,      // String literals, e.g., "hello"
    Ident,          // Identifiers, e.g., my_var
    
    // Operators and Symbols
    Plus,           // +
    Minus,          // -
    Star,           // *
    Slash,          // /
    DoubleSlash,    // //
    EqEq,           // ==
    Neq,            // !=
    Lt,             // <
    Gt,             // >
    LtEq,           // <=
    GtEq,           // >=
    Amp,            // &
    AmpAmp,         // &&
    PipePipe,       // ||
    Bang,           // !
    Eq,             // =
    
    // Punctuation
    Hash,           // #
    Underscore,     // _
    OpenParen,      // (
    CloseParen,     // )
    OpenBracket,    // [
    CloseBracket,   // ]
    Semi,           // ;
    Colon,          // :
    Comma,          // ,
    Dot,            // .
    DotDot,         // ..
    Pipe,           // |
    Arrow,          // ->
    StarEq,         // *=
    
    // Layout and Control
    Indent,         // Block indentation increase
    Dedent,         // Block indentation decrease
    Newline,        // \n
    None,           // None
    Eof             // End of file
};

/**
 * @brief Represents the payload value associated with a token.
 * 
 * Tokens own their literal or identifier text even though tokenization reads through a
 * string_view. This ensures parser tokens never depend on the source buffer's lifetime.
 * 
 * - std::monostate: Tokens with no associated value (e.g., keywords, operators).
 * - std::int64_t: Integer literals.
 * - double: Floating-point literals.
 * - std::string: Identifiers and string literals.
 */
using TokenValue = std::variant<std::monostate, std::int64_t, double, std::string>;

/**
 * @brief Represents a single lexical token.
 * 
 * A token is the fundamental unit of syntax produced by the lexer and consumed by the parser.
 */
struct Token {
    /// The type of the token.
    TokenType kind;
    /// The specific value of the token (if applicable).
    TokenValue value;
    /// The line number where this token starts (1-indexed).
    std::size_t line;
    /// The column number where this token starts (1-indexed).
    std::size_t column;
};

/**
 * @brief The result of a tokenization pass.
 * 
 * Contains either a successful vector of tokens or an error Diagnostic.
 */
using TokenizeResult = std::variant<std::vector<Token>, Diagnostic>;

/**
 * @brief Converts a TokenType to a string representation for debugging.
 * 
 * @param kind The token type.
 * @return A constant character pointer representing the token's name.
 */
const char* token_type_name(TokenType kind);

/**
 * @brief Converts a Token to a detailed string, including its value and location.
 * 
 * @param token The token to convert.
 * @return A string representing the token.
 */
std::string token_to_string(const Token& token);

/**
 * @brief Tokenizes the provided source code, returning diagnostics on error.
 * 
 * This is the preferred entry point for compiler stages: errors stay as Diagnostics instead
 * of throwing exceptions, which keeps frontend failures reportable to the CLI or tests.
 * 
 * @param source The source code string to tokenize.
 * @return A TokenizeResult containing either the tokens or an error.
 */
TokenizeResult tokenize_with_diagnostic(std::string_view source);

/**
 * @brief Tokenizes the provided source code, throwing on error.
 * 
 * @param source The source code string to tokenize.
 * @return A vector of parsed tokens.
 * @throws std::runtime_error If tokenization fails.
 */
std::vector<Token> tokenize(std::string_view source);
