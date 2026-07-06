#pragma once

#include "diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// The lexer emits both language tokens and layout tokens. Indent/Dedent/Newline
// let the parser treat significant whitespace like regular syntax.
enum class TokenType {
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
    // Let,
    // Mut,
    If,             // "if"
    Elif,           // "elif"
    Else,           // "else"
    While,          // "while"
    For,            // "for"
    
    IntLit,         // e.g., 42
    FloatLit,       // e.g., 3.14
    // BoolLit,
    StringLit,      // e.g., "hello"
    Ident,          // e.g., my_var
    
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
    
    Indent,         // Block indentation increase
    Dedent,         // Block indentation decrease
    Newline,        // \n
    Eof             // End of file
};

// Tokens own literal/identifier text even though tokenization reads through a
// string_view, so parser tokens never depend on the source buffer lifetime.
using TokenValue = std::variant<std::monostate, std::int64_t, double, std::string>;

struct Token {
    TokenType kind;
    TokenValue value;
    std::size_t line;
    std::size_t column;
};

using TokenizeResult = std::variant<std::vector<Token>, Diagnostic>;

const char* token_type_name(TokenType kind);
std::string token_to_string(const Token& token);

// Preferred entry point for compiler stages: errors stay as Diagnostics instead
// of throwing, which keeps frontend failures reportable to the CLI/tests.
TokenizeResult tokenize_with_diagnostic(std::string_view source);
std::vector<Token> tokenize(std::string_view source);
