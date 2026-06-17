#pragma once

#include "diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

enum class TokenType {
    Return,
    Int,
    Float,
    str,
    Bool,
    Tensor,
    Tuple,
    List,
    True,
    False,
    Layer,
    Fn,
    Config,
    Let,
    Mut,
    If,
    Elif,
    Else,
    While,
    For,
    IntLit,
    FloatLit,
    BoolLit,
    StringLit,
    Ident,
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    EqEq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Amp,
    AmpAmp,
    PipePipe,
    Bang,
    Eq,
    Hash,
    Underscore,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    Semi,
    Colon,
    Comma,
    Dot,
    DotDot,
    Pipe,
    Arrow,
    StarEq,
    Indent,
    Dedent,
    Newline,
    Eof,
};

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

TokenizeResult tokenize_with_diagnostic(const std::string& source);
std::vector<Token> tokenize(const std::string& source);
