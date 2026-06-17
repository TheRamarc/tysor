#include "lexer.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <iostream>

namespace {

void push_token(
    std::vector<Token>& tokens,
    TokenType kind,
    TokenValue value,
    std::size_t line,
    std::size_t column
) {
    tokens.push_back(Token{kind, std::move(value), line, column});
}

Diagnostic lexer_error(std::string message, std::size_t line, std::size_t column) {
    return Diagnostic::error("lexer", "L0001", std::move(message)).with_span(line, column);
}

bool is_identifier_start(char ch) {
    return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

bool is_identifier_continue(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

bool is_digit(char ch) {
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

std::string token_value_to_string(const TokenValue& value) {
    if (std::holds_alternative<std::monostate>(value)) {
        return {};
    }
    if (const auto* int_value = std::get_if<std::int64_t>(&value)) {
        return std::to_string(*int_value);
    }
    if (const auto* float_value = std::get_if<double>(&value)) {
        std::ostringstream out;
        out << *float_value;
        return out.str();
    }
    return std::get<std::string>(value);
}

const std::unordered_map<std::string, TokenType>& keywords() {
    static const std::unordered_map<std::string, TokenType> table = {
        {"return", TokenType::Return},
        {"int", TokenType::Int},
        {"bool", TokenType::Bool},
        {"float", TokenType::Float},
        {"tensor", TokenType::Tensor},
        {"tuple", TokenType::Tuple},
        {"list", TokenType::List},
        {"true", TokenType::True},
        {"false", TokenType::False},
        {"layer", TokenType::Layer},
        {"fn", TokenType::Fn},
        {"config", TokenType::Config},
        {"let", TokenType::Let},
        {"mut", TokenType::Mut},
        {"if", TokenType::If},
        {"elif", TokenType::Elif},
        {"else", TokenType::Else},
        {"while", TokenType::While},
        {"for", TokenType::For},
    };
    return table;
}

} // namespace

const char* token_type_name(TokenType kind) {
    switch (kind) {
        case TokenType::Return:
            return "RETURN";
        case TokenType::Int:
            return "INT";
        case TokenType::Float:
            return "FLOAT";
        case TokenType::Bool:
            return "BOOL";
        case TokenType::Tensor:
            return "TENSOR";
        case TokenType::Tuple:
            return "TUPLE";
        case TokenType::List:
            return "LIST";
        case TokenType::True:
            return "TRUE";
        case TokenType::False:
            return "FALSE";
        case TokenType::Layer:
            return "LAYER";
        case TokenType::Fn:
            return "FN";
        case TokenType::Config:
            return "CONFIG";
        case TokenType::Let:
            return "LET";
        case TokenType::Mut:
            return "MUT";
        case TokenType::If:
            return "IF";
        case TokenType::Elif:
            return "ELIF";
        case TokenType::Else:
            return "ELSE";
        case TokenType::While:
            return "WHILE";
        case TokenType::For:
            return "FOR";
        case TokenType::IntLit:
            return "INT_LIT";
        case TokenType::FloatLit:
            return "FLOAT_LIT";
        case TokenType::BoolLit:
            return "BOOL_LIT";
        case TokenType::StringLit:
            return "STRING_LIT";
        case TokenType::Ident:
            return "IDENT";
        case TokenType::Plus:
            return "PLUS";
        case TokenType::Minus:
            return "MINUS";
        case TokenType::Star:
            return "STAR";
        case TokenType::Slash:
            return "SLASH";
        case TokenType::DoubleSlash:
            return "DOUBLE_SLASH";
        case TokenType::EqEq:
            return "EQ_EQ";
        case TokenType::Neq:
            return "NEQ";
        case TokenType::Lt:
            return "LT";
        case TokenType::Gt:
            return "GT";
        case TokenType::LtEq:
            return "LT_EQ";
        case TokenType::GtEq:
            return "GT_EQ";
        case TokenType::Amp:
            return "AMP";
        case TokenType::AmpAmp:
            return "AMP_AMP";
        case TokenType::PipePipe:
            return "PIPE_PIPE";
        case TokenType::Bang:
            return "BANG";
        case TokenType::Eq:
            return "EQUALS";
        case TokenType::Hash:
            return "HASH";
        case TokenType::Underscore:
            return "UNDERSCORE";
        case TokenType::OpenParen:
            return "OPEN_PAREN";
        case TokenType::CloseParen:
            return "CLOSE_PAREN";
        case TokenType::OpenBracket:
            return "OPEN_BRACKET";
        case TokenType::CloseBracket:
            return "CLOSE_BRACKET";
        case TokenType::Semi:
            return "SEMI";
        case TokenType::Colon:
            return "COLON";
        case TokenType::Comma:
            return "COMMA";
        case TokenType::Dot:
            return "DOT";
        case TokenType::DotDot:
            return "DOT_DOT";
        case TokenType::Pipe:
            return "PIPE";
        case TokenType::Arrow:
            return "ARROW";
        case TokenType::StarEq:
            return "STAR_EQ";
        case TokenType::Indent:
            return "INDENT";
        case TokenType::Dedent:
            return "DEDENT";
        case TokenType::Newline:
            return "NEWLINE";
        case TokenType::Eof:
            return "EOF";
    }
    return "EOF";
}

std::string token_to_string(const Token& token) {
    std::ostringstream out;
    out << token_type_name(token.kind);
    if (!std::holds_alternative<std::monostate>(token.value)) {
        out << '(' << token_value_to_string(token.value) << ')';
    }
    out << " (" << token.line << ':' << token.column << ')';
    return out.str();
}

TokenizeResult tokenize_with_diagnostic(const std::string& source) {
    std::vector<Token> tokens;
    std::size_t line = 1;
    std::size_t column = 1;
    std::vector<std::size_t> indent_stack = {0};
    bool start_of_line = true;
    std::size_t paren_level = 0;
    std::size_t index = 0;

    while (index < source.size()) {
        if (start_of_line) {
            std::size_t current_indent = 0;
            while (index < source.size() && (source[index] == ' ' || source[index] == '\t')) {
                if (source[index] == ' ') {
                    current_indent += 1;
                } else {
                    current_indent = (current_indent / 8 + 1) * 8;
                }
                index += 1;
            }

            if (index < source.size() && source[index] != '\n' && source[index] != '#') {
                const bool is_continuation =
                    source[index] == '-' && index + 1 < source.size() && source[index + 1] == '>';
                if (paren_level == 0 && !is_continuation) {
                    if (current_indent > indent_stack.back()) {
                        indent_stack.push_back(current_indent);
                        push_token(
                            tokens,
                            TokenType::Indent,
                            std::monostate{},
                            line,
                            current_indent + 1
                        );

                    } else {
                        while (current_indent < indent_stack.back()) {
                            indent_stack.pop_back();
                            push_token(
                                tokens,
                                TokenType::Dedent,
                                std::monostate{},
                                line,
                                current_indent + 1
                            );
                        }
                        if (current_indent != indent_stack.back()) {
                            return lexer_error("Indentation error", line, current_indent + 1);
                        }
                    }
                }
            }

            column = current_indent + 1;
            start_of_line = false;
        }

        if (index >= source.size()) {
            break;
        }

        const char ch = source[index];
        const std::size_t start_column = column;

        if (is_identifier_start(ch)) {
            std::string text;
            text.push_back(ch);
            while (index + 1 < source.size() && is_identifier_continue(source[index + 1])) {
                index += 1;
                column += 1;
                text.push_back(source[index]);
            }

            const auto keyword = keywords().find(text);
            if (keyword != keywords().end()) {
                push_token(tokens, keyword->second, std::monostate{}, line, start_column);
            } else if (text == "None") {
                push_token(tokens, TokenType::Ident, std::string("None"), line, start_column);
            } else {
                push_token(tokens, TokenType::Ident, std::move(text), line, start_column);
            }
            index += 1;
            column += 1;
            continue;
        }

        if (ch == '"') {
            std::string text;
            index += 1;
            column += 1;
            while (index < source.size() && source[index] != '"') {
                if (source[index] == '\n') {
                    return lexer_error("Unterminated string literal", line, column);
                }
                text.push_back(source[index]);
                index += 1;
                column += 1;
            }
            if (index >= source.size()) {
                return lexer_error("Unterminated string literal", line, column);
            }
            push_token(tokens, TokenType::StringLit, std::move(text), line, start_column);
            index += 1;
            column += 1;
            continue;
        }

        if (is_digit(ch)) {
            std::string text;
            text.push_back(ch);
            while (index + 1 < source.size() && is_digit(source[index + 1])) {
                index += 1;
                column += 1;
                text.push_back(source[index]);
            }

            bool is_float = false;
            if (index + 1 < source.size() && source[index + 1] == '.') {
                is_float = true;
                index += 1;
                column += 1;
                text.push_back(source[index]);
                while (index + 1 < source.size() && is_digit(source[index + 1])) {
                    index += 1;
                    column += 1;
                    text.push_back(source[index]);
                }
            }

            if (index + 1 < source.size() &&
                (source[index + 1] == 'e' || source[index + 1] == 'E')) {
                is_float = true;
                index += 1;
                column += 1;
                text.push_back(source[index]);
                if (index + 1 < source.size() &&
                    (source[index + 1] == '+' || source[index + 1] == '-')) {
                    index += 1;
                    column += 1;
                    text.push_back(source[index]);
                }
                while (index + 1 < source.size() && is_digit(source[index + 1])) {
                    index += 1;
                    column += 1;
                    text.push_back(source[index]);
                }
            }

            try {
                if (is_float) {
                    push_token(tokens, TokenType::FloatLit, std::stod(text), line, start_column);
                } else {
                    push_token(
                        tokens,
                        TokenType::IntLit,
                        static_cast<std::int64_t>(std::stoll(text)),
                        line,
                        start_column
                    );
                }
            } catch (const std::exception& error) {
                return lexer_error(
                    std::string("Invalid numeric literal '") + text + "': " + error.what(),
                    line,
                    start_column
                );
            }

            index += 1;
            column += 1;
            continue;
        }

        switch (ch) {
            case ';':
                push_token(tokens, TokenType::Semi, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case ':':
                push_token(tokens, TokenType::Colon, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case '.':
                if (index + 1 < source.size() && source[index + 1] == '.') {
                    push_token(tokens, TokenType::DotDot, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Dot, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '=':
                if (index + 1 < source.size() && source[index + 1] == '=') {
                    push_token(tokens, TokenType::EqEq, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Eq, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '!':
                if (index + 1 < source.size() && source[index + 1] == '=') {
                    push_token(tokens, TokenType::Neq, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Bang, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '&':
                if (index + 1 < source.size() && source[index + 1] == '&') {
                    push_token(tokens, TokenType::AmpAmp, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Amp, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '|':
                if (index + 1 < source.size() && source[index + 1] == '|') {
                    push_token(tokens, TokenType::PipePipe, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Pipe, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '<':
                if (index + 1 < source.size() && source[index + 1] == '=') {
                    push_token(tokens, TokenType::LtEq, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Lt, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '>':
                if (index + 1 < source.size() && source[index + 1] == '=') {
                    push_token(tokens, TokenType::GtEq, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Gt, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '+':
                push_token(tokens, TokenType::Plus, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case '-':
                if (index + 1 < source.size() && source[index + 1] == '>') {
                    push_token(tokens, TokenType::Arrow, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Minus, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '*':
                if (index + 1 < source.size() && source[index + 1] == '=') {
                    push_token(tokens, TokenType::StarEq, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Star, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '/':
                if (index + 1 < source.size() && source[index + 1] == '/') {
                    push_token(tokens, TokenType::DoubleSlash, std::monostate{}, line, column);
                    index += 2;
                    column += 2;
                } else {
                    push_token(tokens, TokenType::Slash, std::monostate{}, line, column);
                    index += 1;
                    column += 1;
                }
                break;
            case '#':
                while (index < source.size() && source[index] != '\n') {
                    index += 1;
                    column += 1;
                }
                break;
            case '(':
                paren_level += 1;
                push_token(tokens, TokenType::OpenParen, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case ')':
                if (paren_level > 0) {
                    paren_level -= 1;
                }
                push_token(tokens, TokenType::CloseParen, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case '[':
                paren_level += 1;
                push_token(tokens, TokenType::OpenBracket, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case ']':
                if (paren_level > 0) {
                    paren_level -= 1;
                }
                push_token(tokens, TokenType::CloseBracket, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case ',':
                push_token(tokens, TokenType::Comma, std::monostate{}, line, column);
                index += 1;
                column += 1;
                break;
            case '\n': {
                if (paren_level == 0) {
                    bool next_is_continuation = false;
                    std::size_t next = index + 1;
                    while (next < source.size() && (source[next] == ' ' || source[next] == '\t')) {
                        next += 1;
                    }
                    if (next + 1 < source.size() && source[next] == '-' && source[next + 1] == '>') {
                        next_is_continuation = true;
                    }
                    if (!next_is_continuation) {
                        push_token(tokens, TokenType::Newline, std::monostate{}, line, column);
                    }
                }
                line += 1;
                column = 1;
                start_of_line = true;
                index += 1;
                break;
            }
            case ' ':
            case '\t':
            case '\r':
                column += 1;
                index += 1;
                break;
            default:
                if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
                    column += 1;
                    index += 1;
                    break;
                }
                return lexer_error(std::string("Unknown character '") + ch + "'", line, column);
        }
    }

    while (indent_stack.size() > 1) {
        indent_stack.pop_back();
        push_token(tokens, TokenType::Dedent, std::monostate{}, line, column);
    }
    push_token(tokens, TokenType::Eof, std::monostate{}, line, column);
    return tokens;
}

std::vector<Token> tokenize(const std::string& source) {
    TokenizeResult result = tokenize_with_diagnostic(source);
    if (const auto* tokens = std::get_if<std::vector<Token>>(&result)) {
        return *tokens;
    }
    throw std::runtime_error(std::get<Diagnostic>(result).to_string());
}

