#include "lexer.h"
#include "parser.h"

#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

bool parse_ok(const std::string& name, const std::string& source, const std::string& expected) {
    TokenizeResult tokenized = tokenize_with_diagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        std::cerr << name << ": tokenization failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parse_program();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        std::cerr << name << ": parse failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    const std::string ast = ast_to_string(std::get<Program>(parsed));
    if (!expected.empty() && ast.find(expected) == std::string::npos) {
        std::cerr << name << ": AST did not contain expected text\n";
        std::cerr << "expected: " << expected << '\n';
        std::cerr << ast << '\n';
        return false;
    }
    return true;
}

bool parse_fails(const std::string& name, const std::string& source, const std::string& expected) {
    TokenizeResult tokenized = tokenize_with_diagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        std::cerr << name << ": tokenization failed before parser: " << diagnostic->to_string()
                  << '\n';
        return false;
    }

    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parse_program();
    const auto* diagnostic = std::get_if<Diagnostic>(&parsed);
    if (diagnostic == nullptr) {
        std::cerr << name << ": parse unexpectedly succeeded\n";
        return false;
    }
    if (diagnostic->message.find(expected) == std::string::npos) {
        std::cerr << name << ": wrong parser error\n";
        std::cerr << "expected: " << expected << '\n';
        std::cerr << diagnostic->to_string() << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks = {
        parse_ok(
            "nested-call",
            "layer Model(x: tensor[float16]): tensor[float16]:\n"
            "  return rsqrt(sqrt(x))\n",
            "return rsqrt(sqrt(x))"
        ),
        parse_ok(
            "arrow-stage-with-call-argument",
            "layer Model(x):\n"
            "  return x -> reshape(heads: heads())\n",
            "return x -> reshape(heads: heads())"
        ),
        parse_ok(
            "control-flow",
            "layer Model(x):\n"
            "  if ready(x):\n"
            "    return x\n"
            "  else:\n"
            "    return fallback(x)\n",
            "else\n"
        ),
        parse_fails(
            "old-global-declaration",
            "x: tensor[float16]\n",
            "Top-level variable declarations must start with 'let'"
        ),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
