#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

std::variant<Program, Diagnostic> parse_program(const std::string& source) {
    TokenizeResult tokenized = tokenize_with_diagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        return *diagnostic;
    }

    Parser parser(std::get<std::vector<Token>>(std::move(tokenized)));
    ParseResult parsed = parser.parse_program();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }
    return std::get<Program>(std::move(parsed));
}

bool analyze_ok(const std::string& name, const std::string& source, std::size_t min_calls) {
    auto parsed = parse_program(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        std::cerr << name << ": parse failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    SemanticAnalyzer analyzer;
    SemanticResult result = analyzer.analyze_with_info(std::get<Program>(parsed));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        std::cerr << name << ": semantic analysis failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    const SemanticInfo& info = std::get<SemanticInfo>(result);
    if (info.calls.size() < min_calls) {
        std::cerr << name << ": expected at least " << min_calls << " call(s), saw "
                  << info.calls.size() << '\n';
        return false;
    }
    return true;
}

bool analyze_fails(const std::string& name, const std::string& source, const std::string& expected) {
    auto parsed = parse_program(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        std::cerr << name << ": parse failed before semantic check: " << diagnostic->to_string()
                  << '\n';
        return false;
    }

    SemanticAnalyzer analyzer;
    SemanticResult result = analyzer.analyze_with_info(std::get<Program>(parsed));
    const auto* diagnostic = std::get_if<Diagnostic>(&result);
    if (diagnostic == nullptr) {
        std::cerr << name << ": semantic analysis unexpectedly succeeded\n";
        return false;
    }
    if (diagnostic->message.find(expected) == std::string::npos) {
        std::cerr << name << ": wrong semantic diagnostic\n";
        std::cerr << "expected: " << expected << '\n';
        std::cerr << diagnostic->to_string() << '\n';
        return false;
    }
    return true;
}

bool semantic_info_records_core_facts() {
    auto parsed = parse_program(
        "config settings:\n"
        "  depth: int32 = 2\n"
        "\n"
        "let global_step: int32 = 0\n"
        "\n"
        "layer model(mut x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
        "  let mut y = matmul(x, w)\n"
        "  y = relu(y)\n"
        "  let z: int32 = settings.depth\n"
        "  return y\n"
    );
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        std::cerr << "semantic-info-core: parse failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    SemanticAnalyzer analyzer;
    SemanticResult result = analyzer.analyze_with_info(std::get<Program>(parsed));
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        std::cerr << "semantic-info-core: analysis failed: " << diagnostic->to_string() << '\n';
        return false;
    }

    const SemanticInfo& info = std::get<SemanticInfo>(result);
    const auto has_symbol = [&](const std::string& name, SemanticSymbolKind kind) {
        return std::any_of(info.symbols.begin(), info.symbols.end(), [&](const SemanticSymbol& symbol) {
            return symbol.name == name && symbol.kind == kind;
        });
    };
    const auto has_call = [&](const std::string& callee, SemanticCallTargetKind target) {
        return std::any_of(info.calls.begin(), info.calls.end(), [&](const SemanticCallInfo& call) {
            return call.callee == callee && call.target == target;
        });
    };
    const auto has_config_access = std::any_of(
        info.config_field_accesses.begin(),
        info.config_field_accesses.end(),
        [](const SemanticConfigFieldAccessInfo& access) {
            return access.config_name == "settings" && access.field_name == "depth" &&
                   access.field_type.base == TypeBase::Int;
        }
    );
    const auto has_assignment = std::any_of(
        info.assignments.begin(),
        info.assignments.end(),
        [](const SemanticAssignmentInfo& assignment) {
            return assignment.target_name == "y" && assignment.mutable_symbol &&
                   assignment.target_type.base == TypeBase::Tensor;
        }
    );

    if (!has_symbol("matmul", SemanticSymbolKind::BuiltinFunction) ||
        !has_symbol("settings", SemanticSymbolKind::Config) ||
        !has_symbol("global_step", SemanticSymbolKind::Global) ||
        !has_symbol("x", SemanticSymbolKind::Parameter) ||
        !has_symbol("y", SemanticSymbolKind::Local) ||
        !has_call("matmul", SemanticCallTargetKind::BuiltinFunction) ||
        !has_call("relu", SemanticCallTargetKind::BuiltinFunction) ||
        !has_config_access || !has_assignment) {
        std::cerr << "semantic-info-core: semantic info missed an expected fact\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    const std::vector<bool> checks = {
        semantic_info_records_core_facts(),
        analyze_ok(
            "matmul-relu",
            "layer model(x: tensor[float16], w: tensor[float16]): tensor[float16]:\n"
            "  let y = matmul(x, w)\n"
            "  return relu(y)\n",
            2
        ),
        analyze_ok(
            "arrow-repeat",
            "layer model(x: tensor[float16]): tensor[float16]:\n"
            "  return x -> relu()[2] * 2\n",
            1
        ),
        analyze_fails(
            "immutable-assignment",
            "layer model(x: tensor[float16]): tensor[float16]:\n"
            "  let y = x\n"
            "  y = x\n"
            "  return y\n",
            "Cannot assign to immutable variable 'y'"
        ),
        analyze_fails(
            "unknown-symbol",
            "layer model(x: tensor[float16]): tensor[float16]:\n"
            "  return missing\n",
            "Undefined variable 'missing'"
        ),
        analyze_fails(
            "fn-arrow-callable-local",
            "fn helper(x: tensor[float16]): tensor[float16]:\n"
            "  let proj = linear(8)\n"
            "  return x -> proj()\n",
            "cannot be used inside fn"
        ),
        analyze_fails(
            "train-objective-string",
            "layer model(x):\n"
            "  return x\n"
            "\n"
            "config model:\n"
            "  optimizer = \"adam\"\n"
            "  objective = \"loss\"\n",
            "Field 'objective' must reference a named tensor root"
        ),
        analyze_fails(
            "train-tuple-mismatch",
            "layer model(loss: tensor[float16]): tensor[float16]:\n"
            "  return loss\n"
            "\n"
            "config model:\n"
            "  optimizer = (adam, sgd)\n"
            "  lr = (1e-4, 5e-4, 1e-3)\n"
            "  objective = loss\n",
            "Tuple-valued training config fields must have the same length"
        ),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
