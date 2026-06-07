#include "cli_pipeline.h"

#include <iostream>
#include <sstream>
#include <utility>

namespace {

void print_section(const char* start, const std::string& body, const char* end) {
    std::cout << '\n' << start << '\n';
    std::cout << body << '\n';
    std::cout << end << '\n';
}

std::string make_token_dump(const std::vector<Token>& tokens) {
    std::ostringstream out;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            out << '\n';
        }
        out << token_to_string(tokens[index]);
    }
    return out.str();
}

} // namespace

CompileResult compile_source(const std::string& source, const CliOptions& options) {
    TokenizeResult tokenized = tokenize_with_diagnostic(source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&tokenized)) {
        return *diagnostic;
    }

    auto tokens = std::get<std::vector<Token>>(std::move(tokenized));
    CompiledProgram compiled;
    compiled.token_count = tokens.size();
    compiled.tokens = std::move(tokens);
    if (options.tokens) {
        compiled.token_dump = make_token_dump(compiled.tokens);
    }

    Parser parser(compiled.tokens);
    ParseResult parsed = parser.parse_program();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
    }
    compiled.program = std::get<Program>(std::move(parsed));
    if (options.ast) {
        compiled.ast_dump = ast_to_string(compiled.program);
    }

    SemanticAnalyzer analyzer;
    SemanticResult semantic_result = analyzer.analyze_with_info(compiled.program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&semantic_result)) {
        return *diagnostic;
    }
    compiled.semantic_info = std::get<SemanticInfo>(std::move(semantic_result));
    if (options.semantics) {
        compiled.semantics_dump = semantic_info_summary(compiled.semantic_info, compiled.program);
    }

    FrontendLowerer lowerer(compiled.program, compiled.semantic_info);
    FrontendResult frontend_result = lowerer.lower();
    if (const auto* diagnostic = std::get_if<Diagnostic>(&frontend_result)) {
        return *diagnostic;
    }
    compiled.lowered = std::get<LoweredModule>(std::move(frontend_result));
    if (options.ir) {
        compiled.ir_dump = frontend_ir_to_string(compiled.lowered);
    }
    if (options.graph || options.plan || options.run || options.backward || options.train) {
        auto graph_result = build_graph_module(compiled.lowered);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&graph_result)) {
            return *diagnostic;
        }
        compiled.graph = std::get<GraphModule>(std::move(graph_result));
        if (options.graph) {
            compiled.graph_dump = graph_ir_to_string(*compiled.graph);
        }
    }
    if (options.plan || options.run || options.backward || options.train) {
        auto plan_result = compile_plan_module(*compiled.graph, options.backend);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&plan_result)) {
            return *diagnostic;
        }
        compiled.plan = std::get<PlanModule>(std::move(plan_result));
        if (options.plan) {
            compiled.plan_dump = execution_plan_to_string(*compiled.plan);
        }
    }
    return compiled;
}

void print_bootstrap_summary(const CliOptions& options, std::size_t source_size) {
    std::cout << "tysor c++17 port bootstrap\n";
    std::cout << "input=" << *options.input_path << '\n';
    std::cout << "bytes=" << source_size << '\n';
    std::cout << "entry=" << options.entry << '\n';
    std::cout << "backend=" << backend_name(options.backend) << '\n';
}

void print_compile_summary(const CompiledProgram& compiled) {
    std::cout << "tokens=" << compiled.token_count << '\n';
    std::cout << "program=configs:" << compiled.program.configs.size()
              << " layers:" << compiled.program.layers.size()
              << " functions:" << compiled.program.functions.size()
              << " globals:" << compiled.program.globals.size() << '\n';
    std::cout << lowered_module_summary(compiled.lowered) << '\n';
    if (compiled.graph.has_value()) {
        std::cout << graph_module_summary(*compiled.graph) << '\n';
    }
    if (compiled.plan.has_value()) {
        std::cout << execution_plan_summary(*compiled.plan) << '\n';
    }
}

void print_requested_dumps(const CompiledProgram& compiled) {
    if (compiled.token_dump.has_value()) {
        print_section("--- Tokenization Step ---", *compiled.token_dump, "-------------------------");
    }
    if (compiled.ast_dump.has_value()) {
        print_section("--- Parsing Step ---", *compiled.ast_dump, "--------------------");
    }
    if (compiled.semantics_dump.has_value()) {
        print_section(
            "--- Semantic Analysis Step ---",
            *compiled.semantics_dump,
            "------------------------------"
        );
    }
    if (compiled.ir_dump.has_value()) {
        print_section("--- Lowered Frontend IR ---", *compiled.ir_dump, "---------------------------");
    }
    if (compiled.graph_dump.has_value()) {
        print_section("--- Lowered Graph IR ---", *compiled.graph_dump, "------------------------");
    }
    if (compiled.plan_dump.has_value()) {
        print_section("--- Execution Plan ---", *compiled.plan_dump, "----------------------");
    }
}
