#pragma once

#include "cli_args.h"
#include "execution_plan.h"
#include "frontend_ir.h"
#include "graph_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

struct CompiledProgram {
    std::size_t token_count = 0;
    std::vector<Token> tokens;
    Program program;
    SemanticInfo semantic_info;
    LoweredModule lowered;
    std::optional<GraphModule> graph;
    std::optional<PlanModule> plan;
    std::optional<std::string> token_dump;
    std::optional<std::string> ast_dump;
    std::optional<std::string> semantics_dump;
    std::optional<std::string> ir_dump;
    std::optional<std::string> graph_dump;
    std::optional<std::string> plan_dump;
};

using CompileResult = std::variant<CompiledProgram, Diagnostic>;

CompileResult compile_source(const std::string& source, const CliOptions& options);

void print_bootstrap_summary(const CliOptions& options, std::size_t source_size);
void print_compile_summary(const CompiledProgram& compiled);
void print_requested_dumps(const CompiledProgram& compiled);
