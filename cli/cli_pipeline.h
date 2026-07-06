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

/**
 * @brief Represents the comprehensive result of compiling a program.
 * 
 * Contains the artifacts generated at each stage of the compiler pipeline, 
 * ranging from raw tokens and AST to lowered modules, execution plans, and 
 * optional string dumps used for debugging and diagnostics.
 */
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

/**
 * @brief Drives the end-to-end compilation process for a given source string.
 *
 * Runs the source through the lexer, parser, semantic analyzer, and lowerers
 * to produce execution plans and graphs based on the requested CLI options.
 *
 * @param source The input program source code.
 * @param options CLI options controlling pipeline behavior and dumps.
 * @return CompileResult The populated CompiledProgram on success, or a Diagnostic error.
 */
CompileResult compile_source(const std::string& source, const CliOptions& options);

/**
 * @brief Prints an initial summary of the compilation inputs and options.
 *
 * @param options The user-provided CLI options.
 * @param source_size The size of the input source file in bytes.
 */
void print_bootstrap_summary(const CliOptions& options, std::size_t source_size);

/**
 * @brief Prints quantitative and structural metrics of the compiled program.
 *
 * Displays counts of tokens, layers, functions, and optionally IR node statistics.
 *
 * @param compiled The fully populated CompiledProgram.
 */
void print_compile_summary(const CompiledProgram& compiled);

/**
 * @brief Prints any requested diagnostic dumps (AST, IR, Execution Plan, etc.) to standard output.
 *
 * @param compiled The CompiledProgram containing the optional string dumps.
 */
void print_requested_dumps(const CompiledProgram& compiled);
