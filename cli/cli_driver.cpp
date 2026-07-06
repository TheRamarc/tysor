#include "cli_driver.h"

#include "backward_executor.h"
#include "cli_args.h"
#include "cli_pipeline.h"
#include "graph_executor.h"
#include "metal_executor.h"
#include "train_executor.h"

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace {

// Helper function to read the entire contents of a source file into a string
std::variant<std::string, Diagnostic> read_source_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return Diagnostic::error("cli", "C0002", "could not read " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

// Intercepts and executes standalone commands like --help, --version, or --metal-device
std::optional<int> handle_builtin_command(const std::vector<std::string>& raw_args) {
    if (raw_args.size() == 1 && (raw_args[0] == "-h" || raw_args[0] == "--help")) {
        std::cout << usage();
        return 0;
    }
    if (raw_args.size() == 1 && (raw_args[0] == "-V" || raw_args[0] == "--version")) {
        std::cout << "cpptysor 0.1.0\n";
        return 0;
    }
    if (raw_args.size() == 1 && raw_args[0] == "--metal-device") {
        auto result = probe_native_metal_device();
        if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
            std::cerr << diagnostic->to_string() << '\n';
            return 1;
        }
        std::cout << "metal_device " << std::get<std::string>(result) << '\n';
        return 0;
    }
    return std::nullopt;
}

std::variant<CliOptions, int> parse_or_report(const std::vector<std::string>& raw_args) {
    CliParseResult parsed = parse_cli(raw_args);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&parsed)) {
        std::cerr << "error: " << diagnostic->message << '\n';
        std::cerr << "usage: " << usage();
        return 2;
    }
    return std::get<CliOptions>(std::move(parsed));
}

// Checks if the user requested any actions that are not yet implemented in the C++ port
std::optional<Diagnostic> unsupported_requested_actions(const CliOptions& options) {
    std::vector<std::string> actions;
    if (options.emit_metal) {
        actions.push_back("--emit-metal");
    }
    if (options.emit_pytorch) {
        actions.push_back("--emit-pytorch");
    }
    if (options.emit_cuda) {
        actions.push_back("--emit-cuda");
    }
    if (actions.empty()) {
        return std::nullopt;
    }

    std::ostringstream message;
    message << "requested action";
    if (actions.size() > 1) {
        message << "s";
    }
    message << (actions.size() == 1 ? " requires" : " require");
    message << " backend codegen/runtime, which is not ported in cpptysor yet:";
    for (const auto& action : actions) {
        message << ' ' << action;
    }
    return Diagnostic::error("cli", "C0003", message.str())
        .with_help("cpptysor currently supports compiler pipeline dumps and local --run execution");
}

// Executes the compiled entry function using the appropriate backend runtime
std::optional<Diagnostic> run_entry_function(const CompiledProgram& compiled, const CliOptions& options) {
    if (!compiled.plan) {
        return Diagnostic::error("runtime", "R0001", "execution plan was not built for --run");
    }

    GraphExecutorOptions executor_options;
    executor_options.tensor_shapes = options.tensor_shapes;
    
    // Dispatch to the correct backend executor
    auto execution = options.backend == BackendKind::Metal
        ? execute_metal_plan_module(*compiled.plan, options.entry, executor_options)
        : execute_plan_module(*compiled.plan, options.entry, executor_options);
        
    if (const auto* diagnostic = std::get_if<Diagnostic>(&execution)) {
        return *diagnostic;
    }
    
    const GraphExecutionResult& result = std::get<GraphExecutionResult>(execution);
    if (result.outputs.empty()) {
        return Diagnostic::error("runtime", "R0001", "Entry function did not return a value");
    }
    if (result.outputs.size() != 1) {
        return Diagnostic::error("runtime", "R0001", "Runtime interpreter currently supports a single return value");
    }
    
    // Print the result to stdout
    print_graph_runtime_value(result.outputs.begin()->second);
    return std::nullopt;
}

// Core driver function for executing the full compiler pipeline and runtime actions
int run_program(CliOptions options) {
    // Read the source file from disk
    auto source_result = read_source_file(*options.input_path);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&source_result)) {
        std::cerr << diagnostic->to_string() << '\n';
        return 1;
    }

    const std::string source = std::get<std::string>(std::move(source_result));
    print_bootstrap_summary(options, source.size());
    std::cout.flush();

    // Compile the source code to an execution plan
    CompileResult compiled_result = compile_source(source, options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&compiled_result)) {
        std::cerr << diagnostic->to_string() << '\n';
        return 1;
    }

    // Output any requested debug dumps
    const auto compiled = std::get<CompiledProgram>(std::move(compiled_result));
    print_compile_summary(compiled);
    print_requested_dumps(compiled);

    // Validate that the requested actions are supported
    if (const auto unsupported = unsupported_requested_actions(options)) {
        std::cout.flush();
        std::cerr << unsupported->to_string() << '\n';
        return 1;
    }

    // Execute the forward pass if requested
    if (options.run) {
        if (const auto diagnostic = run_entry_function(compiled, options)) {
            std::cout.flush();
            std::cerr << diagnostic->to_string() << '\n';
            return 1;
        }
    }

    // Execute the backward pass (gradients) if requested
    if (options.backward) {
        if (!compiled.plan) {
            std::cout.flush();
            std::cerr << Diagnostic::error("runtime", "R0003", "execution plan was not built for --backward").to_string() << '\n';
            return 1;
        }
        GraphExecutorOptions executor_options;
        executor_options.tensor_shapes = options.tensor_shapes;
        if (const auto diagnostic = run_backward_plan_module(compiled.lowered, *compiled.plan, options.entry, executor_options)) {
            std::cout.flush();
            std::cerr << diagnostic->to_string() << '\n';
            return 1;
        }
    }

    // Execute the training loop if requested
    if (options.train) {
        if (!compiled.plan) {
            std::cout.flush();
            std::cerr << Diagnostic::error("runtime", "R0004", "execution plan was not built for --train").to_string() << '\n';
            return 1;
        }
        GraphExecutorOptions executor_options;
        executor_options.tensor_shapes = options.tensor_shapes;
        if (const auto diagnostic = run_train_plan_module(compiled.lowered, *compiled.plan, options.entry, executor_options)) {
            std::cout.flush();
            std::cerr << diagnostic->to_string() << '\n';
            return 1;
        }
    }

    if (options.print_pipeline) {
        std::cout << "pipeline=lexer->parser->semantic->frontend_ir->graph_ir->execution_plan->runtime\n";
    }

    return 0;
}

} // namespace

int run_cli(const std::vector<std::string>& raw_args) {
    if (const auto exit_code = handle_builtin_command(raw_args)) {
        return *exit_code;
    }

    auto parsed = parse_or_report(raw_args);
    if (const auto* exit_code = std::get_if<int>(&parsed)) {
        return *exit_code;
    }

    return run_program(std::get<CliOptions>(std::move(parsed)));
}
