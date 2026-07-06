#pragma once

#include "diagnostic.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

/**
 * @brief Identifies the target execution backend for the compiled program.
 */
enum class BackendKind {
    Local,
    Metal,
    PyTorch,
    Cuda,
    Rocm,
};

/**
 * @brief Encapsulates the parsed configuration options driven by command-line arguments.
 */
struct CliOptions {
    /**
     * @brief The path to the input source file.
     * 1. Why it exists: Specifies which file the compiler should read and process.
     * 2. What it tracks: The file path as an optional string (empty if reading from stdin or not provided).
     * 3. What mutates/updates it: Populated by the argument parser when encountering a positional argument.
     */
    std::optional<std::string> input_path;

    /**
     * @brief Flag to emit Metal backend code.
     * 1. Why it exists: Configures the compiler to output code for Apple Metal targets.
     * 2. What it tracks: True if the user requested Metal emission, otherwise false.
     * 3. What mutates/updates it: Mutated by the `--emit-metal` command-line flag.
     */
    bool emit_metal = false;

    /**
     * @brief Flag to emit PyTorch backend code.
     * 1. Why it exists: Configures the compiler to generate PyTorch-compatible code/bindings.
     * 2. What it tracks: True if the user requested PyTorch emission, otherwise false.
     * 3. What mutates/updates it: Mutated by the `--emit-pytorch` command-line flag.
     */
    bool emit_pytorch = false;

    /**
     * @brief Flag to emit CUDA backend code.
     * 1. Why it exists: Configures the compiler to output code for NVIDIA CUDA targets.
     * 2. What it tracks: True if the user requested CUDA emission, otherwise false.
     * 3. What mutates/updates it: Mutated by the `--emit-cuda` command-line flag.
     */
    bool emit_cuda = false;

    /**
     * @brief Flag to execute the compiled program immediately.
     * 1. Why it exists: Enables a JIT-like or immediate execution mode directly from the CLI.
     * 2. What it tracks: True if the user intends to run the model after compilation.
     * 3. What mutates/updates it: Mutated by the `--run` command-line flag.
     */
    bool run = false;

    /**
     * @brief Flag to enable backward pass (autodiff) generation.
     * 1. Why it exists: Triggers the generation of gradients and reverse-mode AD.
     * 2. What it tracks: True if the backward pass logic should be compiled.
     * 3. What mutates/updates it: Mutated by the `--backward` command-line flag.
     */
    bool backward = false;

    /**
     * @brief Flag to enable training mode.
     * 1. Why it exists: Instructs the compiler to apply training-specific optimizations (e.g., preserving activations).
     * 2. What it tracks: True if the training pipeline is requested.
     * 3. What mutates/updates it: Mutated by the `--train` command-line flag.
     */
    bool train = false;

    /**
     * @brief Flag to dump the token stream.
     * 1. Why it exists: Assists in debugging the lexer.
     * 2. What it tracks: True if a token dump is requested.
     * 3. What mutates/updates it: Mutated by the `--tokens` command-line flag.
     */
    bool tokens = false;

    /**
     * @brief Flag to dump the Abstract Syntax Tree.
     * 1. Why it exists: Assists in debugging the parser.
     * 2. What it tracks: True if an AST dump is requested.
     * 3. What mutates/updates it: Mutated by the `--ast` command-line flag.
     */
    bool ast = false;

    /**
     * @brief Flag to dump semantic analysis information.
     * 1. Why it exists: Assists in debugging type checking and symbol resolution.
     * 2. What it tracks: True if a semantics dump is requested.
     * 3. What mutates/updates it: Mutated by the `--semantics` command-line flag.
     */
    bool semantics = false;

    /**
     * @brief Flag to dump the Intermediate Representation (IR).
     * 1. Why it exists: Assists in debugging the lowered module.
     * 2. What it tracks: True if an IR dump is requested.
     * 3. What mutates/updates it: Mutated by the `--ir` command-line flag.
     */
    bool ir = false;

    /**
     * @brief Flag to dump the computational graph.
     * 1. Why it exists: Assists in debugging graph generation and high-level structure.
     * 2. What it tracks: True if a graph dump is requested.
     * 3. What mutates/updates it: Mutated by the `--graph` command-line flag.
     */
    bool graph = false;

    /**
     * @brief Flag to dump the execution plan.
     * 1. Why it exists: Assists in debugging backend execution planning and allocations.
     * 2. What it tracks: True if an execution plan dump is requested.
     * 3. What mutates/updates it: Mutated by the `--plan` command-line flag.
     */
    bool plan = false;

    /**
     * @brief Flag to print the compilation pipeline summary.
     * 1. Why it exists: Gives users a high-level statistical overview of the compilation process.
     * 2. What it tracks: True if pipeline summary output is enabled.
     * 3. What mutates/updates it: Mutated by the `--print-pipeline` command-line flag.
     */
    bool print_pipeline = false;

    /**
     * @brief Flag indicating if the target backend was explicitly overridden.
     * 1. Why it exists: Distinguishes between a default backend and a user-specified one for conflict resolution.
     * 2. What it tracks: True if the user provided a backend-specific execution flag.
     * 3. What mutates/updates it: Mutated by explicit backend flags (e.g., `--metal`, `--cuda`).
     */
    bool backend_overridden = false;

    /**
     * @brief The name of the entry function to execute.
     * 1. Why it exists: Defines the main starting point of the compiled program.
     * 2. What it tracks: The entry function identifier (defaults to "model").
     * 3. What mutates/updates it: Mutated by the `--entry` command-line flag.
     */
    std::string entry = "model";

    /**
     * @brief The selected execution backend.
     * 1. Why it exists: Dictates which backend engine is used when running the program.
     * 2. What it tracks: The chosen BackendKind (e.g., Local, Metal, PyTorch).
     * 3. What mutates/updates it: Mutated by backend-specific command-line flags.
     */
    BackendKind backend = BackendKind::Local;

    /**
     * @brief Shapes of input tensors for the entry function.
     * 1. Why it exists: Allows specifying dynamic input dimensions explicitly via the CLI.
     * 2. What it tracks: A mapping from tensor names (strings) to their shape dimensions.
     * 3. What mutates/updates it: Mutated by the `--shape` command-line flag and its arguments.
     */
    std::map<std::string, std::vector<std::int64_t>> tensor_shapes;
};

using CliParseResult = std::variant<CliOptions, Diagnostic>;

/**
 * @brief Retrieves the string representation of a BackendKind.
 *
 * @param backend The backend enum value.
 * @return const char* The string name of the backend.
 */
const char* backend_name(BackendKind backend);

/**
 * @brief Retrieves the usage instructions for the CLI driver.
 *
 * @return std::string The usage text to be displayed to the user.
 */
std::string usage();

/**
 * @brief Parses raw command-line arguments into a structured CliOptions object.
 *
 * Validates flags, arguments, and handles type conversion for configuration fields.
 *
 * @param raw_args A vector of raw command line string arguments.
 * @return CliParseResult The parsed CliOptions, or a Diagnostic if a parsing error occurred.
 */
CliParseResult parse_cli(const std::vector<std::string>& raw_args);
