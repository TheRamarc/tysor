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
    std::optional<std::string> input_path;
    bool emit_metal = false;
    bool emit_pytorch = false;
    bool emit_cuda = false;
    bool run = false;
    bool backward = false;
    bool train = false;
    bool tokens = false;
    bool ast = false;
    bool semantics = false;
    bool ir = false;
    bool graph = false;
    bool plan = false;
    bool print_pipeline = false;
    bool backend_overridden = false;
    std::string entry = "model";
    BackendKind backend = BackendKind::Local;
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
