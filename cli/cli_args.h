#pragma once

#include "diagnostic.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

enum class BackendKind {
    Local,
    Metal,
    PyTorch,
    Cuda,
    Rocm,
};

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

const char* backend_name(BackendKind backend);
std::string usage();
CliParseResult parse_cli(const std::vector<std::string>& raw_args);
