#include "cli_args.h"
#include "diagnostic.h"

#include <cstdlib>
#include <utility>

namespace {

// Helper function to create standard CLI diagnostic errors
Diagnostic cli_error(std::string message) {
  return Diagnostic::error(DiagnosticCode::CliError, std::move(message));
}

// Parses a string backend identifier into the corresponding BackendKind enum.
std::variant<BackendKind, Diagnostic> parse_backend(const std::string &value) {
  if (value == "local") {
    return BackendKind::Local;
  }
  if (value == "metal") {
    return BackendKind::Metal;
  }
  if (value == "pytorch") {
    return BackendKind::PyTorch;
  }
  if (value == "cuda") {
    return BackendKind::Cuda;
  }
  if (value == "rocm") {
    return BackendKind::Rocm;
  }
  return cli_error("unknown backend: " + value);
}

// Parses a shape specification string in the format "name=dim1xdim2x..."
std::variant<std::pair<std::string, std::vector<std::int64_t>>, Diagnostic>
parse_shape_spec(const std::string &spec) {
  // Find the separator between the tensor name and its dimensions
  const std::size_t equals = spec.find('=');
  if (equals == std::string::npos || equals == 0 || equals + 1 >= spec.size()) {
    return cli_error("expected --shape name=dimxdim, got '" + spec + "'");
  }

  std::string name = spec.substr(0, equals);
  std::vector<std::int64_t> dims;
  std::size_t start = equals + 1;

  // Parse dimensions separated by 'x'
  while (start <= spec.size()) {
    const std::size_t end = spec.find('x', start);
    const std::string part = spec.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    if (part.empty()) {
      return cli_error("invalid dimension '' in shape '" + spec + "'");
    }

    // Convert the dimension string to an integer
    char *parsed_end = nullptr;
    const long long value = std::strtoll(part.c_str(), &parsed_end, 10);
    if (parsed_end == part.c_str() || *parsed_end != '\0' || value <= 0) {
      return cli_error("invalid dimension '" + part + "' in shape '" + spec +
                       "'");
    }
    dims.push_back(static_cast<std::int64_t>(value));

    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }

  if (dims.empty()) {
    return cli_error("shape '" + spec +
                     "' must contain at least one dimension");
  }
  return std::make_pair(std::move(name), std::move(dims));
}

} // namespace

const char *backend_name(BackendKind backend) {
  switch (backend) {
  case BackendKind::Local:
    return "local";
  case BackendKind::Metal:
    return "metal";
  case BackendKind::PyTorch:
    return "pytorch";
  case BackendKind::Cuda:
    return "cuda";
  case BackendKind::Rocm:
    return "rocm";
  }
  return "local";
}

std::string usage() {
  return "cpptysor <input.ty> [options]\n"
         "\n"
         "Options:\n"
         "  --run                         Run the entry function with the "
         "selected backend\n"
         "  --backward                    Run the backward pass for supported "
         "programs\n"
         "  --train                       Run a supported training config\n"
         "  --emit-metal                  Print generated Metal source\n"
         "  --emit-pytorch                Print generated PyTorch source\n"
         "  --emit-cuda                   Print generated CUDA source\n"
         "  --tokens                      Print lexer tokens\n"
         "  --ast                         Print parser AST summary\n"
         "  --semantics                   Print semantic summary\n"
         "  --ir                          Print lowered frontend IR\n"
         "  --graph                       Print lowered graph IR\n"
         "  --plan                        Print backend execution plan\n"
         "  --print-pipeline              Print the active compiler pipeline\n"
         "  --entry <name>                Entry function/layer name, defaults "
         "to model\n"
         "  --backend <local|metal|pytorch|cuda|rocm>\n"
         "                                Execution backend, defaults to "
         "local\n"
         "  --shape <name=dimxdim>        Tensor input shape, repeatable\n"
         "  --metal-device                Probe native Metal device "
         "availability\n"
         "  -h, --help                    Show this help\n"
         "  -V, --version                 Show version\n";
}

CliParseResult parse_cli(const std::vector<std::string> &raw_args) {
  CliOptions options;

  // Loop through arguments, updating options structure and parsing required
  // values
  for (std::size_t index = 0; index < raw_args.size(); ++index) {
    const std::string &arg = raw_args[index];

    if (arg == "--emit-metal") {
      options.emit_metal = true;
    } else if (arg == "--emit-pytorch") {
      options.emit_pytorch = true;
    } else if (arg == "--emit-cuda") {
      options.emit_cuda = true;
    } else if (arg == "--run") {
      options.run = true;
    } else if (arg == "--backward") {
      options.backward = true;
    } else if (arg == "--train") {
      options.train = true;
    } else if (arg == "--tokens") {
      options.tokens = true;
    } else if (arg == "--ast") {
      options.ast = true;
    } else if (arg == "--semantics") {
      options.semantics = true;
    } else if (arg == "--ir") {
      options.ir = true;
    } else if (arg == "--graph") {
      options.graph = true;
    } else if (arg == "--plan") {
      options.plan = true;
    } else if (arg == "--print-pipeline") {
      options.print_pipeline = true;
    } else if (arg == "--entry") {
      if (index + 1 >= raw_args.size()) {
        return cli_error("missing value for --entry");
      }
      options.entry = raw_args[++index];
    } else if (arg == "--backend") {
      if (index + 1 >= raw_args.size()) {
        return cli_error("missing value for --backend");
      }
      auto parsed = parse_backend(raw_args[++index]);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
      }
      options.backend = std::get<BackendKind>(parsed);
      options.backend_overridden = true;
    } else if (arg == "--shape") {
      if (index + 1 >= raw_args.size()) {
        return cli_error("missing value for --shape");
      }
      auto parsed = parse_shape_spec(raw_args[++index]);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&parsed)) {
        return *diagnostic;
      }
      auto shape = std::get<std::pair<std::string, std::vector<std::int64_t>>>(
          std::move(parsed));
      options.tensor_shapes.emplace(std::move(shape.first),
                                    std::move(shape.second));
    } else if (!arg.empty() && arg.rfind("--", 0) == 0) {
      // Unrecognized double-dash flag
      return cli_error("unknown option: " + arg);
    } else {
      // Treat anything not matching a flag as the input path
      if (options.input_path.has_value()) {
        return cli_error("multiple input paths provided");
      }
      options.input_path = arg;
    }
  }

  if (!options.input_path.has_value()) {
    return cli_error("missing input path");
  }

  return options;
}
