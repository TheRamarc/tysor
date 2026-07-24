#include "backward_executor.h"
#include "cli_pipeline.h"

#include <iostream>
#include <string>
#include <variant>

int main() {
    const std::string source =
        "layer model(x: tensor[float32]): tensor[float32]:\n"
        "  y = x -> SiLU()\n"
        "  return y\n"
        "\n"
        "config model:\n"
        "  optimizer = \"sgd\"\n"
        "  lr = 0.1\n"
        "  objective = y\n";

    CliOptions options;
    options.input_path = "backward_smoke.ty";
    options.backward = true;
    options.backend = BackendKind::Local;
    options.tensor_shapes["x"] = {2, 3};

    auto compiled_result = compile_source(source, options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&compiled_result)) {
        std::cerr << diagnostic->toString() << '\n';
        return 1;
    }
    CompiledProgram compiled = std::get<CompiledProgram>(std::move(compiled_result));
    if (!compiled.plan) {
        std::cerr << "backward-smoke: expected execution plan\n";
        return 1;
    }

    GraphExecutorOptions executor_options;
    executor_options.tensor_shapes = options.tensor_shapes;
    if (auto diagnostic = run_backward_plan_module(compiled.lowered, *compiled.plan, "model", executor_options)) {
        std::cerr << diagnostic->toString() << '\n';
        return 1;
    }
    return 0;
}
