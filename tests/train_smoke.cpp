#include "cli_pipeline.h"
#include "train_executor.h"

#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace {

int run_train_smoke(
    const std::string& name,
    const std::string& source,
    const std::map<std::string, std::vector<std::int64_t>>& shapes
) {
    CliOptions options;
    options.input_path = name + ".ty";
    options.train = true;
    options.backend = BackendKind::Local;
    options.tensor_shapes = shapes;

    auto compiled_result = compile_source(source, options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&compiled_result)) {
        std::cerr << diagnostic->to_string() << '\n';
        return 1;
    }
    CompiledProgram compiled = std::get<CompiledProgram>(std::move(compiled_result));
    if (!compiled.plan) {
        std::cerr << name << ": expected execution plan\n";
        return 1;
    }

    GraphExecutorOptions executor_options;
    executor_options.tensor_shapes = options.tensor_shapes;
    if (auto diagnostic = run_train_plan_module(compiled.lowered, *compiled.plan, "model", executor_options)) {
        std::cerr << diagnostic->to_string() << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const std::string linear_source =
        "layer model(x: tensor[float32], target: tensor[float32]): tensor[float32]:\n"
        "  let proj = linear(3, 3, true)\n"
        "  let logits = x -> proj()\n"
        "  let loss = cross_entropy(logits, target)\n"
        "  return loss\n"
        "\n"
        "config model:\n"
        "  optimizer = \"sgd\"\n"
        "  lr = 0.05\n"
        "  iteration = 2\n"
        "  objective = loss\n";

    const std::string transformer_source =
        "layer model(idx: tensor[float16], target: tensor[float16]): tensor[float16]:\n"
        "  let tok = Embedding(32, 8)\n"
        "  let out_proj = linear(16, 3, false)\n"
        "\n"
        "  let mut x = tok(idx)\n"
        "  x = rms_norm(x, 8)\n"
        "  x = reshape(x, 1, 2, 4)\n"
        "  x = rope(x, 4, 10000.0)\n"
        "  x = repeat_kv(x, 2)\n"
        "  x = causal_mask(x)\n"
        "  x = flatten_heads(x)\n"
        "  let logits = x -> out_proj()\n"
        "  let loss = cross_entropy(logits, target)\n"
        "  return loss\n"
        "\n"
        "config model:\n"
        "  optimizer = \"sgd\"\n"
        "  lr = 0.1\n"
        "  iteration = 3\n"
        "  objective = loss\n";

    if (run_train_smoke("linear_train_smoke", linear_source, {{"x", {2, 3}}, {"target", {2, 3}}}) != 0) {
        return 1;
    }
    if (run_train_smoke("transformer_train_smoke", transformer_source, {{"idx", {1}}, {"target", {1, 3}}}) != 0) {
        return 1;
    }

    return 0;
}
