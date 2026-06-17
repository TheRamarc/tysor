#include "cli_pipeline.h"
#include "graph_executor.h"
#include "runtime_tensor.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

volatile float benchmark_sink = 0.0F;

struct BenchResult {
    std::string name;
    std::size_t iterations = 0;
    double total_ms = 0.0;
    double avg_us = 0.0;
};

void consume_tensor(const SimpleTensor& tensor) {
    if (!tensor.data.empty()) {
        benchmark_sink += tensor.data[tensor.data.size() / 2];
    }
}

void consume_and_release(SimpleTensor tensor, RuntimeTensorWorkspace& workspace) {
    consume_tensor(tensor);
    workspace.release(std::move(tensor));
}

void consume_execution(const GraphExecutionResult& execution) {
    for (const auto& item : execution.outputs) {
        if (const auto* tensor = std::get_if<SimpleTensor>(&item.second)) {
            consume_tensor(*tensor);
            return;
        }
        if (const auto* value = std::get_if<double>(&item.second)) {
            benchmark_sink += static_cast<float>(*value);
            return;
        }
        if (const auto* value = std::get_if<std::int64_t>(&item.second)) {
            benchmark_sink += static_cast<float>(*value);
            return;
        }
    }
}

void release_execution(GraphExecutionResult&& execution, RuntimeTensorWorkspace& workspace) {
    for (auto& item : execution.outputs) {
        if (auto* tensor = std::get_if<SimpleTensor>(&item.second)) {
            workspace.release(std::move(*tensor));
        }
    }
    for (auto& item : execution.values) {
        if (auto* tensor = std::get_if<SimpleTensor>(&item.second)) {
            workspace.release(std::move(*tensor));
        }
    }
}

SimpleTensor require_tensor_result(std::variant<SimpleTensor, Diagnostic> result, const std::string& name) {
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        throw std::runtime_error(name + ": " + diagnostic->to_string());
    }
    return std::get<SimpleTensor>(std::move(result));
}

template <typename Function>
BenchResult time_case(std::string name, std::size_t iterations, Function function) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < iterations; ++index) {
        function();
    }
    const auto stop = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
    return BenchResult{std::move(name), iterations, elapsed, elapsed * 1000.0 / static_cast<double>(iterations)};
}

PlanModule compile_graph_benchmark_plan() {
    const std::string source =
        "layer model(x: tensor[float32], w: tensor[float32]): tensor[float32]:\n"
        "  return relu(matmul(x, w))\n";

    CliOptions options;
    options.input_path = "runtime_bench.ty";
    options.backend = BackendKind::Local;
    options.run = true;

    auto compiled_result = compile_source(source, options);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&compiled_result)) {
        throw std::runtime_error(diagnostic->to_string());
    }
    CompiledProgram compiled = std::get<CompiledProgram>(std::move(compiled_result));
    if (!compiled.plan) {
        throw std::runtime_error("runtime_bench: expected compiled execution plan");
    }
    return *compiled.plan;
}

GraphExecutionResult require_graph_result(GraphExecutionResultVariant result) {
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        throw std::runtime_error(diagnostic->to_string());
    }
    return std::get<GraphExecutionResult>(std::move(result));
}

void print_result(const BenchResult& result) {
    std::cout << std::left << std::setw(24) << result.name
              << std::right << std::setw(12) << result.iterations
              << std::setw(14) << std::fixed << std::setprecision(3) << result.total_ms
              << std::setw(14) << std::fixed << std::setprecision(3) << result.avg_us << '\n';
}

void print_workspace_stats(const std::string& name, const RuntimeTensorWorkspace& workspace) {
    const RuntimeTensorWorkspaceStats stats = workspace.stats();
    std::cout << name
              << " allocations=" << stats.allocations
              << " reuses=" << stats.reuses
              << " releases=" << stats.releases
              << " cached_buffers=" << stats.cached_buffers
              << " cached_bytes=" << stats.cached_bytes
              << '\n';
}

} // namespace

int main() {
    try {
        const std::vector<std::int64_t> square_shape{64, 64};
        const std::vector<std::int64_t> reshape_shape{128, 32};
        const std::vector<std::int64_t> flatten_shape{16, 8, 32};
        const std::vector<std::int64_t> softmax_shape{128, 128};
        const std::vector<std::int64_t> linear_input_shape{32, 128};

        const SimpleTensor lhs = make_synthetic_tensor(square_shape, "float32");
        const SimpleTensor rhs = make_synthetic_tensor(square_shape, "float32");
        const SimpleTensor flatten_input = make_synthetic_tensor(flatten_shape, "float32");
        const SimpleTensor softmax_input = make_synthetic_tensor(softmax_shape, "float32");
        const SimpleTensor linear_input = make_synthetic_tensor(linear_input_shape, "float32");
        RuntimeTensorWorkspace direct_workspace;
        RuntimeTensorWorkspace graph_workspace;

        LinearClosure linear;
        linear.in_features = 128;
        linear.out_features = 64;
        linear.with_bias = true;
        linear.dtype = "float32";

        const PlanModule graph_plan = compile_graph_benchmark_plan();
        GraphExecutorOptions graph_options;
        graph_options.tensor_shapes["x"] = {32, 64};
        graph_options.tensor_shapes["w"] = {64, 32};
        graph_options.collect_intermediate_values = false;
        graph_options.tensor_workspace = &graph_workspace;

        const std::vector<BenchResult> results{
            time_case("matmul_64x64", 150, [&]() {
                consume_and_release(require_tensor_result(matmul(lhs, rhs, &direct_workspace), "matmul_64x64"), direct_workspace);
            }),
            time_case("elementwise_add_4k", 2000, [&]() {
                consume_and_release(
                    require_tensor_result(elementwise_binary(FeBinaryOp::Add, lhs, rhs, &direct_workspace), "elementwise_add_4k"),
                    direct_workspace
                );
            }),
            time_case("relu_4k", 5000, [&]() {
                consume_and_release(apply_relu(lhs, &direct_workspace), direct_workspace);
            }),
            time_case("reshape_4k", 10000, [&]() {
                consume_and_release(
                    require_tensor_result(apply_reshape(lhs, reshape_shape, &direct_workspace), "reshape_4k"),
                    direct_workspace
                );
            }),
            time_case("flatten_heads_4k", 10000, [&]() {
                consume_and_release(
                    require_tensor_result(apply_flatten_heads(flatten_input, &direct_workspace), "flatten_heads_4k"),
                    direct_workspace
                );
            }),
            time_case("softmax_128x128", 400, [&]() {
                consume_and_release(
                    require_tensor_result(apply_softmax(softmax_input, &direct_workspace), "softmax_128x128"),
                    direct_workspace
                );
            }),
            time_case("linear_32x128_to_64", 250, [&]() {
                consume_and_release(
                    require_tensor_result(apply_linear(linear, linear_input, &direct_workspace), "linear_32x128_to_64"),
                    direct_workspace
                );
            }),
            time_case("graph_matmul_relu", 150, [&]() {
                GraphExecutionResult execution = require_graph_result(execute_plan_module(graph_plan, "model", graph_options));
                consume_execution(execution);
                release_execution(std::move(execution), graph_workspace);
            }),
        };

        std::cout << std::left << std::setw(24) << "benchmark"
                  << std::right << std::setw(12) << "iterations"
                  << std::setw(14) << "total_ms"
                  << std::setw(14) << "avg_us" << '\n';
        for (const auto& result : results) {
            print_result(result);
        }
        std::cout << "checksum=" << benchmark_sink << '\n';
        print_workspace_stats("direct_workspace", direct_workspace);
        print_workspace_stats("graph_workspace", graph_workspace);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime_bench: " << error.what() << '\n';
        return 1;
    }
}
