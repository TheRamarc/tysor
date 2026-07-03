project(
  'tysor',
  'cpp',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
  ],
)

fs = import('fs')

cpp = meson.get_compiler('cpp')
cpp_args = []
if cpp.get_id() == 'msvc'
  cpp_args += ['/permissive-']
endif

include_dirs = include_directories(
  'backend/metal',
  'cli',
  'core',
  'frontend',
  'ir',
  'runtime',
)

frontend_sources = files(
  'core/diagnostic.cpp',
  'frontend/lexer.cpp',
  'frontend/ops.cpp',
  'frontend/parser.cpp',
  'frontend/semantic_analyzer.cpp',
  'ir/execution_plan.cpp',
  'ir/frontend_ir.cpp',
  'ir/graph_ir.cpp',
)

runtime_sources = files(
  'runtime/backward_executor.cpp',
  'runtime/graph_executor.cpp',
  'runtime/runtime_tensor.cpp',
  'runtime/train_executor.cpp',
)

metal_bridge = []
metal_deps = []
objcpp_args = []

if host_machine.system() == 'darwin'
  add_languages('objcpp', native: false, required: true)

  foreach candidate : [
    'backend/metal/metal_bridge.mm',
    '../native/metal_bridge.mm',
    '../tysor/native/metal_bridge.mm',
  ]
    if metal_bridge.length() == 0 and fs.exists(candidate)
      metal_bridge = files(candidate)
    endif
  endforeach

  if metal_bridge.length() == 0
    error('Could not find metal_bridge.mm')
  endif

  metal_deps += dependency('appleframeworks', modules: ['Foundation', 'Metal'])
  objcpp_args += ['-fobjc-arc']
endif

tysor = executable(
  'tysor',
  files(
    'cli/main.cpp',
    'backend/metal/metal_executor.cpp',
    'cli/cli_args.cpp',
    'cli/cli_driver.cpp',
    'cli/cli_pipeline.cpp',
  ) + frontend_sources + runtime_sources + metal_bridge,
  include_directories: include_dirs,
  cpp_args: cpp_args,
  objcpp_args: objcpp_args,
  dependencies: metal_deps,
)

parser_smoke = executable(
  'cpptysor_parser_smoke',
  files('tests/parser_smoke.cpp') + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

semantic_smoke = executable(
  'cpptysor_semantic_smoke',
  files('tests/semantic_smoke.cpp') + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

frontend_ir_smoke = executable(
  'cpptysor_frontend_ir_smoke',
  files('tests/frontend_ir_smoke.cpp') + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

graph_ir_smoke = executable(
  'cpptysor_graph_ir_smoke',
  files('tests/graph_ir_smoke.cpp') + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

ops_smoke = executable(
  'cpptysor_ops_smoke',
  files(
    'tests/ops_smoke.cpp',
    'core/diagnostic.cpp',
    'frontend/lexer.cpp',
    'frontend/ops.cpp',
    'frontend/parser.cpp',
  ),
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

execution_plan_smoke = executable(
  'cpptysor_execution_plan_smoke',
  files('tests/execution_plan_smoke.cpp') + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

graph_executor_smoke = executable(
  'cpptysor_graph_executor_smoke',
  files(
    'tests/graph_executor_smoke.cpp',
    'runtime/graph_executor.cpp',
    'runtime/runtime_tensor.cpp',
  ) + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

runtime_bench = executable(
  'cpptysor_runtime_bench',
  files(
    'bench/runtime_bench.cpp',
    'cli/cli_args.cpp',
    'cli/cli_pipeline.cpp',
    'runtime/graph_executor.cpp',
    'runtime/runtime_tensor.cpp',
  ) + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

backward_smoke = executable(
  'cpptysor_backward_smoke',
  files(
    'tests/backward_smoke.cpp',
    'cli/cli_args.cpp',
    'cli/cli_pipeline.cpp',
    'runtime/backward_executor.cpp',
    'runtime/graph_executor.cpp',
    'runtime/runtime_tensor.cpp',
  ) + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

train_smoke = executable(
  'cpptysor_train_smoke',
  files(
    'tests/train_smoke.cpp',
    'cli/cli_args.cpp',
    'cli/cli_pipeline.cpp',
    'runtime/graph_executor.cpp',
    'runtime/runtime_tensor.cpp',
    'runtime/train_executor.cpp',
  ) + frontend_sources,
  include_directories: include_dirs,
  cpp_args: cpp_args,
)

metal_executor_smoke = executable(
  'cpptysor_metal_executor_smoke',
  files(
    'tests/metal_executor_smoke.cpp',
    'backend/metal/metal_executor.cpp',
    'runtime/graph_executor.cpp',
    'runtime/runtime_tensor.cpp',
  ) + frontend_sources + metal_bridge,
  include_directories: include_dirs,
  cpp_args: cpp_args,
  objcpp_args: objcpp_args,
  dependencies: metal_deps,
)

test('parser_smoke', parser_smoke)
test('semantic_smoke', semantic_smoke)
test('frontend_ir_smoke', frontend_ir_smoke)
test('graph_ir_smoke', graph_ir_smoke)
test('ops_smoke', ops_smoke)
test('execution_plan_smoke', execution_plan_smoke)
test('graph_executor_smoke', graph_executor_smoke)
test('backward_smoke', backward_smoke)
test('train_smoke', train_smoke)
test('metal_executor_smoke', metal_executor_smoke)
