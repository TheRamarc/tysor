cmake_minimum_required(VERSION 3.20)

project(cpptysor LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPPTYSOR_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/cli
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend
    ${CMAKE_CURRENT_SOURCE_DIR}/ir
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

set(CPPTYSOR_FRONTEND_SOURCES
    core/diagnostic.cpp
    frontend/lexer.cpp
    frontend/ops.cpp
    frontend/parser.cpp
    frontend/semantic_analyzer.cpp
    ir/execution_plan.cpp
    ir/frontend_ir.cpp
    ir/graph_ir.cpp
)

set(CPPTYSOR_RUNTIME_SOURCES
    runtime/backward_executor.cpp
    runtime/graph_executor.cpp
    runtime/runtime_tensor.cpp
    runtime/train_executor.cpp
)

if(APPLE)
    enable_language(OBJCXX)
    set(CPPTYSOR_METAL_BRIDGE_CANDIDATES
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/metal_bridge.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/../native/metal_bridge.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/../tysor/native/metal_bridge.mm
    )
    foreach(candidate IN LISTS CPPTYSOR_METAL_BRIDGE_CANDIDATES)
        if(EXISTS ${candidate})
            set(CPPTYSOR_METAL_BRIDGE ${candidate})
            break()
        endif()
    endforeach()
    if(NOT CPPTYSOR_METAL_BRIDGE)
        message(FATAL_ERROR "Could not find metal_bridge.mm")
    endif()
    set_source_files_properties(${CPPTYSOR_METAL_BRIDGE} PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-fobjc-arc"
    )
else()
    set(CPPTYSOR_METAL_BRIDGE)
endif()

function(cpptysor_configure_target target)
    target_include_directories(${target} PRIVATE ${CPPTYSOR_INCLUDE_DIRS})
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(cpptysor_add_executable target)
    add_executable(${target} ${ARGN})
    cpptysor_configure_target(${target})
endfunction()

cpptysor_add_executable(cpptysor
    cli/main.cpp
    backend/metal/metal_executor.cpp
    cli/cli_args.cpp
    cli/cli_driver.cpp
    cli/cli_pipeline.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
    ${CPPTYSOR_RUNTIME_SOURCES}
    ${CPPTYSOR_METAL_BRIDGE}
)

cpptysor_add_executable(cpptysor_parser_smoke
    tests/parser_smoke.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
)

cpptysor_add_executable(cpptysor_semantic_smoke
    tests/semantic_smoke.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
)

cpptysor_add_executable(cpptysor_frontend_ir_smoke
    tests/frontend_ir_smoke.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
)

cpptysor_add_executable(cpptysor_graph_ir_smoke
    tests/graph_ir_smoke.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
)

cpptysor_add_executable(cpptysor_ops_smoke
    tests/ops_smoke.cpp
    core/diagnostic.cpp
    frontend/lexer.cpp
    frontend/ops.cpp
    frontend/parser.cpp
)

cpptysor_add_executable(cpptysor_execution_plan_smoke
    tests/execution_plan_smoke.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
)

cpptysor_add_executable(cpptysor_graph_executor_smoke
    tests/graph_executor_smoke.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
    runtime/graph_executor.cpp
    runtime/runtime_tensor.cpp
)

cpptysor_add_executable(cpptysor_backward_smoke
    tests/backward_smoke.cpp
    cli/cli_args.cpp
    cli/cli_pipeline.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
    runtime/backward_executor.cpp
    runtime/graph_executor.cpp
    runtime/runtime_tensor.cpp
)

cpptysor_add_executable(cpptysor_train_smoke
    tests/train_smoke.cpp
    cli/cli_args.cpp
    cli/cli_pipeline.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
    runtime/graph_executor.cpp
    runtime/runtime_tensor.cpp
    runtime/train_executor.cpp
)

cpptysor_add_executable(cpptysor_metal_executor_smoke
    tests/metal_executor_smoke.cpp
    backend/metal/metal_executor.cpp
    ${CPPTYSOR_FRONTEND_SOURCES}
    runtime/graph_executor.cpp
    runtime/runtime_tensor.cpp
    ${CPPTYSOR_METAL_BRIDGE}
)

if(APPLE)
    target_link_libraries(cpptysor PRIVATE "-framework Foundation" "-framework Metal")
    target_link_libraries(cpptysor_metal_executor_smoke PRIVATE "-framework Foundation" "-framework Metal")
endif()

enable_testing()
add_test(NAME parser_smoke COMMAND cpptysor_parser_smoke)
add_test(NAME semantic_smoke COMMAND cpptysor_semantic_smoke)
add_test(NAME frontend_ir_smoke COMMAND cpptysor_frontend_ir_smoke)
add_test(NAME graph_ir_smoke COMMAND cpptysor_graph_ir_smoke)
add_test(NAME ops_smoke COMMAND cpptysor_ops_smoke)
add_test(NAME execution_plan_smoke COMMAND cpptysor_execution_plan_smoke)
add_test(NAME graph_executor_smoke COMMAND cpptysor_graph_executor_smoke)
add_test(NAME backward_smoke COMMAND cpptysor_backward_smoke)
add_test(NAME train_smoke COMMAND cpptysor_train_smoke)
add_test(NAME metal_executor_smoke COMMAND cpptysor_metal_executor_smoke)
