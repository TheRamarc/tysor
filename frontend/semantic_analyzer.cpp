#include "semantic_analyzer.h"

#include "ops.h"

#include <algorithm>
#include <optional>
#include <set>
#include <sstream>
#include <iostream>
#include <type_traits>
#include <utility>

namespace {

bool is_callable(const Type& type) {
    return type.base == TypeBase::Callable;
}

Type tensor_any() {
    return Type::tensor(std::nullopt, std::nullopt, std::nullopt);
}

Type infer_call_result_type(const std::string& callee, const Type& declared_type, const std::vector<Type>& arg_types) {
    if (declared_type.base == TypeBase::Callable && !arg_types.empty()) {
        if (declared_type.callable_return && declared_type.callable_return->base == TypeBase::Tensor &&
            arg_types[0].base == TypeBase::Tensor) {
            return Type::callable(Type::tensor(
                arg_types[0].tensor_dtype,
                declared_type.callable_return->tensor_shape_expr,
                arg_types[0].tensor_rank
            ));
        }
        return declared_type;
    }

    if (declared_type.base != TypeBase::Tensor || arg_types.empty() ||
        arg_types[0].base != TypeBase::Tensor) {
        return declared_type;
    }

    const Type& first = arg_types[0];
    if (callee == "reshape") {
        return Type::tensor(first.tensor_dtype, std::nullopt, std::nullopt);
    }
    if (callee == "sum" || callee == "mean") {
        return Type::tensor(first.tensor_dtype, std::nullopt, 1);
    }
    if (callee == "matmul") {
        return Type::tensor(
            first.tensor_dtype,
            declared_type.tensor_shape_expr,
            first.tensor_rank ? first.tensor_rank : declared_type.tensor_rank
        );
    }
    if (callee == "cross_entropy") {
        return Type::tensor(first.tensor_dtype, declared_type.tensor_shape_expr, declared_type.tensor_rank);
    }
    if (preserves_first_tensor_arg(callee)) {
        return first;
    }
    return Type::tensor(first.tensor_dtype, declared_type.tensor_shape_expr, first.tensor_rank);
}

Type default_unconstrained_numeric_type(Type type) {
    if (type.base == TypeBase::Int && !type.scalar_dtype) {
        return Type::int32();
    }
    if (type.base == TypeBase::Float && !type.scalar_dtype) {
        return Type::float64();
    }
    if (type.base == TypeBase::List || type.base == TypeBase::Tuple) {
        for (auto& element : type.elements) {
            element = default_unconstrained_numeric_type(std::move(element));
        }
    }
    return type;
}

Type merge_scalar_float_types(const Type& lhs, const Type& rhs) {
    auto rank = [](const Type& type) {
        if (type.scalar_dtype == "float64") {
            return 3;
        }
        if (type.scalar_dtype == "float16") {
            return 1;
        }
        return 2;
    };
    const int merged = std::max(rank(lhs), rank(rhs));
    if (merged == 3) {
        return Type::float64();
    }
    if (merged == 1) {
        return Type::float16();
    }
    if (!lhs.scalar_dtype && !rhs.scalar_dtype) {
        return Type::float_type();
    }
    return Type::float32();
}

Type merge_scalar_int_types(const Type& lhs, const Type& rhs) {
    auto rank = [](const Type& type) {
        if (type.scalar_dtype == "int64") {
            return 3;
        }
        if (type.scalar_dtype == "int16") {
            return 1;
        }
        return 2;
    };
    const int merged = std::max(rank(lhs), rank(rhs));
    if (merged == 3) {
        return Type::int64();
    }
    if (merged == 1) {
        return Type::int16();
    }
    if (!lhs.scalar_dtype && !rhs.scalar_dtype) {
        return Type::int_type();
    }
    return Type::int32();
}

int count_stage_sites(const Expr& expr);

int count_expr_list(const std::vector<ExprPtr>& exprs) {
    int count = 0;
    for (const auto& expr : exprs) {
        count += count_stage_sites(*expr);
    }
    return count;
}

int count_stage_sites(const Expr& expr) {
    return std::visit(
        [](const auto& value) -> int {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, CallExpr>) {
                return 1;
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                return count_stage_sites(*value.stage) + count_stage_sites(*value.count);
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                return count_stage_sites(*value.operand);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                return count_stage_sites(*value.lhs) + count_stage_sites(*value.rhs);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                return count_stage_sites(*value.then_expr) + count_stage_sites(*value.condition) +
                       count_stage_sites(*value.else_expr);
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                return count_expr_list(value.elements);
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                return count_expr_list(value.elements);
            } else {
                return 0;
            }
        },
        expr.kind
    );
}

std::optional<std::size_t> tuple_arity(const Expr& expr) {
    if (const auto* tuple = std::get_if<TupleExpr>(&expr.kind)) {
        return tuple->elements.size();
    }
    return std::nullopt;
}

void collect_expr_identifiers(const Expr& expr, std::set<std::string>& symbols);

void collect_expr_list_identifiers(const std::vector<ExprPtr>& exprs, std::set<std::string>& symbols) {
    for (const auto& expr : exprs) {
        collect_expr_identifiers(*expr, symbols);
    }
}

void collect_expr_identifiers(const Expr& expr, std::set<std::string>& symbols) {
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IdentifierExpr>) {
                if (value.name != "None") {
                    symbols.insert(value.name);
                }
            } else if constexpr (std::is_same_v<T, CallExpr>) {
                for (const auto& arg : value.args) {
                    collect_expr_identifiers(*arg.value, symbols);
                }
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                collect_expr_identifiers(*value.stage, symbols);
                collect_expr_identifiers(*value.count, symbols);
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                collect_expr_identifiers(*value.operand, symbols);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                collect_expr_identifiers(*value.lhs, symbols);
                collect_expr_identifiers(*value.rhs, symbols);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                collect_expr_identifiers(*value.then_expr, symbols);
                collect_expr_identifiers(*value.condition, symbols);
                collect_expr_identifiers(*value.else_expr, symbols);
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                collect_expr_list_identifiers(value.elements, symbols);
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                collect_expr_list_identifiers(value.elements, symbols);
            } else if constexpr (std::is_same_v<T, ArrowExpr>) {
                collect_expr_identifiers(*value.source, symbols);
                collect_expr_list_identifiers(value.stages, symbols);
            }
        },
        expr.kind
    );
}

void collect_stmt_symbols(const Stmt& stmt, std::set<std::string>& symbols) {
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, VarDecl>) {
                symbols.insert(value.name);
                if (value.init) {
                    collect_expr_identifiers(*value.init, symbols);
                }
            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                symbols.insert(value.name);
                collect_expr_identifiers(*value.value, symbols);
            } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                collect_expr_identifiers(*value.value, symbols);
            } else if constexpr (std::is_same_v<T, ExprStmt>) {
                collect_expr_identifiers(*value.value, symbols);
            } else if constexpr (std::is_same_v<T, ScopeStmt>) {
                for (const auto& inner : value.statements) {
                    collect_stmt_symbols(inner, symbols);
                }
            } else if constexpr (std::is_same_v<T, IfStmt>) {
                collect_expr_identifiers(*value.condition, symbols);
                collect_stmt_symbols(*value.then_stmt, symbols);
                for (const auto& branch : value.elifs) {
                    collect_expr_identifiers(*branch.condition, symbols);
                    collect_stmt_symbols(*branch.body, symbols);
                }
                if (value.else_stmt) {
                    collect_stmt_symbols(*value.else_stmt, symbols);
                }
            }
        },
        stmt.kind
    );
}

// Reviewed
bool is_train_config_field(const std::string& name) {
    return name == "backend" || name == "target" || name == "device" || name == "optimizer" ||
           name == "lr" || name == "learning_rate" || name == "objective" || name == "iteration";
}

// Reviewed
bool is_train_config(const Config& config) {
    return std::any_of(config.fields.begin(), config.fields.end(), [](const Field& field) {
        return field.type.base == TypeBase::Unknown &&
               (is_train_config_field(field.name) || field.name == "subtrain");
    });
}

} // namespace

SemanticAnalyzer::SemanticAnalyzer() {
    register_builtins();
}

std::optional<Diagnostic> SemanticAnalyzer::take_last_diagnostic() {
    auto diagnostic = last_diagnostic_;
    last_diagnostic_.reset();
    return diagnostic;
}

SemanticResult SemanticAnalyzer::analyze_with_info(const Program& program) {
    begin_analysis();
    if (auto diagnostic = analyze(program)) {
        return *diagnostic;
    }
    return semantic_info_;
}

void SemanticAnalyzer::begin_analysis() {
    scopes_.clear();
    functions_.clear();
    layers_.clear();
    configs_.clear();
    register_builtins();
    last_expr_type_ = Type::void_type();
    current_return_type_.reset();
    current_callable_has_return_ = false;
    current_callable_kind_ = CallableKind::None;
    current_callable_name_.reset();
    semantic_info_ = SemanticInfo{};
    last_diagnostic_.reset();

    for (const auto& spec : all_builtin_signatures()) {
        record_symbol(SemanticSymbol{
            spec.name,
            SemanticSymbolKind::BuiltinFunction,
            spec.return_type,
            false,
            0,
            std::nullopt,
            std::nullopt,
        });
    }
}

void SemanticAnalyzer::register_builtins() {
    for (const auto& spec : all_builtin_signatures()) {
        functions_[spec.name] =
            Signature{spec.name, spec.return_type, spec.arg_types, spec.min_arity, spec.max_arity};
    }
}
// U-Review
std::optional<Diagnostic> SemanticAnalyzer::analyze(const Program& program) {
    if (auto diagnostic = collect_configs(program)) {
        return diagnostic;
    }
    if (auto diagnostic = collect_layers(program)) {
        return diagnostic;
    }
    if (auto diagnostic = collect_functions(program)) {
        return diagnostic;
    }

    push_scope();
    for (const auto& stmt : program.globals) {
        if (auto diagnostic = analyze_stmt(stmt, program)) {
            return diagnostic;
        }
    }
    for (const auto& layer : program.layers) {
        if (auto diagnostic = visit_callable(
                layer.args,
                layer.return_type,
                layer.body,
                layer.span,
                layer.name,
                CallableKind::Layer,
                "Layer",
                program
            )) {
            return diagnostic;
        }
    }
    for (const auto& function : program.functions) {
        if (auto diagnostic = visit_callable(
                function.args,
                function.return_type,
                function.body,
                function.span,
                function.name,
                CallableKind::Function,
                "Function",
                program
            )) {
            return diagnostic;
        }
    }
    pop_scope();
    return std::nullopt;
}

// Reviewed
std::optional<Diagnostic> SemanticAnalyzer::collect_configs(const Program& program) {
    for (const auto& config : program.configs) {
        if (configs_.count(config.name) != 0) {
            return error(config.span, "Duplicate config '" + config.name + "'");
        }
        std::map<std::string, Type> fields;
        for (const auto& field : config.fields) {
            if (auto diagnostic = validate_declared_type(field.type, config.span)) {
                return diagnostic;
            }
            if (fields.count(field.name) != 0) {
                return error(config.span, "Duplicate field '" + field.name + "' in config '" + config.name + "'");
            }
            fields[field.name] = field.type;
        }
        configs_[config.name] = fields;
        record_symbol(SemanticSymbol{config.name, SemanticSymbolKind::Config, Type::void_type(), false, 0, std::nullopt, config.span});
        for (const auto& field : config.fields) {
            record_symbol(SemanticSymbol{field.name, SemanticSymbolKind::ConfigField, field.type, false, 0, config.name, config.span});
        }
        if (is_train_config(config)) {
            if (auto diagnostic = validate_train_config(config, program)) {
                return diagnostic;
            }
        }
    }
    return std::nullopt;
}
// Reviewed
std::optional<Diagnostic> SemanticAnalyzer::collect_layers(const Program& program) {
    for (const auto& layer : program.layers) {
        if (auto diagnostic = validate_declared_type(layer.return_type, layer.span)) {
            return diagnostic;
        }
        for (const auto& arg : layer.args) {
            if (auto diagnostic = validate_declared_type(arg.type, layer.span)) {
                return diagnostic;
            }
        }
        if (layers_.count(layer.name) != 0) {
            return error(layer.span, "Duplicate layer '" + layer.name + "'");
        }
        // need attention 
        if (functions_.count(layer.name) != 0) {
            return error(layer.span, "Layer '" + layer.name + "' conflicts with an existing function or builtin");
        }
        const auto min_arity = static_cast<std::size_t>(std::count_if(
            layer.args.begin(),
            layer.args.end(),
            [](const Arg& arg) { return !arg.default_value; }
        ));
        std::vector<Type> arg_types;
        for (const auto& arg : layer.args) {
            arg_types.push_back(arg.type);
        }
        layers_[layer.name] = Signature{layer.name, layer.return_type, arg_types, min_arity, layer.args.size()};
        record_symbol(SemanticSymbol{layer.name, SemanticSymbolKind::Layer, layer.return_type, false, 0, std::nullopt, layer.span});
    }
    return std::nullopt;
}
// Reviewed
std::optional<Diagnostic> SemanticAnalyzer::collect_functions(const Program& program) {
    for (const auto& function : program.functions) {
        if (auto diagnostic = validate_declared_type(function.return_type, function.span)) {
            return diagnostic;
        }
        for (const auto& arg : function.args) {
            if (auto diagnostic = validate_declared_type(arg.type, function.span)) {
                return diagnostic;
            }
        }
        if (functions_.count(function.name) != 0 || layers_.count(function.name) != 0) {
            return error(function.span, "Function '" + function.name + "' conflicts with an existing declaration");
        }
        const auto min_arity = static_cast<std::size_t>(std::count_if(
            function.args.begin(),
            function.args.end(),
            [](const Arg& arg) { return !arg.default_value; }
        ));
        std::vector<Type> arg_types;
        for (const auto& arg : function.args) {
            arg_types.push_back(arg.type);
        }
        functions_[function.name] =
            Signature{function.name, function.return_type, arg_types, min_arity, function.args.size()};
        record_symbol(SemanticSymbol{function.name, SemanticSymbolKind::Function, function.return_type, false, 0, std::nullopt, function.span});
    }
    return std::nullopt;
}
// U-Review
std::optional<Diagnostic> SemanticAnalyzer::analyze_stmt(const Stmt& stmt, const Program& program) {
    return std::visit(
        [&](const auto& value) -> std::optional<Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, ReturnStmt>) {
                auto analyzed = analyze_expr(*value.value, program);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                    return *diagnostic;
                }
                Type value_type = std::get<Type>(std::move(analyzed));
                current_callable_has_return_ = true;
                if (current_return_type_ && !is_compatible(*current_return_type_, value_type)) {
                    return error(
                        stmt.span,
                        "Return type mismatch. Expected " + type_to_string(*current_return_type_) +
                            ", got " + type_to_string(value_type)
                    );
                }
            } else if constexpr (std::is_same_v<T, ExprStmt>) {
                auto analyzed = analyze_expr(*value.value, program);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                    return *diagnostic;
                }
            } else if constexpr (std::is_same_v<T, VarDecl>) {
                Type final_type = value.type;
                if (value.init) {
                    auto analyzed = analyze_expr(*value.init, program);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                        return *diagnostic;
                    }
                    Type init_type = std::get<Type>(std::move(analyzed));
                    if (value.type.base == TypeBase::Unknown) {
                        final_type = default_unconstrained_numeric_type(std::move(init_type));
                    } else {
                        if (auto diagnostic = validate_declared_type(value.type, stmt.span)) {
                            return diagnostic;
                        }
                        if (!is_compatible(value.type, init_type)) {
                            return error(stmt.span, "Initialization type mismatch for '" + value.name + "'");
                        }
                    }
                } else if (auto diagnostic = validate_declared_type(value.type, stmt.span)) {
                    return diagnostic;
                }
                const auto kind = current_callable_name_ ? SemanticSymbolKind::Local : SemanticSymbolKind::Global;
                record_declaration(stmt.span, value.name, kind, final_type, value.is_mutable);
                return declare_var(value.name, std::move(final_type), value.is_mutable, kind, stmt.span);
            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                auto analyzed = analyze_expr(*value.value, program);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                    return *diagnostic;
                }
                Type value_type = std::get<Type>(std::move(analyzed));
                const Symbol* symbol = find_var(value.name);
                if (symbol == nullptr) {
                    return error(stmt.span, "Cannot assign to undefined variable '" + value.name + "'; declare it with 'let'");
                }
                if (!symbol->mutable_symbol) {
                    return error(stmt.span, "Cannot assign to immutable variable '" + value.name + "'; declare it with 'mut'");
                }
                if (!is_compatible(symbol->type, value_type)) {
                    return error(stmt.span, "Assignment type mismatch for '" + value.name + "'");
                }
                record_assignment(stmt.span, value.name, symbol->kind, symbol->type, value_type, symbol->mutable_symbol);
            } else if constexpr (std::is_same_v<T, ScopeStmt>) {
                push_scope();
                for (const auto& inner : value.statements) {
                    if (auto diagnostic = analyze_stmt(inner, program)) {
                        return diagnostic;
                    }
                }
                pop_scope();
            } else if constexpr (std::is_same_v<T, IfStmt>) {
                auto condition = analyze_expr(*value.condition, program);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                if (auto diagnostic = ensure_condition_type(std::get<Type>(condition), value.condition->span, "If condition")) {
                    return diagnostic;
                }
                if (auto diagnostic = analyze_stmt(*value.then_stmt, program)) {
                    return diagnostic;
                }
                for (const auto& branch : value.elifs) {
                    auto branch_condition = analyze_expr(*branch.condition, program);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&branch_condition)) {
                        return *diagnostic;
                    }
                    if (auto diagnostic = ensure_condition_type(std::get<Type>(branch_condition), branch.condition->span, "Elif condition")) {
                        return diagnostic;
                    }
                    if (auto diagnostic = analyze_stmt(*branch.body, program)) {
                        return diagnostic;
                    }
                }
                if (value.else_stmt) {
                    return analyze_stmt(*value.else_stmt, program);
                }
            }
            return std::nullopt;
        },
        stmt.kind
    );
}

std::variant<Type, Diagnostic> SemanticAnalyzer::analyze_expr(const Expr& expr, const Program& program) {
    auto result = std::visit(
        [&](const auto& value) -> std::variant<Type, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IntLiteral>) { // the compiler keeps only the IntLiteral branch. why we are using the constexpr
                return Type::int_type();
            } else if constexpr (std::is_same_v<T, FloatLiteral>) {
                return Type::float_type();
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                return Type::bool_type();
            } else if constexpr (std::is_same_v<T, StringLiteral>) {
                return Type::str_type();
            } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
                return visit_identifier(value.name, expr.span);
            } else if constexpr (std::is_same_v<T, CallExpr>) {
                return visit_call(value.callee, value.args, expr.span, program);
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                return error(expr.span, "Repeat suffix '[n]' is only valid inside arrow pipeline stages");
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                return visit_unary(*value.operand, value.op, expr.span, program);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                return visit_binary(*value.lhs, *value.rhs, value.op, expr.span, program);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                return visit_ternary(*value.then_expr, *value.condition, *value.else_expr, expr.span, program);
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                std::vector<Type> types;
                for (const auto& element : value.elements) {
                    auto analyzed = analyze_expr(*element, program);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                        return *diagnostic;
                    }
                    types.push_back(std::get<Type>(std::move(analyzed)));
                }
                return Type::tuple(std::move(types));
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                std::vector<Type> types;
                for (const auto& element : value.elements) {
                    auto analyzed = analyze_expr(*element, program);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                        return *diagnostic;
                    }
                    Type element_type = std::get<Type>(std::move(analyzed));
                    const bool known = std::any_of(types.begin(), types.end(), [&](const Type& known_type) {
                        return is_compatible(known_type, element_type);
                    });
                    if (!known) {
                        types.push_back(std::move(element_type));
                    }
                }
                return Type::list(std::move(types));
            } else if constexpr (std::is_same_v<T, ArrowExpr>) {
                return visit_arrow(*value.source, value.stages, program);
            }
        },
        expr.kind
    );
    if (const auto* type = std::get_if<Type>(&result)) {
        last_expr_type_ = *type;
        record_expr_type(expr, *type);
    }
    return result;
}

std::variant<Type, Diagnostic> SemanticAnalyzer::visit_identifier(const std::string& name, const SourceSpan& span) {
    if (name == "None") {
        return Type::void_type();
    }
    if (const Symbol* symbol = find_var(name)) {
        record_identifier(span, name, symbol->kind, symbol->type, symbol->mutable_symbol);
        return symbol->type;
    }
    if (configs_.count(name) != 0) {
        record_identifier(span, name, SemanticSymbolKind::Config, Type::void_type(), false);
        return Type::void_type();
    }
    return error(span, "Undefined variable '" + name + "'");
}

std::variant<Type, Diagnostic> SemanticAnalyzer::visit_call(
    const std::string& callee,
    const std::vector<CallArgument>& args,
    const SourceSpan& span,
    const Program& program
) {
    auto function = functions_.find(callee);
    if (function != functions_.end()) {
        if (auto diagnostic = ensure_call_allowed(callee, false, span)) {
            return *diagnostic;
        }
        if (auto diagnostic = validate_signature_arity(function->second, args.size(), span)) {
            return *diagnostic;
        }
        std::vector<Type> arg_types;
        for (std::size_t index = 0; index < args.size(); ++index) {
            auto analyzed = analyze_expr(*args[index].value, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            Type actual = std::get<Type>(std::move(analyzed));
            arg_types.push_back(actual);
            if (index < function->second.arg_types.size() &&
                !is_compatible(function->second.arg_types[index], actual)) {
                return error(span, "Argument " + std::to_string(index + 1) + " to '" + callee +
                                       "' has incompatible type. Expected " +
                                       type_to_string(function->second.arg_types[index]) + ", got " +
                                       type_to_string(actual));
            }
        }
        Type result = infer_call_result_type(callee, function->second.return_type, arg_types);
        record_call(span, callee, is_builtin_op(callee) ? SemanticCallTargetKind::BuiltinFunction
                                                        : SemanticCallTargetKind::Function,
                    result, false);
        return result;
    }

    auto layer = layers_.find(callee);
    if (layer != layers_.end()) {
        if (auto diagnostic = ensure_call_allowed(callee, true, span)) {
            return *diagnostic;
        }
        if (auto diagnostic = validate_signature_arity(layer->second, args.size(), span)) {
            return *diagnostic;
        }
        std::vector<Type> arg_types;
        for (std::size_t index = 0; index < args.size(); ++index) {
            auto analyzed = analyze_expr(*args[index].value, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            Type actual = std::get<Type>(std::move(analyzed));
            arg_types.push_back(actual);
            if (index < layer->second.arg_types.size() && !is_compatible(layer->second.arg_types[index], actual)) {
                return error(span, "Argument " + std::to_string(index + 1) + " to '" + callee + "' has incompatible type");
            }
        }
        Type result = infer_call_result_type(callee, layer->second.return_type, arg_types);
        record_call(span, callee, SemanticCallTargetKind::Layer, result, false);
        return result;
    }

    if (const Symbol* symbol = find_var(callee)) {
        if (!is_callable(symbol->type)) {
            return error(span, "Variable '" + callee + "' is not callable");
        }
        if (args.size() != 1) {
            return error(span, "Callable value '" + callee + "' must be invoked with exactly one pipeline/input argument");
        }
        auto input = analyze_expr(*args[0].value, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&input)) {
            return *diagnostic;
        }
        Type input_type = std::get<Type>(std::move(input));
        Type result = tensor_any();
        if (symbol->type.callable_return) {
            if (symbol->type.callable_return->base == TypeBase::Tensor && input_type.base == TypeBase::Tensor) {
                result = Type::tensor(input_type.tensor_dtype, symbol->type.callable_return->tensor_shape_expr, input_type.tensor_rank);
            } else {
                result = *symbol->type.callable_return;
            }
        }
        record_call(span, callee, SemanticCallTargetKind::CallableLocal, result, false);
        return result;
    }

    return error(span, "Undefined function or layer '" + callee + "'");
}

std::variant<Type, Diagnostic> SemanticAnalyzer::visit_unary(
    const Expr& operand,
    TokenType op,
    const SourceSpan& span,
    const Program& program
) {
    auto analyzed = analyze_expr(operand, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
    }
    Type operand_type = std::get<Type>(std::move(analyzed));
    if (op == TokenType::Bang) {
        if (auto diagnostic = ensure_condition_type(operand_type, span, "Unary '!'")) {
            return *diagnostic;
        }
        return Type::bool_type();
    }
    if (op == TokenType::Minus &&
        !(operand_type.base == TypeBase::Int || operand_type.base == TypeBase::Float ||
          operand_type.base == TypeBase::Tensor || operand_type.base == TypeBase::Unknown)) {
        return error(span, "Unary '-' expects int, float, or tensor operand");
    }
    return operand_type;
}

std::variant<Type, Diagnostic> SemanticAnalyzer::visit_binary(
    const Expr& lhs,
    const Expr& rhs,
    TokenType op,
    const SourceSpan& span,
    const Program& program
) {
    if (op == TokenType::Dot) {
        const auto* lhs_id = std::get_if<IdentifierExpr>(&lhs.kind);
        const auto* rhs_id = std::get_if<IdentifierExpr>(&rhs.kind);
        if (lhs_id && rhs_id) {
            auto config = configs_.find(lhs_id->name);
            if (config != configs_.end()) {
                auto field = config->second.find(rhs_id->name);
                if (field != config->second.end()) {
                    record_config_field_access(span, lhs_id->name, rhs_id->name, field->second);
                    return field->second;
                }
                return error(span, "Unknown field '" + rhs_id->name + "' on config '" + lhs_id->name + "'");
            }
        }
        return error(span, "Only config field access of the form config_name.field is supported");
    }

    auto left = analyze_expr(lhs, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
        return *diagnostic;
    }
    auto right = analyze_expr(rhs, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
        return *diagnostic;
    }
    Type lhs_type = std::get<Type>(std::move(left));
    Type rhs_type = std::get<Type>(std::move(right));

    if (op == TokenType::Plus || op == TokenType::Minus || op == TokenType::Star ||
        op == TokenType::Slash || op == TokenType::DoubleSlash) {
        if (lhs_type.base == TypeBase::Tensor || rhs_type.base == TypeBase::Tensor) {
            return merge_tensor_types(lhs_type, rhs_type);
        }
        if (lhs_type.base == TypeBase::Unknown || rhs_type.base == TypeBase::Unknown) {
            return Type::unknown();
        }
        if (lhs_type.base == TypeBase::Float || rhs_type.base == TypeBase::Float) {
            return merge_scalar_float_types(lhs_type, rhs_type);
        }
        if (lhs_type.base == TypeBase::Int && rhs_type.base == TypeBase::Int) {
            return merge_scalar_int_types(lhs_type, rhs_type);
        }
        return error(span, "Arithmetic operation has incompatible operand types");
    }
    if (op == TokenType::EqEq || op == TokenType::Neq) {
        if (!is_compatible(lhs_type, rhs_type)) {
            return error(span, "Comparison expects compatible operand types");
        }
        return Type::bool_type();
    }
    if (op == TokenType::Lt || op == TokenType::Gt || op == TokenType::LtEq || op == TokenType::GtEq) {
        if (!((lhs_type.base == TypeBase::Int || lhs_type.base == TypeBase::Float || lhs_type.base == TypeBase::Unknown) &&
              (rhs_type.base == TypeBase::Int || rhs_type.base == TypeBase::Float || rhs_type.base == TypeBase::Unknown))) {
            return error(span, "Ordered comparisons require int or float operands");
        }
        return Type::bool_type();
    }
    if (op == TokenType::AmpAmp || op == TokenType::PipePipe) {
        if (auto diagnostic = ensure_condition_type(lhs_type, lhs.span, "Logical operand")) {
            return *diagnostic;
        }
        if (auto diagnostic = ensure_condition_type(rhs_type, rhs.span, "Logical operand")) {
            return *diagnostic;
        }
        return Type::bool_type();
    }
    return error(span, "Unsupported binary operator in semantic analysis");
}

std::variant<Type, Diagnostic> SemanticAnalyzer::visit_ternary(
    const Expr& then_expr,
    const Expr& condition,
    const Expr& else_expr,
    const SourceSpan& span,
    const Program& program
) {
    auto condition_type = analyze_expr(condition, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&condition_type)) {
        return *diagnostic;
    }
    if (auto diagnostic = ensure_condition_type(std::get<Type>(condition_type), condition.span, "Ternary condition")) {
        return *diagnostic;
    }
    auto left = analyze_expr(then_expr, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
        return *diagnostic;
    }
    auto right = analyze_expr(else_expr, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
        return *diagnostic;
    }
    Type then_type = std::get<Type>(std::move(left));
    Type else_type = std::get<Type>(std::move(right));
    if (!is_compatible(then_type, else_type) && !is_compatible(else_type, then_type)) {
        return error(span, "Ternary branches must have compatible types");
    }
    return then_type.base == TypeBase::Void ? else_type : then_type;
}

std::variant<Type, Diagnostic> SemanticAnalyzer::visit_arrow(
    const Expr& source,
    const std::vector<ExprPtr>& stages,
    const Program& program
) {
    auto current = analyze_expr(source, program);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&current)) {
        return *diagnostic;
    }
    Type current_type = std::get<Type>(std::move(current));
    for (const auto& stage : stages) {
        auto next = analyze_stage(*stage, current_type, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&next)) {
            return *diagnostic;
        }
        current_type = std::get<Type>(std::move(next));
        record_expr_type(*stage, current_type);
    }
    return current_type;
}

std::variant<Type, Diagnostic> SemanticAnalyzer::analyze_stage(
    const Expr& expr,
    const Type& input_type,
    const Program& program
) {
    if (const auto* call = std::get_if<CallExpr>(&expr.kind)) {
        return analyze_arrow_call(call->callee, call->args, expr.span, input_type, program);
    }
    if (const auto* repeat = std::get_if<RepeatExpr>(&expr.kind)) {
        if (!std::holds_alternative<CallExpr>(repeat->stage->kind)) {
            return error(expr.span, "Repeated arrow stage must begin with a call");
        }
        auto count = analyze_expr(*repeat->count, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&count)) {
            return *diagnostic;
        }
        if (std::get<Type>(count).base != TypeBase::Int) {
            return error(repeat->count->span, "Repeated arrow stage count must have type int");
        }
        const auto& call = std::get<CallExpr>(repeat->stage->kind);
        return analyze_arrow_call(call.callee, call.args, repeat->stage->span, input_type, program);
    }
    if (const auto* unary = std::get_if<UnaryExpr>(&expr.kind)) {
        Type operand_type;
        if (count_stage_sites(*unary->operand) > 0) {
            auto analyzed = analyze_stage(*unary->operand, input_type, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            operand_type = std::get<Type>(std::move(analyzed));
        } else {
            auto analyzed = analyze_expr(*unary->operand, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            operand_type = std::get<Type>(std::move(analyzed));
        }
        if (unary->op == TokenType::Bang) {
            if (auto diagnostic = ensure_condition_type(operand_type, expr.span, "Unary '!'")) {
                return *diagnostic;
            }
            return Type::bool_type();
        }
        if (unary->op == TokenType::Minus &&
            !(operand_type.base == TypeBase::Int || operand_type.base == TypeBase::Float ||
              operand_type.base == TypeBase::Tensor || operand_type.base == TypeBase::Unknown)) {
            return error(expr.span, "Unary '-' expects int, float, or tensor operand");
        }
        return operand_type;
    }
    if (const auto* binary = std::get_if<BinaryExpr>(&expr.kind)) {
        Type lhs_type;
        if (count_stage_sites(*binary->lhs) > 0) {
            auto analyzed = analyze_stage(*binary->lhs, input_type, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            lhs_type = std::get<Type>(std::move(analyzed));
        } else {
            auto analyzed = analyze_expr(*binary->lhs, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            lhs_type = std::get<Type>(std::move(analyzed));
        }

        Type rhs_type;
        if (count_stage_sites(*binary->rhs) > 0) {
            auto analyzed = analyze_stage(*binary->rhs, input_type, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            rhs_type = std::get<Type>(std::move(analyzed));
        } else {
            auto analyzed = analyze_expr(*binary->rhs, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            rhs_type = std::get<Type>(std::move(analyzed));
        }

        if (binary->op == TokenType::Plus || binary->op == TokenType::Minus ||
            binary->op == TokenType::Star || binary->op == TokenType::Slash ||
            binary->op == TokenType::DoubleSlash) {
            if (lhs_type.base == TypeBase::Tensor || rhs_type.base == TypeBase::Tensor) {
                return merge_tensor_types(lhs_type, rhs_type);
            }
            if (lhs_type.base == TypeBase::Unknown || rhs_type.base == TypeBase::Unknown) {
                return Type::unknown();
            }
            if (lhs_type.base == TypeBase::Float || rhs_type.base == TypeBase::Float) {
                return merge_scalar_float_types(lhs_type, rhs_type);
            }
            if (lhs_type.base == TypeBase::Int && rhs_type.base == TypeBase::Int) {
                return merge_scalar_int_types(lhs_type, rhs_type);
            }
            return error(expr.span, "Arithmetic operation has incompatible operand types");
        }
        if (binary->op == TokenType::EqEq || binary->op == TokenType::Neq) {
            if (!is_compatible(lhs_type, rhs_type)) {
                return error(expr.span, "Comparison expects compatible operand types");
            }
            return Type::bool_type();
        }
        if (binary->op == TokenType::Lt || binary->op == TokenType::Gt ||
            binary->op == TokenType::LtEq || binary->op == TokenType::GtEq) {
            if (!((lhs_type.base == TypeBase::Int || lhs_type.base == TypeBase::Float ||
                   lhs_type.base == TypeBase::Unknown) &&
                  (rhs_type.base == TypeBase::Int || rhs_type.base == TypeBase::Float ||
                   rhs_type.base == TypeBase::Unknown))) {
                return error(expr.span, "Ordered comparisons require int or float operands");
            }
            return Type::bool_type();
        }
        if (binary->op == TokenType::AmpAmp || binary->op == TokenType::PipePipe) {
            if (auto diagnostic = ensure_condition_type(lhs_type, binary->lhs->span, "Logical operand")) {
                return *diagnostic;
            }
            if (auto diagnostic = ensure_condition_type(rhs_type, binary->rhs->span, "Logical operand")) {
                return *diagnostic;
            }
            return Type::bool_type();
        }
        return analyze_expr(expr, program);
    }
    if (const auto* ternary = std::get_if<TernaryExpr>(&expr.kind)) {
        auto condition = count_stage_sites(*ternary->condition) > 0
            ? analyze_stage(*ternary->condition, input_type, program)
            : analyze_expr(*ternary->condition, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
            return *diagnostic;
        }
        if (auto diagnostic = ensure_condition_type(std::get<Type>(condition), ternary->condition->span, "Ternary condition")) {
            return *diagnostic;
        }

        auto then_type = count_stage_sites(*ternary->then_expr) > 0
            ? analyze_stage(*ternary->then_expr, input_type, program)
            : analyze_expr(*ternary->then_expr, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&then_type)) {
            return *diagnostic;
        }
        auto else_type = count_stage_sites(*ternary->else_expr) > 0
            ? analyze_stage(*ternary->else_expr, input_type, program)
            : analyze_expr(*ternary->else_expr, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&else_type)) {
            return *diagnostic;
        }
        Type lhs = std::get<Type>(std::move(then_type));
        Type rhs = std::get<Type>(std::move(else_type));
        if (!is_compatible(lhs, rhs) && !is_compatible(rhs, lhs)) {
            return error(expr.span, "Ternary branches must have compatible types");
        }
        return lhs.base == TypeBase::Void ? rhs : lhs;
    }
    if (const auto* tuple = std::get_if<TupleExpr>(&expr.kind)) {
        std::vector<Type> types;
        for (const auto& element : tuple->elements) {
            auto analyzed = count_stage_sites(*element) > 0
                ? analyze_stage(*element, input_type, program)
                : analyze_expr(*element, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            types.push_back(std::get<Type>(std::move(analyzed)));
        }
        return Type::tuple(std::move(types));
    }
    if (const auto* list = std::get_if<ListExpr>(&expr.kind)) {
        std::vector<Type> types;
        for (const auto& element : list->elements) {
            auto analyzed = count_stage_sites(*element) > 0
                ? analyze_stage(*element, input_type, program)
                : analyze_expr(*element, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
                return *diagnostic;
            }
            Type element_type = std::get<Type>(std::move(analyzed));
            const bool known = std::any_of(types.begin(), types.end(), [&](const Type& known_type) {
                return is_compatible(known_type, element_type);
            });
            if (!known) {
                types.push_back(std::move(element_type));
            }
        }
        return Type::list(std::move(types));
    }
    if (count_stage_sites(expr) == 0) {
        return analyze_expr(expr, program);
    }
    return error(expr.span, "Unsupported compound arrow stage in semantic analysis");
}

std::variant<Type, Diagnostic> SemanticAnalyzer::analyze_arrow_call(
    const std::string& callee,
    const std::vector<CallArgument>& args,
    const SourceSpan& span,
    const Type& input_type,
    const Program& program
) {
    std::vector<Type> arg_types{input_type};
    for (const auto& arg : args) {
        auto analyzed = analyze_expr(*arg.value, program);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&analyzed)) {
            return *diagnostic;
        }
        arg_types.push_back(std::get<Type>(std::move(analyzed)));
    }

    auto function = functions_.find(callee);
    if (function != functions_.end()) {
        Type stage_type;
        if (function->second.return_type.base == TypeBase::Callable) {
            if (current_callable_kind_ == CallableKind::Function) {
                return error(span, "Arrow stage '" + callee + "' resolves to a callable/layer-like stage and cannot be used inside fn");
            }
            std::vector<Type> ctor_arg_types(arg_types.begin() + 1, arg_types.end());
            if (auto diagnostic = validate_signature_arity(function->second, ctor_arg_types.size(), span)) {
                return *diagnostic;
            }
            stage_type = infer_call_result_type(callee, function->second.return_type, ctor_arg_types);
        } else {
            if (auto diagnostic = validate_signature_arity(function->second, arg_types.size(), span)) {
                return *diagnostic;
            }
            stage_type = infer_call_result_type(callee, function->second.return_type, arg_types);
        }
        auto unwrapped = unwrap_callable_stage(stage_type, input_type);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&unwrapped)) {
            return *diagnostic;
        }
        Type result = std::get<Type>(std::move(unwrapped));
        record_call(span, callee, is_builtin_op(callee) ? SemanticCallTargetKind::BuiltinFunction
                                                        : SemanticCallTargetKind::Function,
                    result, true);
        return result;
    }

    auto layer = layers_.find(callee);
    if (layer != layers_.end()) {
        if (auto diagnostic = ensure_call_allowed(callee, true, span)) {
            return *diagnostic;
        }
        Type result = layer->second.return_type;
        record_call(span, callee, SemanticCallTargetKind::Layer, result, true);
        return result;
    }

    if (const Symbol* symbol = find_var(callee)) {
        if (!is_callable(symbol->type)) {
            return error(span, "Arrow stage '" + callee + "' is not callable");
        }
        if (current_callable_kind_ == CallableKind::Function) {
            return error(
                span,
                "Arrow stage '" + callee +
                    "' is a callable/layer value and cannot be used inside fn; fn arrow stages must stay function-only"
            );
        }
        auto unwrapped = unwrap_callable_stage(symbol->type, input_type);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&unwrapped)) {
            return *diagnostic;
        }
        Type result = std::get<Type>(std::move(unwrapped));
        record_call(span, callee, SemanticCallTargetKind::CallableLocal, result, true);
        return result;
    }
    return error(span, "Arrow stage '" + callee + "' is not callable");
}

std::variant<Type, Diagnostic> SemanticAnalyzer::unwrap_callable_stage(const Type& stage_type, const Type& input_type) {
    if (stage_type.base != TypeBase::Callable) {
        return stage_type;
    }
    if (!stage_type.callable_return) {
        return tensor_any();
    }
    if (stage_type.callable_return->base == TypeBase::Tensor && input_type.base == TypeBase::Tensor) {
        return Type::tensor(input_type.tensor_dtype, stage_type.callable_return->tensor_shape_expr, input_type.tensor_rank);
    }
    return *stage_type.callable_return;
}

std::optional<Diagnostic> SemanticAnalyzer::validate_train_config(const Config& config, const Program& program) {
    std::optional<std::size_t> variant_count;
    for (const auto& field : config.fields) {
        if (field.name == "subtrain") {
            const SourceSpan span = field.init ? field.init->span : config.span;
            return error(
                span,
                "Field 'subtrain' is no longer supported; use tuple-valued fields on the training config instead"
            );
        }
        if (!is_train_config_field(field.name) || !field.init) {
            continue;
        }
        auto arity = tuple_arity(*field.init);
        if (!arity) {
            continue;
        }
        if (*arity == 0) {
            return error(
                field.init->span,
                "Training config field '" + field.name + "' cannot use an empty tuple"
            );
        }
        if (variant_count && *variant_count != *arity) {
            return error(
                field.init->span,
                "Tuple-valued training config fields must have the same length; use scalar values to broadcast"
            );
        }
        if (!variant_count) {
            variant_count = *arity;
        }
    }

    std::vector<std::string> model_symbols = collect_model_symbols(program);
    for (const auto& field : config.fields) {
        if (field.name != "objective" || !field.init) {
            continue;
        }

        std::vector<const Expr*> objective_exprs;
        if (const auto* tuple = std::get_if<TupleExpr>(&field.init->kind)) {
            for (const auto& element : tuple->elements) {
                objective_exprs.push_back(element.get());
            }
        } else {
            objective_exprs.push_back(field.init.get());
        }

        for (const Expr* expr : objective_exprs) {
            const auto* identifier = std::get_if<IdentifierExpr>(&expr->kind);
            if (identifier == nullptr) {
                return error(
                    expr->span,
                    "Field 'objective' must reference a named tensor root, not a string literal or arbitrary expression"
                );
            }
            if (!model_symbols.empty() &&
                std::find(model_symbols.begin(), model_symbols.end(), identifier->name) ==
                    model_symbols.end()) {
                return error(
                    expr->span,
                    "Field 'objective' references unknown model root '" + identifier->name + "'"
                );
            }
        }
    }

    return std::nullopt;
}

std::vector<std::string> SemanticAnalyzer::collect_model_symbols(const Program& program) const {
    const Layer* model = nullptr;
    for (const auto& layer : program.layers) {
        if (layer.name == "model") {
            model = &layer;
            break;
        }
    }
    if (model == nullptr) {
        return {};
    }

    std::set<std::string> symbols;
    for (const auto& arg : model->args) {
        symbols.insert(arg.name);
    }
    collect_stmt_symbols(model->body, symbols);
    return {symbols.begin(), symbols.end()};
}

std::optional<Diagnostic> SemanticAnalyzer::visit_callable(
    const std::vector<Arg>& args,
    const Type& return_type,
    const Stmt& body,
    const SourceSpan& span,
    const std::string& name,
    CallableKind kind,
    const char* label,
    const Program& program
) {
    const auto previous_return = current_return_type_;
    const auto previous_has_return = current_callable_has_return_;
    const auto previous_kind = current_callable_kind_;
    const auto previous_name = current_callable_name_;
    current_return_type_ = return_type;
    current_callable_has_return_ = false;
    current_callable_kind_ = kind;
    current_callable_name_ = name;

    push_scope();
    for (const auto& arg : args) {
        if (arg.default_value) {
            auto default_type = analyze_expr(*arg.default_value, program);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&default_type)) {
                return *diagnostic;
            }
            if (!is_compatible(arg.type, std::get<Type>(default_type))) {
                return error(span, "Default value for argument '" + arg.name + "' has incompatible type");
            }
        }
        if (auto diagnostic = declare_var(arg.name, arg.type, arg.is_mutable, SemanticSymbolKind::Parameter, span)) {
            return diagnostic;
        }
    }
    if (auto diagnostic = analyze_stmt(body, program)) {
        return diagnostic;
    }
    pop_scope();

    if (return_type.base != TypeBase::Void && !current_callable_has_return_) {
        return error(span, std::string(label) + " '" + name + "' is missing a return statement");
    }

    current_return_type_ = previous_return;
    current_callable_has_return_ = previous_has_return;
    current_callable_kind_ = previous_kind;
    current_callable_name_ = previous_name;
    return std::nullopt;
}

std::optional<Diagnostic> SemanticAnalyzer::declare_var(
    const std::string& name,
    Type type,
    bool mutable_symbol,
    SemanticSymbolKind kind,
    const SourceSpan& span
) {
    if (scopes_.empty()) {
        push_scope();
    }
    auto& scope = scopes_.back();
    if (scope.count(name) != 0) {
        return error(span, "Variable '" + name + "' already declared in this scope");
    }
    Symbol symbol;
    symbol.type = type;
    symbol.mutable_symbol = mutable_symbol;
    symbol.is_callable = is_callable(type);
    if (type.callable_return) {
        symbol.callable_return_type = *type.callable_return;
    }
    symbol.kind = kind;
    scope[name] = symbol;
    record_symbol(SemanticSymbol{name, kind, type, mutable_symbol, scopes_.size(), current_callable_name_, span});
    return std::nullopt;
}

const Symbol* SemanticAnalyzer::find_var(const std::string& name) const {
    for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); ++scope) {
        auto found = scope->find(name);
        if (found != scope->end()) {
            return &found->second;
        }
    }
    return nullptr;
}

bool SemanticAnalyzer::is_compatible(const Type& target, const Type& source) const {
    if (target.base == TypeBase::Unknown || source.base == TypeBase::Unknown) {
        return true;
    }
    if (target.base != source.base) {
        return false;
    }
    if ((target.base == TypeBase::Float || target.base == TypeBase::Int) &&
        target.scalar_dtype && source.scalar_dtype && target.scalar_dtype != source.scalar_dtype) {
        return false;
    }
    if (target.base == TypeBase::Tensor) {
        if (target.tensor_dtype && source.tensor_dtype && target.tensor_dtype != source.tensor_dtype) {
            return false;
        }
        if (target.tensor_shape_expr && source.tensor_shape_expr &&
            target.tensor_shape_expr != source.tensor_shape_expr) {
            return false;
        }
        if (target.tensor_rank && source.tensor_rank && target.tensor_rank != source.tensor_rank) {
            return false;
        }
    }
    if (target.base == TypeBase::Tuple) {
        if (target.elements.size() != source.elements.size()) {
            return false;
        }
        for (std::size_t index = 0; index < target.elements.size(); ++index) {
            if (!is_compatible(target.elements[index], source.elements[index])) {
                return false;
            }
        }
    }
    if (target.base == TypeBase::List) {
        if (target.elements.empty() || source.elements.empty()) {
            return true;
        }
        return std::all_of(source.elements.begin(), source.elements.end(), [&](const Type& rhs) {
            return std::any_of(target.elements.begin(), target.elements.end(), [&](const Type& lhs) {
                return is_compatible(lhs, rhs);
            });
        });
    }
    if (target.base == TypeBase::Callable) {
        if (!target.callable_return || !source.callable_return) {
            return !target.callable_return && !source.callable_return;
        }
        return is_compatible(*target.callable_return, *source.callable_return);
    }
    return true;
}

Type SemanticAnalyzer::merge_tensor_types(const Type& lhs, const Type& rhs) const {
    if (lhs.base != TypeBase::Tensor) {
        return rhs;
    }
    if (rhs.base != TypeBase::Tensor) {
        return lhs;
    }
    return Type::tensor(
        lhs.tensor_dtype ? lhs.tensor_dtype : rhs.tensor_dtype,
        lhs.tensor_shape_expr ? lhs.tensor_shape_expr : rhs.tensor_shape_expr,
        lhs.tensor_rank ? lhs.tensor_rank : rhs.tensor_rank
    );
}

//reviewed
std::optional<Diagnostic> SemanticAnalyzer::validate_declared_type(const Type& type, const SourceSpan& span) {
    // static for this vector persist between calls.
    static const std::vector<std::string> tensor_dtypes = {
        "float16", "float32", "float64", "bfloat16", "int16", "int32", "int64",
    };
    if (type.base == TypeBase::Int && type.scalar_dtype &&
        !(*type.scalar_dtype == "int16" || *type.scalar_dtype == "int32" || *type.scalar_dtype == "int64")) {
        return error(span, "Unsupported scalar integer type '" + *type.scalar_dtype + "'");
    }
    if (type.base == TypeBase::Float && type.scalar_dtype &&
        !(*type.scalar_dtype == "float16" || *type.scalar_dtype == "float32" || *type.scalar_dtype == "float64")) {
        return error(span, "Unsupported scalar float type '" + *type.scalar_dtype + "'");
    }
    if (type.base == TypeBase::Tensor && type.tensor_dtype &&
        std::find(tensor_dtypes.begin(), tensor_dtypes.end(), *type.tensor_dtype) == tensor_dtypes.end()) {
        return error(span, "Unsupported tensor dtype '" + *type.tensor_dtype + "'");
    }
    // not sure here
    for (const auto& element : type.elements) {
        if (auto diagnostic = validate_declared_type(element, span)) {
            return diagnostic;
        }
    }
    if (type.callable_return) {
        return validate_declared_type(*type.callable_return, span);
    }
    return std::nullopt;
}

std::optional<Diagnostic> SemanticAnalyzer::validate_signature_arity(
    const Signature& signature,
    std::size_t actual_arity,
    const SourceSpan& span
) {
    if (actual_arity < signature.min_arity || actual_arity > signature.max_arity) {
        std::ostringstream expected;
        if (signature.min_arity == signature.max_arity) {
            expected << signature.min_arity;
        } else {
            expected << signature.min_arity << " to " << signature.max_arity;
        }
        return error(
            span,
            "Call to '" + signature.name + "' expects " + expected.str() +
                " argument(s), but got " + std::to_string(actual_arity)
        );
    }
    return std::nullopt;
}

std::optional<Diagnostic> SemanticAnalyzer::ensure_condition_type(
    const Type& type,
    const SourceSpan& span,
    const std::string& context
) {
    if (!(type.base == TypeBase::Bool || type.base == TypeBase::Unknown)) {
        return error(span, context + " must have type bool, but got " + type_to_string(type));
    }
    return std::nullopt;
}

std::optional<Diagnostic> SemanticAnalyzer::ensure_call_allowed(
    const std::string& callee,
    bool is_layer,
    const SourceSpan& span
) {
    if (is_layer && current_callable_kind_ == CallableKind::Function) {
        return error(span, "Function '" + callee + "' is a layer and cannot be called from fn");
    }
    return std::nullopt;
}

Diagnostic SemanticAnalyzer::error(const SourceSpan& span, const std::string& message) {
    Diagnostic diagnostic = Diagnostic::error("semantic", "S0001", message).with_source_span(span);
    last_diagnostic_ = diagnostic;
    return diagnostic;
}
// Reviewed
void SemanticAnalyzer::push_scope() {
    scopes_.push_back({});
}

void SemanticAnalyzer::pop_scope() {
    if (!scopes_.empty()) {
        scopes_.pop_back();
    }
}
//reviewed
void SemanticAnalyzer::record_symbol(SemanticSymbol symbol) {
    semantic_info_.symbols.push_back(std::move(symbol));
}

void SemanticAnalyzer::record_expr_type(const Expr& expr, Type type) {
    semantic_info_.exprs.push_back(SemanticExprInfo{expr.span, std::move(type), current_callable_name_});
}

void SemanticAnalyzer::record_identifier(
    const SourceSpan& span,
    const std::string& name,
    SemanticSymbolKind target,
    Type type,
    bool mutable_symbol
) {
    semantic_info_.identifiers.push_back(
        SemanticIdentifierInfo{span, name, target, std::move(type), mutable_symbol, current_callable_name_}
    );
}

void SemanticAnalyzer::record_assignment(
    const SourceSpan& span,
    const std::string& name,
    SemanticSymbolKind target,
    Type target_type,
    Type value_type,
    bool mutable_symbol
) {
    semantic_info_.assignments.push_back(SemanticAssignmentInfo{
        span,
        name,
        target,
        std::move(target_type),
        std::move(value_type),
        mutable_symbol,
        current_callable_name_,
    });
}

void SemanticAnalyzer::record_config_field_access(
    const SourceSpan& span,
    const std::string& config_name,
    const std::string& field_name,
    Type field_type
) {
    semantic_info_.config_field_accesses.push_back(
        SemanticConfigFieldAccessInfo{span, config_name, field_name, std::move(field_type), current_callable_name_}
    );
}

void SemanticAnalyzer::record_declaration(
    const SourceSpan& span,
    const std::string& name,
    SemanticSymbolKind kind,
    Type final_type,
    bool mutable_symbol
) {
    semantic_info_.declarations.push_back(
        SemanticDeclarationInfo{span, name, kind, std::move(final_type), mutable_symbol, current_callable_name_}
    );
}

void SemanticAnalyzer::record_call(
    const SourceSpan& span,
    const std::string& callee,
    SemanticCallTargetKind target,
    Type result_type,
    bool arrow_stage
) {
    semantic_info_.calls.push_back(
        SemanticCallInfo{span, callee, target, std::move(result_type), current_callable_name_, arrow_stage}
    );
}

const char* semantic_symbol_kind_name(SemanticSymbolKind kind) {
    switch (kind) {
        case SemanticSymbolKind::BuiltinFunction:
            return "builtin";
        case SemanticSymbolKind::Function:
            return "function";
        case SemanticSymbolKind::Layer:
            return "layer";
        case SemanticSymbolKind::Config:
            return "config";
        case SemanticSymbolKind::ConfigField:
            return "config_field";
        case SemanticSymbolKind::Global:
            return "global";
        case SemanticSymbolKind::Parameter:
            return "parameter";
        case SemanticSymbolKind::Local:
            return "local";
    }
    return "local";
}

const char* semantic_call_target_kind_name(SemanticCallTargetKind kind) {
    switch (kind) {
        case SemanticCallTargetKind::BuiltinFunction:
            return "builtin";
        case SemanticCallTargetKind::Function:
            return "function";
        case SemanticCallTargetKind::Layer:
            return "layer";
        case SemanticCallTargetKind::CallableLocal:
            return "callable_local";
    }
    return "function";
}

std::string semantic_info_summary(const SemanticInfo& info, const Program& program) {
    std::ostringstream out;
    out << "ok\n";
    out << "configs=" << program.configs.size() << '\n';
    out << "layers=" << program.layers.size() << '\n';
    out << "functions=" << program.functions.size() << '\n';
    out << "globals=" << program.globals.size() << '\n';
    out << "symbols=" << info.symbols.size() << '\n';
    out << "exprs=" << info.exprs.size() << '\n';
    out << "identifiers=" << info.identifiers.size() << '\n';
    out << "assignments=" << info.assignments.size() << '\n';
    out << "config_field_accesses=" << info.config_field_accesses.size() << '\n';
    out << "declarations=" << info.declarations.size() << '\n';
    out << "calls=" << info.calls.size();
    return out.str();
}
