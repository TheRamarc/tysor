#include "frontend_ir.h"

#include "ops.h"
#include "parser.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <utility>

namespace {

bool same_span(const SourceSpan& lhs, const SourceSpan& rhs) {
    return lhs.line == rhs.line && lhs.column == rhs.column;
}

bool owner_matches(const std::optional<std::string>& lhs, const std::optional<std::string>& rhs) {
    return lhs == rhs;
}

bool is_train_config_field(const std::string& name) {
    return name == "backend" || name == "target" || name == "device" || name == "optimizer" ||
           name == "lr" || name == "learning_rate" || name == "objective" || name == "iteration";
}

bool is_train_config(const Config& config) {
    return std::any_of(config.fields.begin(), config.fields.end(), [](const Field& field) {
        return field.type.base == TypeBase::Unknown &&
               (is_train_config_field(field.name) || field.name == "subtrain");
    });
}

int count_arrow_stage_sites(const Expr& expr);

int count_expr_list(const std::vector<ExprPtr>& exprs) {
    int count = 0;
    for (const auto& expr : exprs) {
        count += count_arrow_stage_sites(*expr);
    }
    return count;
}

int count_arrow_stage_sites(const Expr& expr) {
    return std::visit(
        [](const auto& value) -> int {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, CallExpr>) {
                return 1;
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                return count_arrow_stage_sites(*value.stage) + count_arrow_stage_sites(*value.count);
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                return count_arrow_stage_sites(*value.operand);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                return count_arrow_stage_sites(*value.lhs) + count_arrow_stage_sites(*value.rhs);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                return count_arrow_stage_sites(*value.then_expr) +
                       count_arrow_stage_sites(*value.condition) +
                       count_arrow_stage_sites(*value.else_expr);
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

bool is_none(const FeValue& value) {
    return std::holds_alternative<std::monostate>(value.value);
}

std::variant<double, Diagnostic> as_double(const FeValue& value, const SourceSpan& span) {
    if (const auto* int_value = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<double>(*int_value);
    }
    if (const auto* float_value = std::get_if<double>(&value.value)) {
        return *float_value;
    }
    return Diagnostic::error("frontend_ir", "F0001", "Expected numeric constant").with_source_span(span);
}

std::variant<std::int64_t, Diagnostic> as_int(const FeValue& value, const SourceSpan& span) {
    if (const auto* int_value = std::get_if<std::int64_t>(&value.value)) {
        return *int_value;
    }
    return Diagnostic::error("frontend_ir", "F0001", "Expected integer constant").with_source_span(span);
}

std::variant<bool, Diagnostic> as_bool(const FeValue& value, const SourceSpan& span) {
    if (const auto* bool_value = std::get_if<bool>(&value.value)) {
        return *bool_value;
    }
    return Diagnostic::error("frontend_ir", "F0001", "Expected boolean constant").with_source_span(span);
}

std::optional<FeValue> pick_broadcast_value(const std::vector<FeValue>& values, std::size_t index) {
    if (values.empty()) {
        return std::nullopt;
    }
    if (index < values.size()) {
        return values[index];
    }
    if (values.size() == 1) {
        return values.front();
    }
    return std::nullopt;
}

std::optional<std::string> pick_broadcast_string(const std::vector<std::string>& values, std::size_t index) {
    if (values.empty()) {
        return std::nullopt;
    }
    if (index < values.size()) {
        return values[index];
    }
    if (values.size() == 1) {
        return values.front();
    }
    return std::nullopt;
}

std::string fe_type_to_string(const FeType& type) {
    switch (type.kind) {
        case FeTypeKind::Unknown:
            return "unknown";
        case FeTypeKind::Int:
            return type.scalar_dtype.value_or("int");
        case FeTypeKind::Float:
            return type.scalar_dtype.value_or("float");
        case FeTypeKind::Bool:
            return "bool";
        case FeTypeKind::Str:
            return "str";
        case FeTypeKind::Tensor:
            if (type.tensor_dtype && type.tensor_shape_expr) {
                return "tensor[" + *type.tensor_dtype + ", " + *type.tensor_shape_expr + "]";
            }
            if (type.tensor_dtype) {
                return "tensor[" + *type.tensor_dtype + "]";
            }
            if (type.tensor_shape_expr) {
                return "tensor[" + *type.tensor_shape_expr + "]";
            }
            return "tensor";
        case FeTypeKind::Tuple: {
            std::ostringstream out;
            out << '(';
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    out << ", ";
                }
                out << fe_type_to_string(type.elements[index]);
            }
            out << ')';
            return out.str();
        }
        case FeTypeKind::List: {
            std::ostringstream out;
            out << '[';
            for (std::size_t index = 0; index < type.elements.size(); ++index) {
                if (index != 0) {
                    out << ", ";
                }
                out << fe_type_to_string(type.elements[index]);
            }
            out << ']';
            return out.str();
        }
        case FeTypeKind::Callable:
            return "callable -> " + (type.callable_return ? fe_type_to_string(*type.callable_return) : "void");
        case FeTypeKind::None:
            return "none";
    }
    return "unknown";
}

std::string fe_value_to_string(const FeValue& value) {
    return std::visit(
        [](const auto& inner) -> std::string {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return "None";
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(inner);
            } else if constexpr (std::is_same_v<T, double>) {
                std::ostringstream out;
                out << inner;
                return out.str();
            } else if constexpr (std::is_same_v<T, bool>) {
                return inner ? "true" : "false";
            } else if constexpr (std::is_same_v<T, std::string>) {
                return '"' + inner + '"';
            } else if constexpr (std::is_same_v<T, FeTupleValue>) {
                std::ostringstream out;
                out << '[';
                for (std::size_t index = 0; index < inner.values.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    out << fe_value_to_string(inner.values[index]);
                }
                out << ']';
                return out.str();
            } else if constexpr (std::is_same_v<T, FeListValue>) {
                std::ostringstream out;
                out << '[';
                for (std::size_t index = 0; index < inner.values.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    out << fe_value_to_string(inner.values[index]);
                }
                out << ']';
                return out.str();
            }
        },
        value.value
    );
}

std::string fe_binary_op_name(FeBinaryOp op) {
    switch (op) {
        case FeBinaryOp::Add:
            return "+";
        case FeBinaryOp::Sub:
            return "-";
        case FeBinaryOp::Mul:
            return "*";
        case FeBinaryOp::Div:
            return "/";
        case FeBinaryOp::FloorDiv:
            return "//";
        case FeBinaryOp::Eq:
            return "==";
        case FeBinaryOp::NotEq:
            return "!=";
        case FeBinaryOp::Lt:
            return "<";
        case FeBinaryOp::Gt:
            return ">";
        case FeBinaryOp::LtEq:
            return "<=";
        case FeBinaryOp::GtEq:
            return ">=";
        case FeBinaryOp::And:
            return "&&";
        case FeBinaryOp::Or:
            return "||";
        case FeBinaryOp::Not:
            return "!";
    }
    return "?";
}

void append_expr(std::ostringstream& out, const FeExprPtr& expr);

void append_args(std::ostringstream& out, const std::vector<FeCallArg>& args) {
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        if (args[index].name) {
            out << *args[index].name << ": ";
        }
        append_expr(out, args[index].value);
    }
}

void append_expr(std::ostringstream& out, const FeExprPtr& expr) {
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, FeConstantExpr>) {
                out << fe_value_to_string(value.value);
            } else if constexpr (std::is_same_v<T, FeVarExpr>) {
                out << value.symbol;
            } else if constexpr (std::is_same_v<T, FeCallExpr>) {
                out << value.callee << '(';
                append_args(out, value.args);
                out << ')';
            } else if constexpr (std::is_same_v<T, FeLayerCtorExpr>) {
                out << "layer_ctor " << value.callee << '(';
                append_args(out, value.args);
                out << ')';
            } else if constexpr (std::is_same_v<T, FeApplyExpr>) {
                out << "apply(";
                append_expr(out, value.callee);
                if (!value.args.empty()) {
                    out << ", ";
                    append_args(out, value.args);
                }
                out << ')';
            } else if constexpr (std::is_same_v<T, FeTupleExpr>) {
                out << '(';
                for (std::size_t index = 0; index < value.elements.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    append_expr(out, value.elements[index]);
                }
                out << ')';
            } else if constexpr (std::is_same_v<T, FeListExpr>) {
                out << '[';
                for (std::size_t index = 0; index < value.elements.size(); ++index) {
                    if (index != 0) {
                        out << ", ";
                    }
                    append_expr(out, value.elements[index]);
                }
                out << ']';
            } else if constexpr (std::is_same_v<T, FeBinaryExpr>) {
                out << '(';
                append_expr(out, value.lhs);
                out << ' ' << fe_binary_op_name(value.op) << ' ';
                append_expr(out, value.rhs);
                out << ')';
            } else if constexpr (std::is_same_v<T, FeIfThenElseExpr>) {
                out << "if ";
                append_expr(out, value.condition);
                out << " then ";
                append_expr(out, value.then_expr);
                out << " else ";
                append_expr(out, value.else_expr);
            }
        },
        expr->kind
    );
}

void append_stmt(std::ostringstream& out, const FeStmt& stmt, std::size_t indent) {
    const std::string pad(indent, ' ');
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            out << pad;
            if constexpr (std::is_same_v<T, FeVarDeclStmt>) {
                out << (value.mutable_symbol ? "let mut " : "let ") << value.name << ": "
                    << fe_type_to_string(value.type);
                if (value.has_value && value.value) {
                    out << " = ";
                    append_expr(out, value.value);
                }
                out << '\n';
            } else if constexpr (std::is_same_v<T, FeAssignStmt>) {
                out << "assign " << value.name << " = ";
                append_expr(out, value.value);
                out << '\n';
            } else if constexpr (std::is_same_v<T, FeReturnStmt>) {
                out << "return ";
                append_expr(out, value.value);
                out << '\n';
            } else if constexpr (std::is_same_v<T, FeExprStmt>) {
                out << "expr ";
                append_expr(out, value.value);
                out << '\n';
            } else if constexpr (std::is_same_v<T, FeIfStmt>) {
                out << "if ";
                append_expr(out, value.condition);
                out << '\n';
                for (const auto& child : value.then_body) {
                    append_stmt(out, child, indent + 2);
                }
                for (const auto& elif : value.elif_bodies) {
                    out << pad << "elif ";
                    append_expr(out, elif.condition);
                    out << '\n';
                    for (const auto& child : elif.body) {
                        append_stmt(out, child, indent + 2);
                    }
                }
                if (!value.else_body.empty()) {
                    out << pad << "else\n";
                    for (const auto& child : value.else_body) {
                        append_stmt(out, child, indent + 2);
                    }
                }
            }
        },
        stmt.kind
    );
}

} // namespace

FeType FeType::unknown() {
    FeType type;
    type.kind = FeTypeKind::Unknown;
    return type;
}

FeType FeType::int_type() {
    FeType type;
    type.kind = FeTypeKind::Int;
    return type;
}

FeType FeType::int16() {
    FeType type = int_type();
    type.scalar_dtype = "int16";
    return type;
}

FeType FeType::int32() {
    FeType type = int_type();
    type.scalar_dtype = "int32";
    return type;
}

FeType FeType::int64() {
    FeType type = int_type();
    type.scalar_dtype = "int64";
    return type;
}

FeType FeType::float_type() {
    FeType type;
    type.kind = FeTypeKind::Float;
    return type;
}

FeType FeType::float16() {
    FeType type = float_type();
    type.scalar_dtype = "float16";
    return type;
}

FeType FeType::float32() {
    FeType type = float_type();
    type.scalar_dtype = "float32";
    return type;
}

FeType FeType::float64() {
    FeType type = float_type();
    type.scalar_dtype = "float64";
    return type;
}

FeType FeType::str_type() {
    FeType type;
    type.kind = FeTypeKind::Str;
    return type;
}

FeType FeType::bool_type() {
    FeType type;
    type.kind = FeTypeKind::Bool;
    return type;
}

FeType FeType::tensor(
    std::optional<std::string> dtype,
    std::optional<std::string> shape_expr,
    std::optional<std::size_t> rank
) {
    FeType type;
    type.kind = FeTypeKind::Tensor;
    type.tensor_dtype = std::move(dtype);
    type.tensor_shape_expr = std::move(shape_expr);
    type.tensor_rank = rank;
    return type;
}

FeType FeType::tuple(std::vector<FeType> elements) {
    FeType type;
    type.kind = FeTypeKind::Tuple;
    type.elements = std::move(elements);
    return type;
}

FeType FeType::list(std::vector<FeType> elements) {
    FeType type;
    type.kind = FeTypeKind::List;
    type.elements = std::move(elements);
    return type;
}

FeType FeType::callable(FeType return_type) {
    FeType type;
    type.kind = FeTypeKind::Callable;
    type.callable_return = std::make_shared<FeType>(std::move(return_type));
    return type;
}

FeType FeType::void_type() {
    return FeType{};
}

FeType FeType::none() {
    FeType type;
    type.kind = FeTypeKind::None;
    return type;
}

FeValue FeValue::none() {
    return FeValue{};
}

FeValue FeValue::int_value(std::int64_t value) {
    FeValue result;
    result.value = value;
    return result;
}

FeValue FeValue::float_value(double value) {
    FeValue result;
    result.value = value;
    return result;
}

FeValue FeValue::bool_value(bool value) {
    FeValue result;
    result.value = value;
    return result;
}

FeValue FeValue::string_value(std::string value) {
    FeValue result;
    result.value = std::move(value);
    return result;
}

FeValue FeValue::tuple_value(std::vector<FeValue> values) {
    FeValue result;
    result.value = FeTupleValue{std::move(values)};
    return result;
}

FeValue FeValue::list_value(std::vector<FeValue> values) {
    FeValue result;
    result.value = FeListValue{std::move(values)};
    return result;
}

FeExprPtr FeExpr::constant(FeValue value, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeConstantExpr{std::move(value)};
    return expr;
}

FeExprPtr FeExpr::var(std::string symbol, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeVarExpr{std::move(symbol)};
    return expr;
}

FeExprPtr FeExpr::call(std::string callee, std::vector<FeCallArg> args, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeCallExpr{std::move(callee), std::move(args)};
    return expr;
}

FeExprPtr FeExpr::layer_ctor(std::string callee, std::vector<FeCallArg> args, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeLayerCtorExpr{std::move(callee), std::move(args)};
    return expr;
}

FeExprPtr FeExpr::apply(FeExprPtr callee, std::vector<FeCallArg> args, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeApplyExpr{std::move(callee), std::move(args)};
    return expr;
}

FeExprPtr FeExpr::tuple(std::vector<FeExprPtr> elements, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeTupleExpr{std::move(elements)};
    return expr;
}

FeExprPtr FeExpr::list(std::vector<FeExprPtr> elements, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeListExpr{std::move(elements)};
    return expr;
}

FeExprPtr FeExpr::binary(FeBinaryOp op, FeExprPtr lhs, FeExprPtr rhs, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeBinaryExpr{op, std::move(lhs), std::move(rhs)};
    return expr;
}

FeExprPtr FeExpr::if_then_else(FeExprPtr condition, FeExprPtr then_expr, FeExprPtr else_expr, FeType type) {
    auto expr = std::make_shared<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeIfThenElseExpr{std::move(condition), std::move(then_expr), std::move(else_expr)};
    return expr;
}

FeType lower_type(const Type& type) {
    // here do we need the None 
    switch (type.base) {
        case TypeBase::Unknown:
            return FeType::unknown();
        case TypeBase::Int:
            if (type.scalar_dtype == "int16") {
                return FeType::int16();
            }
            if (type.scalar_dtype == "int32") {
                return FeType::int32();
            }
            if (type.scalar_dtype == "int64") {
                return FeType::int64();
            }
            return FeType::int_type();
        case TypeBase::Float:
            if (type.scalar_dtype == "float16") {
                return FeType::float16();
            }
            if (type.scalar_dtype == "float32") {
                return FeType::float32();
            }
            if (type.scalar_dtype == "float64") {
                return FeType::float64();
            }
            return FeType::float_type();
        case TypeBase::Bool:
            return FeType::bool_type();
        case TypeBase::Str:
            return FeType::str_type();
        case TypeBase::Tensor:
            return FeType::tensor(type.tensor_dtype, type.tensor_shape_expr, type.tensor_rank);
        case TypeBase::Tuple: {
            std::vector<FeType> elements;
            for (const auto& element : type.elements) {
                elements.push_back(lower_type(element));
            }
            return FeType::tuple(std::move(elements));
        }
        case TypeBase::List: {
            std::vector<FeType> elements;
            for (const auto& element : type.elements) {
                elements.push_back(lower_type(element));
            }
            return FeType::list(std::move(elements));
        }
        case TypeBase::Callable:
            return FeType::callable(type.callable_return ? lower_type(*type.callable_return)
                                                         : FeType::void_type());
        case TypeBase::None:
            return FeType::none();
    }
    return FeType::unknown();
}

std::variant<FeBinaryOp, Diagnostic> lower_binary_op(TokenType token, const SourceSpan& span) {
    switch (token) {
        case TokenType::Plus:
            return FeBinaryOp::Add;
        case TokenType::Minus:
            return FeBinaryOp::Sub;
        case TokenType::Star:
            return FeBinaryOp::Mul;
        case TokenType::Slash:
            return FeBinaryOp::Div;
        case TokenType::DoubleSlash:
            return FeBinaryOp::FloorDiv;
        case TokenType::EqEq:
            return FeBinaryOp::Eq;
        case TokenType::Neq:
            return FeBinaryOp::NotEq;
        case TokenType::Lt:
            return FeBinaryOp::Lt;
        case TokenType::Gt:
            return FeBinaryOp::Gt;
        case TokenType::LtEq:
            return FeBinaryOp::LtEq;
        case TokenType::GtEq:
            return FeBinaryOp::GtEq;
        case TokenType::AmpAmp:
            return FeBinaryOp::And;
        case TokenType::PipePipe:
            return FeBinaryOp::Or;
        case TokenType::Bang:
            return FeBinaryOp::Not;
        default:
            return Diagnostic::error("frontend_ir", "F0001", "unsupported binary operator").with_source_span(span);
    }
}

FrontendLowerer::FrontendLowerer(const Program& program, const SemanticInfo& semantic_info)
    : program_(program), semantic_info_(semantic_info) {
    for (const auto& config : program_.configs) {
        config_defs_[config.name] = &config;
        for (const auto& field : config.fields) {
            config_field_cache_[config.name][field.name] = EvaluatedConfigField{};
        }
    }
    for (const auto& symbol : semantic_info_.symbols) {
        if (!symbol.owner && (symbol.kind == SemanticSymbolKind::BuiltinFunction ||
                              symbol.kind == SemanticSymbolKind::Function ||
                              symbol.kind == SemanticSymbolKind::Layer)) {
            global_symbols_[symbol.name] = lower_type(symbol.type);
        }
    }
}

std::optional<Diagnostic> FrontendLowerer::take_last_diagnostic() {
    auto diagnostic = last_diagnostic_;
    last_diagnostic_.reset();
    return diagnostic;
}

FrontendResult FrontendLowerer::lower() {
    LoweredModule module;
    for (const auto& config : program_.configs) {
        auto lowered_config = lower_config(config);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_config)) {
            return *diagnostic;
        }
        module.configs.push_back(std::get<FeConfig>(std::move(lowered_config)));
        if (is_train_config(config)) {
            auto lowered_train = lower_train_config(config);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_train)) {
                return *diagnostic;
            }
            module.trains.push_back(std::get<FeTrain>(std::move(lowered_train)));
        }
    }

    current_symbols_.clear();
    for (const auto& stmt : program_.globals) {
        auto lowered = lower_stmt(stmt);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        module.globals.push_back(std::get<FeStmt>(std::move(lowered)));
    }
    for (const auto& layer : program_.layers) {
        auto lowered = lower_layer(layer);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        module.functions.push_back(std::get<FeFunction>(std::move(lowered)));
    }
    for (const auto& function : program_.functions) {
        auto lowered = lower_function(function);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        module.functions.push_back(std::get<FeFunction>(std::move(lowered)));
    }

    const bool has_model_layer = std::any_of(module.functions.begin(), module.functions.end(), [](const FeFunction& function) {
        return function.is_layer && function.name == "model";
    });
    const bool has_model_train = std::any_of(module.trains.begin(), module.trains.end(), [](const FeTrain& train) {
        return train.name == "model";
    });
    if (has_model_layer && has_model_train) {
        auto plan = build_execution_plan(module);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&plan)) {
            return *diagnostic;
        }
        module.execution_plan = std::get<FeExecutionPlan>(std::move(plan));
    }
    return module;
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lower_expr(const Expr& expr) {
    return std::visit(
        [&](const auto& value) -> std::variant<FeExprPtr, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IntLiteral>) {
                return FeExpr::constant(FeValue::int_value(value.value), FeType::int_type());
            } else if constexpr (std::is_same_v<T, FloatLiteral>) {
                return FeExpr::constant(FeValue::float_value(value.value), FeType::float_type());
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                return FeExpr::constant(FeValue::bool_value(value.value), FeType::bool_type());
            } else if constexpr (std::is_same_v<T, StringLiteral>) {
                return FeExpr::constant(FeValue::string_value(value.value), FeType::void_type());
            } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
                if (value.name == "None") {
                    return FeExpr::constant(FeValue::none(), FeType::none());
                }
                auto identifier = semantic_identifier_for_expr(expr, value.name);
                if (!identifier) {
                    return error_span(expr.span, "Frontend lowering missing semantic identifier info for '" + value.name + "'");
                }
                if (identifier->target == SemanticSymbolKind::Config) {
                    return error_span(expr.span, "Config object '" + value.name + "' cannot appear directly in lowered IR");
                }
                return FeExpr::var(value.name, lower_type(identifier->type));
            } else if constexpr (std::is_same_v<T, CallExpr>) {
                auto call = semantic_call_for_expr(expr, value.callee);
                if (!call) {
                    return error_span(expr.span, "Frontend lowering missing semantic call info for '" + value.callee + "'");
                }
                std::vector<FeCallArg> args;
                for (const auto& arg : value.args) {
                    auto lowered = lower_expr(*arg.value);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
                }
                FeType result_type = lower_type(call->result_type);
                if (call->target == SemanticCallTargetKind::CallableLocal) {
                    auto symbol = find_symbol(value.callee);
                    if (!symbol) {
                        return error_span(expr.span, "Frontend lowering could not resolve callable '" + value.callee + "'");
                    }
                    return FeExpr::apply(FeExpr::var(value.callee, *symbol), std::move(args), result_type);
                }
                if (is_callable_library_op(value.callee) && result_type.kind == FeTypeKind::Callable) {
                    return FeExpr::layer_ctor(value.callee, std::move(args), result_type);
                }
                return FeExpr::call(value.callee, std::move(args), result_type);
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                return error_span(expr.span, "Repeat suffix '[n]' is only valid inside arrow pipeline stages");
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                if (value.op == TokenType::Dot) {
                    auto access = semantic_config_field_access_for_expr(expr);
                    if (!access) {
                        return error_span(expr.span, "Frontend lowering missing semantic config-field access info");
                    }
                    auto constant = eval_config_field(access->config_name, access->field_name, expr.span);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&constant)) {
                        return *diagnostic;
                    }
                    return FeExpr::constant(std::get<FeValue>(std::move(constant)), lower_type(access->field_type));
                }
                auto op = lower_binary_op(value.op, expr.span);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
                    return *diagnostic;
                }
                auto lhs = lower_expr(*value.lhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
                    return *diagnostic;
                }
                auto rhs = lower_expr(*value.rhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
                    return *diagnostic;
                }
                auto type = required_semantic_type_for_expr(expr, "binary expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::binary(std::get<FeBinaryOp>(op), std::get<FeExprPtr>(std::move(lhs)), std::get<FeExprPtr>(std::move(rhs)), std::get<FeType>(std::move(type)));
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                auto operand = lower_expr(*value.operand);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&operand)) {
                    return *diagnostic;
                }
                auto type = required_semantic_type_for_expr(expr, "unary expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                if (value.op == TokenType::Minus) {
                    return FeExpr::binary(
                        FeBinaryOp::Sub,
                        FeExpr::constant(FeValue::int_value(0), FeType::int_type()),
                        std::get<FeExprPtr>(std::move(operand)),
                        std::get<FeType>(std::move(type))
                    );
                }
                auto op = lower_binary_op(value.op, expr.span);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
                    return *diagnostic;
                }
                return FeExpr::binary(
                    std::get<FeBinaryOp>(op),
                    std::get<FeExprPtr>(std::move(operand)),
                    FeExpr::constant(FeValue::bool_value(false), FeType::bool_type()),
                    std::get<FeType>(std::move(type))
                );
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                auto condition = lower_expr(*value.condition);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                auto then_expr = lower_expr(*value.then_expr);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&then_expr)) {
                    return *diagnostic;
                }
                auto else_expr = lower_expr(*value.else_expr);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&else_expr)) {
                    return *diagnostic;
                }
                auto type = required_semantic_type_for_expr(expr, "ternary expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::if_then_else(
                    std::get<FeExprPtr>(std::move(condition)),
                    std::get<FeExprPtr>(std::move(then_expr)),
                    std::get<FeExprPtr>(std::move(else_expr)),
                    std::get<FeType>(std::move(type))
                );
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                std::vector<FeExprPtr> elements;
                for (const auto& element : value.elements) {
                    auto lowered = lower_expr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
                }
                auto type = required_semantic_type_for_expr(expr, "tuple expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::tuple(std::move(elements), std::get<FeType>(std::move(type)));
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                std::vector<FeExprPtr> elements;
                for (const auto& element : value.elements) {
                    auto lowered = lower_expr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
                }
                auto type = required_semantic_type_for_expr(expr, "list expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::list(std::move(elements), std::get<FeType>(std::move(type)));
            } else if constexpr (std::is_same_v<T, ArrowExpr>) {
                return lower_arrow_expr(expr);
            }
        },
        expr.kind
    );
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lower_arrow_expr(const Expr& expr) {
    const auto* arrow = std::get_if<ArrowExpr>(&expr.kind);
    if (arrow == nullptr) {
        return lower_expr(expr);
    }
    auto current = lower_expr(*arrow->source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&current)) {
        return *diagnostic;
    }
    FeExprPtr value = std::get<FeExprPtr>(std::move(current));
    for (const auto& stage : arrow->stages) {
        auto lowered = lower_arrow_stage_expr(*stage, value);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        value = std::get<FeExprPtr>(std::move(lowered));
    }
    return value;
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lower_arrow_call_stage(
    const std::string& callee,
    const std::vector<CallArgument>& args,
    const SourceSpan& span,
    FeExprPtr current
) {
    auto call = semantic_call_for_arrow_stage(callee, span);
    if (!call) {
        return error_span(span, "Frontend lowering missing semantic call info for arrow stage '" + callee + "'");
    }
    return lower_semantic_arrow_call_stage(*call, callee, args, std::move(current));
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lower_semantic_arrow_call_stage(
    const SemanticCallInfo& call,
    const std::string& callee,
    const std::vector<CallArgument>& args,
    FeExprPtr current
) {
    FeType result_type = lower_type(call.result_type);
    if (call.target == SemanticCallTargetKind::CallableLocal) {
        auto symbol = find_symbol(callee);
        if (!symbol) {
            return error_span(call.span, "Frontend lowering could not resolve callable '" + callee + "'");
        }
        std::vector<FeCallArg> apply_args{{std::nullopt, std::move(current)}};
        for (const auto& arg : args) {
            auto lowered = lower_expr(*arg.value);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            apply_args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
        }
        return FeExpr::apply(FeExpr::var(callee, *symbol), std::move(apply_args), result_type);
    }

    if (call.target == SemanticCallTargetKind::BuiltinFunction || call.target == SemanticCallTargetKind::Function) {
        auto global = global_symbols_.find(callee);
        if (global != global_symbols_.end() && global->second.kind == FeTypeKind::Callable) {
            std::vector<FeCallArg> ctor_args;
            for (const auto& arg : args) {
                auto lowered = lower_expr(*arg.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                ctor_args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
            }
            FeExprPtr callee_expr = is_callable_library_op(callee)
                ? FeExpr::layer_ctor(callee, std::move(ctor_args), global->second)
                : FeExpr::call(callee, std::move(ctor_args), global->second);
            return FeExpr::apply(callee_expr, std::vector<FeCallArg>{{std::nullopt, std::move(current)}}, result_type);
        }
    }

    std::vector<FeCallArg> lowered_args{{std::nullopt, std::move(current)}};
    for (const auto& arg : args) {
        auto lowered = lower_expr(*arg.value);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        lowered_args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
    }
    return FeExpr::call(callee, std::move(lowered_args), result_type);
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lower_arrow_stage_expr(const Expr& expr, FeExprPtr current) {
    if (const auto* call = std::get_if<CallExpr>(&expr.kind)) {
        return lower_arrow_call_stage(call->callee, call->args, expr.span, std::move(current));
    }
    if (const auto* repeat = std::get_if<RepeatExpr>(&expr.kind)) {
        const auto* call = std::get_if<CallExpr>(&repeat->stage->kind);
        if (call == nullptr) {
            return error_span(repeat->stage->span, "Repeated arrow stage must begin with a call");
        }
        auto count = eval_constant_expr(*repeat->count);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&count)) {
            return *diagnostic;
        }
        auto repeat_value = as_int(std::get<FeValue>(count), repeat->count->span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&repeat_value)) {
            return *diagnostic;
        }
        FeExprPtr value = std::move(current);
        for (std::int64_t index = 0; index < std::get<std::int64_t>(repeat_value); ++index) {
            auto lowered = lower_arrow_call_stage(call->callee, call->args, repeat->stage->span, value);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            value = std::get<FeExprPtr>(std::move(lowered));
        }
        return value;
    }
    if (const auto* binary = std::get_if<BinaryExpr>(&expr.kind)) {
        auto lhs = count_arrow_stage_sites(*binary->lhs) > 0 ? lower_arrow_stage_expr(*binary->lhs, current)
                                                             : lower_expr(*binary->lhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
            return *diagnostic;
        }
        auto rhs = count_arrow_stage_sites(*binary->rhs) > 0 ? lower_arrow_stage_expr(*binary->rhs, current)
                                                             : lower_expr(*binary->rhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
            return *diagnostic;
        }
        auto op = lower_binary_op(binary->op, expr.span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
            return *diagnostic;
        }
        auto type = required_semantic_type_for_expr(expr, "arrow binary expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::binary(
            std::get<FeBinaryOp>(op),
            std::get<FeExprPtr>(std::move(lhs)),
            std::get<FeExprPtr>(std::move(rhs)),
            std::get<FeType>(std::move(type))
        );
    }
    if (const auto* unary = std::get_if<UnaryExpr>(&expr.kind)) {
        auto operand = count_arrow_stage_sites(*unary->operand) > 0 ? lower_arrow_stage_expr(*unary->operand, current)
                                                                    : lower_expr(*unary->operand);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&operand)) {
            return *diagnostic;
        }
        auto type = required_semantic_type_for_expr(expr, "arrow unary expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        if (unary->op == TokenType::Minus) {
            return FeExpr::binary(
                FeBinaryOp::Sub,
                FeExpr::constant(FeValue::int_value(0), FeType::int_type()),
                std::get<FeExprPtr>(std::move(operand)),
                std::get<FeType>(std::move(type))
            );
        }
        auto op = lower_binary_op(unary->op, expr.span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
            return *diagnostic;
        }
        return FeExpr::binary(
            std::get<FeBinaryOp>(op),
            std::get<FeExprPtr>(std::move(operand)),
            FeExpr::constant(FeValue::bool_value(false), FeType::bool_type()),
            std::get<FeType>(std::move(type))
        );
    }
    if (const auto* ternary = std::get_if<TernaryExpr>(&expr.kind)) {
        auto condition = count_arrow_stage_sites(*ternary->condition) > 0
            ? lower_arrow_stage_expr(*ternary->condition, current)
            : lower_expr(*ternary->condition);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
            return *diagnostic;
        }
        auto then_expr = count_arrow_stage_sites(*ternary->then_expr) > 0
            ? lower_arrow_stage_expr(*ternary->then_expr, current)
            : lower_expr(*ternary->then_expr);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&then_expr)) {
            return *diagnostic;
        }
        auto else_expr = count_arrow_stage_sites(*ternary->else_expr) > 0
            ? lower_arrow_stage_expr(*ternary->else_expr, current)
            : lower_expr(*ternary->else_expr);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&else_expr)) {
            return *diagnostic;
        }
        auto type = required_semantic_type_for_expr(expr, "arrow ternary expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::if_then_else(
            std::get<FeExprPtr>(std::move(condition)),
            std::get<FeExprPtr>(std::move(then_expr)),
            std::get<FeExprPtr>(std::move(else_expr)),
            std::get<FeType>(std::move(type))
        );
    }
    if (const auto* tuple = std::get_if<TupleExpr>(&expr.kind)) {
        std::vector<FeExprPtr> elements;
        for (const auto& element : tuple->elements) {
            auto lowered = count_arrow_stage_sites(*element) > 0 ? lower_arrow_stage_expr(*element, current)
                                                                 : lower_expr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
        }
        auto type = required_semantic_type_for_expr(expr, "arrow tuple expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::tuple(std::move(elements), std::get<FeType>(std::move(type)));
    }
    if (const auto* list = std::get_if<ListExpr>(&expr.kind)) {
        std::vector<FeExprPtr> elements;
        for (const auto& element : list->elements) {
            auto lowered = count_arrow_stage_sites(*element) > 0 ? lower_arrow_stage_expr(*element, current)
                                                                 : lower_expr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
        }
        auto type = required_semantic_type_for_expr(expr, "arrow list expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::list(std::move(elements), std::get<FeType>(std::move(type)));
    }
    if (count_arrow_stage_sites(expr) == 0) {
        return lower_expr(expr);
    }
    return error_span(expr.span, "Unsupported compound arrow stage in frontend lowering");
}

std::variant<std::vector<FeStmt>, Diagnostic> FrontendLowerer::lower_scope(const Stmt& stmt) {
    const auto* scope = std::get_if<ScopeStmt>(&stmt.kind);
    if (scope == nullptr) {
        return error_span(stmt.span, "Expected scope statement while lowering");
    }
    auto saved = current_symbols_;
    std::vector<FeStmt> lowered;
    for (const auto& child : scope->statements) {
        auto stmt_result = lower_stmt(child);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&stmt_result)) {
            return *diagnostic;
        }
        lowered.push_back(std::get<FeStmt>(std::move(stmt_result)));
    }
    current_symbols_ = saved;
    return lowered;
}

std::variant<FeStmt, Diagnostic> FrontendLowerer::lower_stmt(const Stmt& stmt) {
    return std::visit(
        [&](const auto& value) -> std::variant<FeStmt, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, ReturnStmt>) {
                auto lowered = lower_expr(*value.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                return FeStmt{FeReturnStmt{std::get<FeExprPtr>(std::move(lowered))}};
            } else if constexpr (std::is_same_v<T, ExprStmt>) {
                auto lowered = lower_expr(*value.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                return FeStmt{FeExprStmt{std::get<FeExprPtr>(std::move(lowered))}};
            } else if constexpr (std::is_same_v<T, VarDecl>) {
                auto declaration = semantic_declaration_for_stmt(stmt, value.name);
                if (!declaration) {
                    return error_span(stmt.span, "Frontend lowering missing semantic declaration info for '" + value.name + "'");
                }
                FeExprPtr lowered_value;
                bool has_value = false;
                if (value.init) {
                    auto lowered = lower_expr(*value.init);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    lowered_value = std::get<FeExprPtr>(std::move(lowered));
                    has_value = true;
                }
                FeType type = lower_type(declaration->final_type);
                bind_symbol(value.name, type);
                return FeStmt{FeVarDeclStmt{value.name, type, lowered_value, has_value, value.is_mutable}};
            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                auto assignment = semantic_assignment_for_stmt(stmt, value.name);
                if (!assignment) {
                    return error_span(stmt.span, "Frontend lowering missing semantic assignment info for '" + value.name + "'");
                }
                auto lowered = lower_expr(*value.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                bind_symbol(value.name, lower_type(assignment->target_type));
                return FeStmt{FeAssignStmt{value.name, std::get<FeExprPtr>(std::move(lowered))}};
            } else if constexpr (std::is_same_v<T, ScopeStmt>) {
                return error_span(stmt.span, "Nested standalone scope statements are not supported in FE lowering");
            } else if constexpr (std::is_same_v<T, IfStmt>) {
                auto condition = lower_expr(*value.condition);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                auto then_body = lower_scope(*value.then_stmt);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&then_body)) {
                    return *diagnostic;
                }
                std::vector<FeElifBody> elifs;
                for (const auto& branch : value.elifs) {
                    auto branch_condition = lower_expr(*branch.condition);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&branch_condition)) {
                        return *diagnostic;
                    }
                    auto body = lower_scope(*branch.body);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&body)) {
                        return *diagnostic;
                    }
                    elifs.push_back(FeElifBody{
                        std::get<FeExprPtr>(std::move(branch_condition)),
                        std::get<std::vector<FeStmt>>(std::move(body)),
                    });
                }
                std::vector<FeStmt> else_body;
                if (value.else_stmt) {
                    auto lowered_else = lower_scope(*value.else_stmt);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_else)) {
                        return *diagnostic;
                    }
                    else_body = std::get<std::vector<FeStmt>>(std::move(lowered_else));
                }
                return FeStmt{FeIfStmt{
                    std::get<FeExprPtr>(std::move(condition)),
                    std::get<std::vector<FeStmt>>(std::move(then_body)),
                    std::move(elifs),
                    std::move(else_body),
                }};
            }
        },
        stmt.kind
    );
}

std::variant<FeFunction, Diagnostic> FrontendLowerer::lower_function(const Function& function) {
    FeFunction lowered;
    lowered.name = function.name;
    lowered.is_layer = false;
    lowered.return_type = lower_type(function.return_type);
    auto saved_symbols = current_symbols_;
    auto saved_owner = current_owner_;
    current_symbols_.clear();
    current_owner_ = function.name;
    for (const auto& arg : function.args) {
        FeType type = lower_type(arg.type);
        lowered.params.push_back({arg.name, type});
        bind_symbol(arg.name, type);
    }
    auto body = lower_scope(function.body);
    current_symbols_ = saved_symbols;
    current_owner_ = saved_owner;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&body)) {
        return *diagnostic;
    }
    lowered.body = std::get<std::vector<FeStmt>>(std::move(body));
    return lowered;
}

std::variant<FeFunction, Diagnostic> FrontendLowerer::lower_layer(const Layer& layer) {
    FeFunction lowered;
    lowered.name = layer.name;
    lowered.is_layer = true;
    lowered.return_type = lower_type(layer.return_type);
    auto saved_symbols = current_symbols_;
    auto saved_owner = current_owner_;
    current_symbols_.clear();
    current_owner_ = layer.name;
    for (const auto& arg : layer.args) {
        FeType type = lower_type(arg.type);
        lowered.params.push_back({arg.name, type});
        bind_symbol(arg.name, type);
    }
    auto body = lower_scope(layer.body);
    current_symbols_ = saved_symbols;
    current_owner_ = saved_owner;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&body)) {
        return *diagnostic;
    }
    lowered.body = std::get<std::vector<FeStmt>>(std::move(body));
    for (const auto& stmt : lowered.body) {
        if (const auto* ret = std::get_if<FeReturnStmt>(&stmt.kind)) {
            if (const auto* var = std::get_if<FeVarExpr>(&ret->value->kind)) {
                lowered.named_outputs.push_back({var->symbol, ret->value->type});
            } else if (const auto* tuple = std::get_if<FeTupleExpr>(&ret->value->kind)) {
                for (const auto& element : tuple->elements) {
                    if (const auto* var = std::get_if<FeVarExpr>(&element->kind)) {
                        lowered.named_outputs.push_back({var->symbol, element->type});
                    }
                }
            }
        }
    }
    return lowered;
}

std::variant<FeConfig, Diagnostic> FrontendLowerer::lower_config(const Config& config) {
    FeConfig lowered;
    lowered.name = config.name;
    for (const auto& field : config.fields) {
        auto value = eval_config_field(config.name, field.name, config.span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
            return *diagnostic;
        }
        lowered.fields[field.name] = std::get<FeValue>(std::move(value));
    }
    return lowered;
}

std::variant<FeTrain, Diagnostic> FrontendLowerer::lower_train_config(const Config& config) {
    FeTrain lowered;
    lowered.name = config.name;
    lowered.variant_count = 1;
    for (const auto& field : config.fields) {
        std::vector<FeValue> values;
        if (field.init) {
            auto evaluated = eval_constant_field_values(*field.init);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                return *diagnostic;
            }
            values = std::get<std::vector<FeValue>>(std::move(evaluated));
        } else {
            values = {FeValue::none()};
        }
        bool assigned = true;
        if (field.name == "backend" || field.name == "target" || field.name == "device") {
            lowered.backends = values;
        } else if (field.name == "optimizer") {
            lowered.optimizers = values;
        } else if (field.name == "lr" || field.name == "learning_rate") {
            lowered.learning_rates = values;
        } else if (field.name == "iteration") {
            lowered.iterations = values;
        } else if (field.name == "objective") {
            lowered.objective_symbols.clear();
            for (const auto& value : values) {
                const auto* string_value = std::get_if<std::string>(&value.value);
                if (string_value == nullptr) {
                    return error_span(config.span, "Training config field 'objective' must reference a named tensor root");
                }
                lowered.objective_symbols.push_back(*string_value);
            }
        } else {
            assigned = false;
        }
        if (assigned) {
            lowered.variant_count = std::max(lowered.variant_count, values.size());
        } else {
            if (values.size() != 1) {
                return error_span(config.span, "Training config field '" + field.name + "' does not support tuple variants; use a scalar value");
            }
            lowered.extra_properties[field.name] = values.front();
        }
    }
    return lowered;
}

std::variant<FeExecutionPlan, Diagnostic> FrontendLowerer::build_execution_plan(const LoweredModule& module) {
    auto model = std::find_if(module.functions.begin(), module.functions.end(), [](const FeFunction& function) {
        return function.is_layer && function.name == "model";
    });
    auto train = std::find_if(module.trains.begin(), module.trains.end(), [](const FeTrain& item) {
        return item.name == "model";
    });
    if (model == module.functions.end() || train == module.trains.end()) {
        return error("Training config lowering requires layer model and config model");
    }
    FeExecutionPlan plan;
    plan.model_entry = model->name;
    for (std::size_t index = 0; index < train->variant_count; ++index) {
        FeExecutionRun run;
        run.run_name = train->variant_count == 1 ? "model" : "model_" + std::to_string(index + 1);
        run.model_name = model->name;
        run.train_name = train->name;
        run.backend = pick_broadcast_value(train->backends, index);
        run.optimizer = pick_broadcast_value(train->optimizers, index);
        run.learning_rate = pick_broadcast_value(train->learning_rates, index);
        run.objective_symbol = pick_broadcast_string(train->objective_symbols, index);
        run.iteration = pick_broadcast_value(train->iterations, index);
        if (run.objective_symbol) {
            for (const auto& output : model->named_outputs) {
                if (output.first == *run.objective_symbol) {
                    run.objective_source = ObjectiveSource::Output;
                    run.objective_type = output.second;
                }
            }
            if (run.objective_source == ObjectiveSource::Unknown) {
                for (const auto& param : model->params) {
                    if (param.first == *run.objective_symbol) {
                        run.objective_source = ObjectiveSource::Param;
                        run.objective_type = param.second;
                    }
                }
            }
            if (run.objective_source == ObjectiveSource::Unknown) {
                for (const auto& stmt : model->body) {
                    if (resolve_objective_stmt(stmt, *run.objective_symbol, run)) {
                        break;
                    }
                }
            }
        }
        plan.runs.push_back(std::move(run));
    }
    return plan;
}

bool FrontendLowerer::resolve_objective_stmt(
    const FeStmt& stmt,
    const std::string& objective_symbol,
    FeExecutionRun& run
) const {
    return std::visit(
        [&](const auto& value) -> bool {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, FeVarDeclStmt>) {
                if (value.name != objective_symbol) {
                    return false;
                }
                run.objective_source = ObjectiveSource::Local;
                run.objective_type = value.type;
                return true;
            } else if constexpr (std::is_same_v<T, FeAssignStmt>) {
                if (value.name != objective_symbol) {
                    return false;
                }
                run.objective_source = ObjectiveSource::Local;
                run.objective_type = value.value->type;
                return true;
            } else if constexpr (std::is_same_v<T, FeIfStmt>) {
                for (const auto& child : value.then_body) {
                    if (resolve_objective_stmt(child, objective_symbol, run)) {
                        return true;
                    }
                }
                for (const auto& elif : value.elif_bodies) {
                    for (const auto& child : elif.body) {
                        if (resolve_objective_stmt(child, objective_symbol, run)) {
                            return true;
                        }
                    }
                }
                for (const auto& child : value.else_body) {
                    if (resolve_objective_stmt(child, objective_symbol, run)) {
                        return true;
                    }
                }
                return false;
            } else {
                return false;
            }
        },
        stmt.kind
    );
}

std::variant<FeValue, Diagnostic> FrontendLowerer::eval_constant_expr(const Expr& expr) {
    return std::visit(
        [&](const auto& value) -> std::variant<FeValue, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IntLiteral>) {
                return FeValue::int_value(value.value);
            } else if constexpr (std::is_same_v<T, FloatLiteral>) {
                return FeValue::float_value(value.value);
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                return FeValue::bool_value(value.value);
            } else if constexpr (std::is_same_v<T, StringLiteral>) {
                return FeValue::string_value(value.value);
            } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
                if (value.name == "None") {
                    return FeValue::none();
                }
                return FeValue::string_value(value.name);
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                std::vector<FeValue> values;
                for (const auto& element : value.elements) {
                    auto evaluated = eval_constant_expr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                        return *diagnostic;
                    }
                    values.push_back(std::get<FeValue>(std::move(evaluated)));
                }
                return FeValue::tuple_value(std::move(values));
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                std::vector<FeValue> values;
                for (const auto& element : value.elements) {
                    auto evaluated = eval_constant_expr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                        return *diagnostic;
                    }
                    values.push_back(std::get<FeValue>(std::move(evaluated)));
                }
                return FeValue::list_value(std::move(values));
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                auto operand = eval_constant_expr(*value.operand);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&operand)) {
                    return *diagnostic;
                }
                return eval_unary(value.op, std::get<FeValue>(std::move(operand)), expr.span);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                if (value.op == TokenType::Dot) {
                    auto access = semantic_config_field_access_for_expr(expr);
                    if (!access) {
                        return error_span(expr.span, "Frontend lowering missing semantic config-field access info");
                    }
                    return eval_config_field(access->config_name, access->field_name, expr.span);
                }
                auto lhs = eval_constant_expr(*value.lhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
                    return *diagnostic;
                }
                auto rhs = eval_constant_expr(*value.rhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
                    return *diagnostic;
                }
                return eval_binary(value.op, std::get<FeValue>(std::move(lhs)), std::get<FeValue>(std::move(rhs)), expr.span);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                auto condition = eval_constant_expr(*value.condition);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                auto bool_condition = as_bool(std::get<FeValue>(condition), value.condition->span);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&bool_condition)) {
                    return *diagnostic;
                }
                return std::get<bool>(bool_condition) ? eval_constant_expr(*value.then_expr)
                                                      : eval_constant_expr(*value.else_expr);
            } else {
                return error_span(expr.span, "Expression is not compile-time constant");
            }
        },
        expr.kind
    );
}

std::variant<std::vector<FeValue>, Diagnostic> FrontendLowerer::eval_constant_field_values(const Expr& expr) {
    if (const auto* tuple = std::get_if<TupleExpr>(&expr.kind)) {
        std::vector<FeValue> values;
        for (const auto& element : tuple->elements) {
            auto evaluated = eval_constant_expr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                return *diagnostic;
            }
            values.push_back(std::get<FeValue>(std::move(evaluated)));
        }
        return values;
    }
    if (const auto* list = std::get_if<ListExpr>(&expr.kind)) {
        std::vector<FeValue> values;
        for (const auto& element : list->elements) {
            auto evaluated = eval_constant_expr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                return *diagnostic;
            }
            values.push_back(std::get<FeValue>(std::move(evaluated)));
        }
        return values;
    }
    auto value = eval_constant_expr(expr);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
        return *diagnostic;
    }
    return std::vector<FeValue>{std::get<FeValue>(std::move(value))};
}

std::variant<FeValue, Diagnostic> FrontendLowerer::eval_config_field(
    const std::string& config_name,
    const std::string& field_name,
    const SourceSpan& span
) {
    auto config = config_defs_.find(config_name);
    if (config == config_defs_.end()) {
        return error_span(span, "Unknown config '" + config_name + "'");
    }
    auto& cache = config_field_cache_[config_name][field_name];
    if (cache.computed) {
        return cache.value;
    }
    if (cache.in_progress) {
        return error_span(span, "Cycle detected while evaluating config field '" + config_name + "." + field_name + "'");
    }
    const Field* field = nullptr;
    for (const auto& candidate : config->second->fields) {
        if (candidate.name == field_name) {
            field = &candidate;
            break;
        }
    }
    if (field == nullptr) {
        return error_span(span, "Config field '" + config_name + "." + field_name + "' does not exist");
    }
    if (!field->init) {
        return error_span(span, "Config field '" + config_name + "." + field_name + "' does not have a compile-time value");
    }
    cache.in_progress = true;
    auto value = eval_constant_expr(*field->init);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
        cache.in_progress = false;
        return *diagnostic;
    }
    cache.value = std::get<FeValue>(std::move(value));
    cache.computed = true;
    cache.in_progress = false;
    return cache.value;
}

std::variant<FeValue, Diagnostic> FrontendLowerer::eval_binary(TokenType op, const FeValue& lhs, const FeValue& rhs, const SourceSpan& span) {
    auto lhs_num = [&]() -> std::variant<double, Diagnostic> { return as_double(lhs, span); };
    auto rhs_num = [&]() -> std::variant<double, Diagnostic> { return as_double(rhs, span); };
    const bool both_int = std::holds_alternative<std::int64_t>(lhs.value) && std::holds_alternative<std::int64_t>(rhs.value);
    if (op == TokenType::Plus || op == TokenType::Minus || op == TokenType::Star || op == TokenType::DoubleSlash) {
        if (both_int) {
            const auto left = std::get<std::int64_t>(lhs.value);
            const auto right = std::get<std::int64_t>(rhs.value);
            if (op == TokenType::Plus) {
                return FeValue::int_value(left + right);
            }
            if (op == TokenType::Minus) {
                return FeValue::int_value(left - right);
            }
            if (op == TokenType::Star) {
                return FeValue::int_value(left * right);
            }
            return FeValue::int_value(static_cast<std::int64_t>(std::floor(static_cast<double>(left) / static_cast<double>(right))));
        }
    }
    if (op == TokenType::Plus || op == TokenType::Minus || op == TokenType::Star ||
        op == TokenType::Slash || op == TokenType::DoubleSlash || op == TokenType::Lt ||
        op == TokenType::Gt || op == TokenType::LtEq || op == TokenType::GtEq) {
        auto left = lhs_num();
        if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
            return *diagnostic;
        }
        auto right = rhs_num();
        if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
            return *diagnostic;
        }
        const double l = std::get<double>(left);
        const double r = std::get<double>(right);
        switch (op) {
            case TokenType::Plus:
                return FeValue::float_value(l + r);
            case TokenType::Minus:
                return FeValue::float_value(l - r);
            case TokenType::Star:
                return FeValue::float_value(l * r);
            case TokenType::Slash:
                return FeValue::float_value(l / r);
            case TokenType::DoubleSlash:
                return FeValue::float_value(std::floor(l / r));
            case TokenType::Lt:
                return FeValue::bool_value(l < r);
            case TokenType::Gt:
                return FeValue::bool_value(l > r);
            case TokenType::LtEq:
                return FeValue::bool_value(l <= r);
            case TokenType::GtEq:
                return FeValue::bool_value(l >= r);
            default:
                break;
        }
    }
    if (op == TokenType::EqEq) {
        if (is_none(lhs) || is_none(rhs)) {
            return FeValue::bool_value(is_none(lhs) && is_none(rhs));
        }
        if (std::holds_alternative<bool>(lhs.value) && std::holds_alternative<bool>(rhs.value)) {
            return FeValue::bool_value(std::get<bool>(lhs.value) == std::get<bool>(rhs.value));
        }
        auto left = as_double(lhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
            return *diagnostic;
        }
        auto right = as_double(rhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
            return *diagnostic;
        }
        return FeValue::bool_value(std::abs(std::get<double>(left) - std::get<double>(right)) < 1e-12);
    }
    if (op == TokenType::Neq) {
        auto equal = eval_binary(TokenType::EqEq, lhs, rhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&equal)) {
            return *diagnostic;
        }
        return FeValue::bool_value(!std::get<bool>(std::get<FeValue>(equal).value));
    }
    if (op == TokenType::AmpAmp || op == TokenType::PipePipe) {
        auto left = as_bool(lhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
            return *diagnostic;
        }
        auto right = as_bool(rhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
            return *diagnostic;
        }
        return FeValue::bool_value(op == TokenType::AmpAmp ? (std::get<bool>(left) && std::get<bool>(right))
                                                           : (std::get<bool>(left) || std::get<bool>(right)));
    }
    return error_span(span, "Constant evaluation failed: unsupported binary operator");
}

std::variant<FeValue, Diagnostic> FrontendLowerer::eval_unary(TokenType op, const FeValue& operand, const SourceSpan& span) {
    if (op == TokenType::Minus) {
        if (std::holds_alternative<std::int64_t>(operand.value)) {
            return FeValue::int_value(-std::get<std::int64_t>(operand.value));
        }
        auto value = as_double(operand, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
            return *diagnostic;
        }
        return FeValue::float_value(-std::get<double>(value));
    }
    if (op == TokenType::Bang) {
        auto value = as_bool(operand, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
            return *diagnostic;
        }
        return FeValue::bool_value(!std::get<bool>(value));
    }
    return error_span(span, "Constant evaluation failed: unsupported unary operator");
}

std::optional<SemanticCallInfo> FrontendLowerer::semantic_call_for_expr(const Expr& expr, const std::string& callee) const {
    for (auto call = semantic_info_.calls.rbegin(); call != semantic_info_.calls.rend(); ++call) {
        if (call->callee == callee && same_span(call->span, expr.span) && !call->arrow_stage &&
            owner_matches(call->owner, current_owner_)) {
            return *call;
        }
    }
    return std::nullopt;
}

std::optional<SemanticCallInfo> FrontendLowerer::semantic_call_for_arrow_stage(const std::string& callee, const SourceSpan& span) const {
    for (auto call = semantic_info_.calls.rbegin(); call != semantic_info_.calls.rend(); ++call) {
        if (call->callee == callee && same_span(call->span, span) && call->arrow_stage &&
            owner_matches(call->owner, current_owner_)) {
            return *call;
        }
    }
    return std::nullopt;
}

std::optional<SemanticIdentifierInfo> FrontendLowerer::semantic_identifier_for_expr(const Expr& expr, const std::string& name) const {
    for (auto identifier = semantic_info_.identifiers.rbegin(); identifier != semantic_info_.identifiers.rend(); ++identifier) {
        if (identifier->name == name && same_span(identifier->span, expr.span) &&
            owner_matches(identifier->owner, current_owner_)) {
            return *identifier;
        }
    }
    return std::nullopt;
}

std::optional<SemanticAssignmentInfo> FrontendLowerer::semantic_assignment_for_stmt(const Stmt& stmt, const std::string& name) const {
    for (auto assignment = semantic_info_.assignments.rbegin(); assignment != semantic_info_.assignments.rend(); ++assignment) {
        if (assignment->target_name == name && same_span(assignment->span, stmt.span) &&
            owner_matches(assignment->owner, current_owner_)) {
            return *assignment;
        }
    }
    return std::nullopt;
}

std::optional<SemanticConfigFieldAccessInfo> FrontendLowerer::semantic_config_field_access_for_expr(const Expr& expr) const {
    for (auto access = semantic_info_.config_field_accesses.rbegin(); access != semantic_info_.config_field_accesses.rend(); ++access) {
        if (same_span(access->span, expr.span) && owner_matches(access->owner, current_owner_)) {
            return *access;
        }
    }
    return std::nullopt;
}

std::optional<SemanticDeclarationInfo> FrontendLowerer::semantic_declaration_for_stmt(const Stmt& stmt, const std::string& name) const {
    for (auto declaration = semantic_info_.declarations.rbegin(); declaration != semantic_info_.declarations.rend(); ++declaration) {
        if (declaration->name == name && same_span(declaration->span, stmt.span) &&
            owner_matches(declaration->owner, current_owner_)) {
            return *declaration;
        }
    }
    return std::nullopt;
}

std::optional<FeType> FrontendLowerer::semantic_type_for_expr(const Expr& expr) const {
    for (auto info = semantic_info_.exprs.rbegin(); info != semantic_info_.exprs.rend(); ++info) {
        if (same_span(info->span, expr.span) && owner_matches(info->owner, current_owner_)) {
            return lower_type(info->type);
        }
    }
    return std::nullopt;
}

std::variant<FeType, Diagnostic> FrontendLowerer::required_semantic_type_for_expr(const Expr& expr, const std::string& context) {
    auto type = semantic_type_for_expr(expr);
    if (!type) {
        return error_span(expr.span, "Frontend lowering missing semantic expression type info for " + context);
    }
    return *type;
}

void FrontendLowerer::bind_symbol(const std::string& name, FeType type) {
    current_symbols_[name] = std::move(type);
}

std::optional<FeType> FrontendLowerer::find_symbol(const std::string& name) const {
    auto local = current_symbols_.find(name);
    if (local != current_symbols_.end()) {
        return local->second;
    }
    auto global = global_symbols_.find(name);
    if (global != global_symbols_.end()) {
        return global->second;
    }
    return std::nullopt;
}

Diagnostic FrontendLowerer::error(const std::string& message) {
    Diagnostic diagnostic = Diagnostic::error("frontend_ir", "F0001", message);
    last_diagnostic_ = diagnostic;
    return diagnostic;
}

Diagnostic FrontendLowerer::error_span(const SourceSpan& span, const std::string& message) {
    Diagnostic diagnostic = Diagnostic::error("frontend_ir", "F0001", message).with_source_span(span);
    last_diagnostic_ = diagnostic;
    return diagnostic;
}

std::string lowered_module_summary(const LoweredModule& module) {
    std::ostringstream out;
    out << "lowered=configs:" << module.configs.size()
        << " trains:" << module.trains.size()
        << " functions:" << module.functions.size()
        << " globals:" << module.globals.size()
        << " execution_plan:" << (module.execution_plan ? "yes" : "no");
    return out.str();
}

std::string frontend_ir_to_string(const LoweredModule& module) {
    std::ostringstream out;
    out << lowered_module_summary(module) << '\n';
    for (const auto& config : module.configs) {
        out << "config " << config.name << " fields=" << config.fields.size() << '\n';
    }
    for (const auto& train : module.trains) {
        out << "train " << train.name << " variants=" << train.variant_count << '\n';
    }
    for (const auto& function : module.functions) {
        out << (function.is_layer ? "layer " : "fn ") << function.name << '(';
        for (std::size_t index = 0; index < function.params.size(); ++index) {
            if (index != 0) {
                out << ", ";
            }
            out << function.params[index].first << ": " << fe_type_to_string(function.params[index].second);
        }
        out << ") -> " << fe_type_to_string(function.return_type) << '\n';
        for (const auto& stmt : function.body) {
            append_stmt(out, stmt, 2);
        }
    }
    for (const auto& stmt : module.globals) {
        append_stmt(out, stmt, 0);
    }
    return out.str();
}
