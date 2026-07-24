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
           name == "lr" || name == "learningRate" || name == "objective" || name == "iteration";
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
                return count_arrow_stage_sites(*value.thenExpr) +
                       count_arrow_stage_sites(*value.condition) +
                       count_arrow_stage_sites(*value.elseExpr);
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
    if (const auto* intValue = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<double>(*intValue);
    }
    if (const auto* floatValue = std::get_if<double>(&value.value)) {
        return *floatValue;
    }
    return Diagnostic::error(DiagnosticCode::FrontendIrError, "Expected numeric constant").withSourceSpan(span);
}

std::variant<std::int64_t, Diagnostic> as_int(const FeValue& value, const SourceSpan& span) {
    if (const auto* intValue = std::get_if<std::int64_t>(&value.value)) {
        return *intValue;
    }
    return Diagnostic::error(DiagnosticCode::FrontendIrError, "Expected integer constant").withSourceSpan(span);
}

std::variant<bool, Diagnostic> as_bool(const FeValue& value, const SourceSpan& span) {
    if (const auto* boolValue = std::get_if<bool>(&value.value)) {
        return *boolValue;
    }
    return Diagnostic::error(DiagnosticCode::FrontendIrError, "Expected boolean constant").withSourceSpan(span);
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
            return type.scalarDtype.value_or("int");
        case FeTypeKind::Float:
            return type.scalarDtype.value_or("float");
        case FeTypeKind::Bool:
            return "bool";
        case FeTypeKind::Str:
            return "str";
        case FeTypeKind::Tensor:
            if (type.tensorDtype && type.tensorShapeExpr) {
                return "tensor[" + *type.tensorDtype + ", " + *type.tensorShapeExpr + "]";
            }
            if (type.tensorDtype) {
                return "tensor[" + *type.tensorDtype + "]";
            }
            if (type.tensorShapeExpr) {
                return "tensor[" + *type.tensorShapeExpr + "]";
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
            return "callable -> " + (type.callableReturn ? fe_type_to_string(*type.callableReturn) : "void");
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
                out << "layerCtor " << value.callee << '(';
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
                append_expr(out, value.thenExpr);
                out << " else ";
                append_expr(out, value.elseExpr);
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
                out  << value.name << ": "
                    << fe_type_to_string(value.type);
                if (value.hasValue && value.value) {
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
                for (const auto& child : value.thenBody) {
                    append_stmt(out, child, indent + 2);
                }
                for (const auto& elif : value.elifBodies) {
                    out << pad << "elif ";
                    append_expr(out, elif.condition);
                    out << '\n';
                    for (const auto& child : elif.body) {
                        append_stmt(out, child, indent + 2);
                    }
                }
                if (!value.elseBody.empty()) {
                    out << pad << "else\n";
                    for (const auto& child : value.elseBody) {
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

FeType FeType::intType() {
    FeType type;
    type.kind = FeTypeKind::Int;
    return type;
}

FeType FeType::int16() {
    FeType type = intType();
    type.scalarDtype = "int16";
    return type;
}

FeType FeType::int32() {
    FeType type = intType();
    type.scalarDtype = "int32";
    return type;
}

FeType FeType::int64() {
    FeType type = intType();
    type.scalarDtype = "int64";
    return type;
}

FeType FeType::floatType() {
    FeType type;
    type.kind = FeTypeKind::Float;
    return type;
}

FeType FeType::float16() {
    FeType type = floatType();
    type.scalarDtype = "float16";
    return type;
}

FeType FeType::float32() {
    FeType type = floatType();
    type.scalarDtype = "float32";
    return type;
}

FeType FeType::float64() {
    FeType type = floatType();
    type.scalarDtype = "float64";
    return type;
}

FeType FeType::strType() {
    FeType type;
    type.kind = FeTypeKind::Str;
    return type;
}

FeType FeType::boolType() {
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
    type.tensorDtype = std::move(dtype);
    type.tensorShapeExpr = std::move(shape_expr);
    type.tensorRank = rank;
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

FeType FeType::callable(FeType returnType) {
    FeType type;
    type.kind = FeTypeKind::Callable;
    type.callableReturn = std::make_shared<FeType>(std::move(returnType));
    return type;
}

FeType FeType::voidType() {
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

FeValue FeValue::intValue(std::int64_t value) {
    FeValue result;
    result.value = value;
    return result;
}

FeValue FeValue::floatValue(double value) {
    FeValue result;
    result.value = value;
    return result;
}

FeValue FeValue::boolValue(bool value) {
    FeValue result;
    result.value = value;
    return result;
}

FeValue FeValue::stringValue(std::string value) {
    FeValue result;
    result.value = std::move(value);
    return result;
}

FeValue FeValue::tupleValue(std::vector<FeValue> values) {
    FeValue result;
    result.value = FeTupleValue{std::move(values)};
    return result;
}

FeValue FeValue::listValue(std::vector<FeValue> values) {
    FeValue result;
    result.value = FeListValue{std::move(values)};
    return result;
}

FeExprPtr FeExpr::constant(Arena& arena, FeValue value, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeConstantExpr{std::move(value)};
    return expr;
}

FeExprPtr FeExpr::var(Arena& arena, std::string symbol, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeVarExpr{std::move(symbol)};
    return expr;
}

FeExprPtr FeExpr::call(Arena& arena, std::string callee, std::vector<FeCallArg> args, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeCallExpr{std::move(callee), std::move(args)};
    return expr;
}

FeExprPtr FeExpr::layerCtor(Arena& arena, std::string callee, std::vector<FeCallArg> args, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeLayerCtorExpr{std::move(callee), std::move(args)};
    return expr;
}

FeExprPtr FeExpr::apply(Arena& arena, FeExprPtr callee, std::vector<FeCallArg> args, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeApplyExpr{std::move(callee), std::move(args)};
    return expr;
}

FeExprPtr FeExpr::tuple(Arena& arena, std::vector<FeExprPtr> elements, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeTupleExpr{std::move(elements)};
    return expr;
}

FeExprPtr FeExpr::list(Arena& arena, std::vector<FeExprPtr> elements, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeListExpr{std::move(elements)};
    return expr;
}

FeExprPtr FeExpr::binary(Arena& arena, FeBinaryOp op, FeExprPtr lhs, FeExprPtr rhs, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeBinaryExpr{op, std::move(lhs), std::move(rhs)};
    return expr;
}

FeExprPtr FeExpr::ifThenElse(Arena& arena, FeExprPtr condition, FeExprPtr thenExpr, FeExprPtr elseExpr, FeType type) {
    auto expr = arena.allocate<FeExpr>();
    expr->type = std::move(type);
    expr->kind = FeIfThenElseExpr{std::move(condition), std::move(thenExpr), std::move(elseExpr)};
    return expr;
}

FeType lowerType(const Type& type) {
    // Translates the parser AST Type into the strongly-typed FeType used in
    // Frontend IR. Handles scalar dtype mappings and nested types.
    switch (type.base) {
        case TypeBase::Unknown:
            return FeType::unknown();
        case TypeBase::Int:
            if (type.scalarDtype == "int16") {
                return FeType::int16();
            }
            if (type.scalarDtype == "int32") {
                return FeType::int32();
            }
            if (type.scalarDtype == "int64") {
                return FeType::int64();
            }
            return FeType::intType();
        case TypeBase::Float:
            if (type.scalarDtype == "float16") {
                return FeType::float16();
            }
            if (type.scalarDtype == "float32") {
                return FeType::float32();
            }
            if (type.scalarDtype == "float64") {
                return FeType::float64();
            }
            return FeType::floatType();
        case TypeBase::Bool:
            return FeType::boolType();
        case TypeBase::Str:
            return FeType::strType();
        case TypeBase::Tensor:
            return FeType::tensor(type.tensorDtype, type.tensorShapeExpr, type.tensorRank);
        case TypeBase::Tuple: {
            std::vector<FeType> elements;
            for (const auto& element : type.elements) {
                elements.push_back(lowerType(element));
            }
            return FeType::tuple(std::move(elements));
        }
        case TypeBase::List: {
            std::vector<FeType> elements;
            for (const auto& element : type.elements) {
                elements.push_back(lowerType(element));
            }
            return FeType::list(std::move(elements));
        }
        case TypeBase::Callable:
            return FeType::callable(type.callableReturn ? lowerType(*type.callableReturn)
                                                         : FeType::voidType());
        case TypeBase::None:
            return FeType::none();
    }
    return FeType::unknown();
}

std::variant<FeBinaryOp, Diagnostic> lowerBinaryOp(TokenType token, const SourceSpan& span) {
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
            return Diagnostic::error(DiagnosticCode::FrontendIrError, "unsupported binary operator").withSourceSpan(span);
    }
}

FrontendLowerer::FrontendLowerer(const Program& program, const SemanticInfo& semantic_info)
    : program_(program), semanticInfo_(semantic_info), arena_(std::make_unique<Arena>()) {
    for (const auto& config : program_.configs) {
        configDefs_[config.name] = &config;
        for (const auto& field : config.fields) {
            configFieldCache_[config.name][field.name] = EvaluatedConfigField{};
        }
    }
    for (const auto& symbol : semanticInfo_.symbols) {
        if (!symbol.owner && (symbol.kind == SemanticSymbolKind::BuiltinFunction ||
                              symbol.kind == SemanticSymbolKind::Function ||
                              symbol.kind == SemanticSymbolKind::Layer)) {
            globalSymbols_[symbol.name] = lowerType(symbol.type);
        }
    }
}

std::optional<Diagnostic> FrontendLowerer::takeLastDiagnostic() {
    auto diagnostic = lastDiagnostic_;
    lastDiagnostic_.reset();
    return diagnostic;
}

FrontendResult FrontendLowerer::lower() {
    LoweredModule module;
    for (const auto& config : program_.configs) {
        auto lowered_config = lowerConfig(config);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_config)) {
            return *diagnostic;
        }
        module.configs.push_back(std::get<FeConfig>(std::move(lowered_config)));
        if (is_train_config(config)) {
            auto lowered_train = lowerTrainConfig(config);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_train)) {
                return *diagnostic;
            }
            module.trains.push_back(std::get<FeTrain>(std::move(lowered_train)));
        }
    }

    currentSymbols_.clear();
    // Lower global statements.
    for (const auto& stmt : program_.globals) {
        auto lowered = lowerStmt(stmt);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        module.globals.push_back(std::get<FeStmt>(std::move(lowered)));
    }
    for (const auto& layer : program_.layers) {
        auto lowered = lowerLayer(layer);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        module.layers.push_back(std::get<FeLayer>(std::move(lowered)));
    }
    for (const auto& function : program_.functions) {
        auto lowered = lowerFunction(function);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        module.functions.push_back(std::get<FeFunction>(std::move(lowered)));
    }

    const bool has_model_layer = std::any_of(module.layers.begin(), module.layers.end(), [](const FeLayer& layer) {
        return layer.name == "model";
    });
    const bool has_model_train = std::any_of(module.trains.begin(), module.trains.end(), [](const FeTrain& train) {
        return train.name == "model";
    });
    if (has_model_layer && has_model_train) {
        auto plan = buildExecutionPlan(module);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&plan)) {
            return *diagnostic;
        }
        module.executionPlan = std::get<FeExecutionPlan>(std::move(plan));
    }
    return module;
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lowerExpr(const Expr& expr) {
    return std::visit(
        [&](const auto& value) -> std::variant<FeExprPtr, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IntLiteral>) {
                return FeExpr::constant(*arena_, FeValue::intValue(value.value), FeType::intType());
            } else if constexpr (std::is_same_v<T, FloatLiteral>) {
                return FeExpr::constant(*arena_, FeValue::floatValue(value.value), FeType::floatType());
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                return FeExpr::constant(*arena_, FeValue::boolValue(value.value), FeType::boolType());
            } else if constexpr (std::is_same_v<T, StringLiteral>) {
                return FeExpr::constant(*arena_, FeValue::stringValue(value.value), FeType::voidType());
            } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
                if (value.name == "None") {
                    return FeExpr::constant(*arena_, FeValue::none(), FeType::none());
                }
                auto identifier = semanticIdentifierForExpr(expr, value.name);
                if (!identifier) {
                    return errorSpan(expr.span, "Frontend lowering missing semantic identifier info for '" + value.name + "'");
                }
                if (identifier->target == SemanticSymbolKind::Config) {
                    return errorSpan(expr.span, "Config object '" + value.name + "' cannot appear directly in lowered IR");
                }
                return FeExpr::var(*arena_, value.name, lowerType(identifier->type));
            } else if constexpr (std::is_same_v<T, CallExpr>) {
                auto call = semanticCallForExpr(expr, value.callee);
                if (!call) {
                    return errorSpan(expr.span, "Frontend lowering missing semantic call info for '" + value.callee + "'");
                }
                std::vector<FeCallArg> args;
                for (const auto& arg : value.args) {
                    auto lowered = lowerExpr(*arg.value);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
                }
                FeType resultType = lowerType(call->resultType);
                if (call->target == SemanticCallTargetKind::CallableLocal) {
                    auto symbol = findSymbol(value.callee);
                    if (!symbol) {
                        return errorSpan(expr.span, "Frontend lowering could not resolve callable '" + value.callee + "'");
                    }
                    return FeExpr::apply(*arena_, FeExpr::var(*arena_, value.callee, *symbol), std::move(args), resultType);
                }
                if (isCallableLibraryOp(value.callee) && resultType.kind == FeTypeKind::Callable) {
                    return FeExpr::layerCtor(*arena_, value.callee, std::move(args), resultType);
                }
                return FeExpr::call(*arena_, value.callee, std::move(args), resultType);
            } else if constexpr (std::is_same_v<T, RepeatExpr>) {
                return errorSpan(expr.span, "Repeat suffix '[n]' is only valid inside arrow pipeline stages");
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                if (value.op == TokenType::Dot) {
                    auto access = semanticConfigFieldAccessForExpr(expr);
                    if (!access) {
                        return errorSpan(expr.span, "Frontend lowering missing semantic config-field access info");
                    }
                    auto constant = evalConfigField(access->configName, access->fieldName, expr.span);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&constant)) {
                        return *diagnostic;
                    }
                    return FeExpr::constant(*arena_, std::get<FeValue>(std::move(constant)), lowerType(access->fieldType));
                }
                auto op = lowerBinaryOp(value.op, expr.span);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
                    return *diagnostic;
                }
                auto lhs = lowerExpr(*value.lhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
                    return *diagnostic;
                }
                auto rhs = lowerExpr(*value.rhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
                    return *diagnostic;
                }
                auto type = requiredSemanticTypeForExpr(expr, "binary expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::binary(*arena_, std::get<FeBinaryOp>(op), std::get<FeExprPtr>(std::move(lhs)), std::get<FeExprPtr>(std::move(rhs)), std::get<FeType>(std::move(type)));
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                auto operand = lowerExpr(*value.operand);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&operand)) {
                    return *diagnostic;
                }
                auto type = requiredSemanticTypeForExpr(expr, "unary expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                if (value.op == TokenType::Minus) {
                    return FeExpr::binary(*arena_, 
                        FeBinaryOp::Sub,
                        FeExpr::constant(*arena_, FeValue::intValue(0), FeType::intType()),
                        std::get<FeExprPtr>(std::move(operand)),
                        std::get<FeType>(std::move(type))
                    );
                }
                auto op = lowerBinaryOp(value.op, expr.span);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
                    return *diagnostic;
                }
                return FeExpr::binary(*arena_, 
                    std::get<FeBinaryOp>(op),
                    std::get<FeExprPtr>(std::move(operand)),
                    FeExpr::constant(*arena_, FeValue::boolValue(false), FeType::boolType()),
                    std::get<FeType>(std::move(type))
                );
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                auto condition = lowerExpr(*value.condition);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                auto thenExpr = lowerExpr(*value.thenExpr);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&thenExpr)) {
                    return *diagnostic;
                }
                auto elseExpr = lowerExpr(*value.elseExpr);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&elseExpr)) {
                    return *diagnostic;
                }
                auto type = requiredSemanticTypeForExpr(expr, "ternary expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::ifThenElse(*arena_, 
                    std::get<FeExprPtr>(std::move(condition)),
                    std::get<FeExprPtr>(std::move(thenExpr)),
                    std::get<FeExprPtr>(std::move(elseExpr)),
                    std::get<FeType>(std::move(type))
                );
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                std::vector<FeExprPtr> elements;
                for (const auto& element : value.elements) {
                    auto lowered = lowerExpr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
                }
                auto type = requiredSemanticTypeForExpr(expr, "tuple expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::tuple(*arena_, std::move(elements), std::get<FeType>(std::move(type)));
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                std::vector<FeExprPtr> elements;
                for (const auto& element : value.elements) {
                    auto lowered = lowerExpr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
                }
                auto type = requiredSemanticTypeForExpr(expr, "list expression");
                if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
                    return *diagnostic;
                }
                return FeExpr::list(*arena_, std::move(elements), std::get<FeType>(std::move(type)));
            } else if constexpr (std::is_same_v<T, ArrowExpr>) {
                return lowerArrowExpr(expr);
            }
        },
        expr.kind
    );
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lowerArrowExpr(const Expr& expr) {
    // Lowers an arrow expression (pipeline) like `x |> f()` by iteratively lowering
    // the source and applying each stage as if it were a function call or operation on the running value.
    const auto* arrow = std::get_if<ArrowExpr>(&expr.kind);
    if (arrow == nullptr) {
        return lowerExpr(expr);
    }
    auto current = lowerExpr(*arrow->source);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&current)) {
        return *diagnostic;
    }
    FeExprPtr value = std::get<FeExprPtr>(std::move(current));
    for (const auto& stage : arrow->stages) {
        auto lowered = lowerArrowStageExpr(*stage, value);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        value = std::get<FeExprPtr>(std::move(lowered));
    }
    return value;
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lowerArrowCallStage(
    const std::string& callee,
    const std::vector<CallArgument>& args,
    const SourceSpan& span,
    FeExprPtr current
) {
    auto call = semanticCallForArrowStage(callee, span);
    if (!call) {
        return errorSpan(span, "Frontend lowering missing semantic call info for arrow stage '" + callee + "'");
    }
    return lowerSemanticArrowCallStage(*call, callee, args, std::move(current));
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lowerSemanticArrowCallStage(
    const SemanticCallInfo& call,
    const std::string& callee,
    const std::vector<CallArgument>& args,
    FeExprPtr current
) {
    FeType resultType = lowerType(call.resultType);
    if (call.target == SemanticCallTargetKind::CallableLocal) {
        auto symbol = findSymbol(callee);
        if (!symbol) {
            return errorSpan(call.span, "Frontend lowering could not resolve callable '" + callee + "'");
        }
        std::vector<FeCallArg> apply_args{{std::nullopt, std::move(current)}};
        for (const auto& arg : args) {
            auto lowered = lowerExpr(*arg.value);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            apply_args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
        }
        return FeExpr::apply(*arena_, FeExpr::var(*arena_, callee, *symbol), std::move(apply_args), resultType);
    }

    if (call.target == SemanticCallTargetKind::BuiltinFunction || call.target == SemanticCallTargetKind::Function) {
        auto global = globalSymbols_.find(callee);
        if (global != globalSymbols_.end() && global->second.kind == FeTypeKind::Callable) {
            std::vector<FeCallArg> ctor_args;
            for (const auto& arg : args) {
                auto lowered = lowerExpr(*arg.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                ctor_args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
            }
            FeExprPtr callee_expr = isCallableLibraryOp(callee)
                ? FeExpr::layerCtor(*arena_, callee, std::move(ctor_args), global->second)
                : FeExpr::call(*arena_, callee, std::move(ctor_args), global->second);
            return FeExpr::apply(*arena_, callee_expr, std::vector<FeCallArg>{{std::nullopt, std::move(current)}}, resultType);
        }
    }

    std::vector<FeCallArg> lowered_args{{std::nullopt, std::move(current)}};
    for (const auto& arg : args) {
        auto lowered = lowerExpr(*arg.value);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
            return *diagnostic;
        }
        lowered_args.push_back(FeCallArg{arg.name, std::get<FeExprPtr>(std::move(lowered))});
    }
    return FeExpr::call(*arena_, callee, std::move(lowered_args), resultType);
}

std::variant<FeExprPtr, Diagnostic> FrontendLowerer::lowerArrowStageExpr(const Expr& expr, FeExprPtr current) {
    if (const auto* call = std::get_if<CallExpr>(&expr.kind)) {
        return lowerArrowCallStage(call->callee, call->args, expr.span, std::move(current));
    }
    if (const auto* repeat = std::get_if<RepeatExpr>(&expr.kind)) {
        const auto* call = std::get_if<CallExpr>(&repeat->stage->kind);
        if (call == nullptr) {
            return errorSpan(repeat->stage->span, "Repeated arrow stage must begin with a call");
        }
        auto count = evalConstantExpr(*repeat->count);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&count)) {
            return *diagnostic;
        }
        auto repeat_value = as_int(std::get<FeValue>(count), repeat->count->span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&repeat_value)) {
            return *diagnostic;
        }
        FeExprPtr value = std::move(current);
        for (std::int64_t index = 0; index < std::get<std::int64_t>(repeat_value); ++index) {
            auto lowered = lowerArrowCallStage(call->callee, call->args, repeat->stage->span, value);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            value = std::get<FeExprPtr>(std::move(lowered));
        }
        return value;
    }
    if (const auto* binary = std::get_if<BinaryExpr>(&expr.kind)) {
        auto lhs = count_arrow_stage_sites(*binary->lhs) > 0 ? lowerArrowStageExpr(*binary->lhs, current)
                                                             : lowerExpr(*binary->lhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
            return *diagnostic;
        }
        auto rhs = count_arrow_stage_sites(*binary->rhs) > 0 ? lowerArrowStageExpr(*binary->rhs, current)
                                                             : lowerExpr(*binary->rhs);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
            return *diagnostic;
        }
        auto op = lowerBinaryOp(binary->op, expr.span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
            return *diagnostic;
        }
        auto type = requiredSemanticTypeForExpr(expr, "arrow binary expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::binary(*arena_, 
            std::get<FeBinaryOp>(op),
            std::get<FeExprPtr>(std::move(lhs)),
            std::get<FeExprPtr>(std::move(rhs)),
            std::get<FeType>(std::move(type))
        );
    }
    if (const auto* unary = std::get_if<UnaryExpr>(&expr.kind)) {
        auto operand = count_arrow_stage_sites(*unary->operand) > 0 ? lowerArrowStageExpr(*unary->operand, current)
                                                                    : lowerExpr(*unary->operand);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&operand)) {
            return *diagnostic;
        }
        auto type = requiredSemanticTypeForExpr(expr, "arrow unary expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        if (unary->op == TokenType::Minus) {
            return FeExpr::binary(*arena_, 
                FeBinaryOp::Sub,
                FeExpr::constant(*arena_, FeValue::intValue(0), FeType::intType()),
                std::get<FeExprPtr>(std::move(operand)),
                std::get<FeType>(std::move(type))
            );
        }
        auto op = lowerBinaryOp(unary->op, expr.span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&op)) {
            return *diagnostic;
        }
        return FeExpr::binary(*arena_, 
            std::get<FeBinaryOp>(op),
            std::get<FeExprPtr>(std::move(operand)),
            FeExpr::constant(*arena_, FeValue::boolValue(false), FeType::boolType()),
            std::get<FeType>(std::move(type))
        );
    }
    if (const auto* ternary = std::get_if<TernaryExpr>(&expr.kind)) {
        auto condition = count_arrow_stage_sites(*ternary->condition) > 0
            ? lowerArrowStageExpr(*ternary->condition, current)
            : lowerExpr(*ternary->condition);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
            return *diagnostic;
        }
        auto thenExpr = count_arrow_stage_sites(*ternary->thenExpr) > 0
            ? lowerArrowStageExpr(*ternary->thenExpr, current)
            : lowerExpr(*ternary->thenExpr);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&thenExpr)) {
            return *diagnostic;
        }
        auto elseExpr = count_arrow_stage_sites(*ternary->elseExpr) > 0
            ? lowerArrowStageExpr(*ternary->elseExpr, current)
            : lowerExpr(*ternary->elseExpr);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&elseExpr)) {
            return *diagnostic;
        }
        auto type = requiredSemanticTypeForExpr(expr, "arrow ternary expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::ifThenElse(*arena_, 
            std::get<FeExprPtr>(std::move(condition)),
            std::get<FeExprPtr>(std::move(thenExpr)),
            std::get<FeExprPtr>(std::move(elseExpr)),
            std::get<FeType>(std::move(type))
        );
    }
    if (const auto* tuple = std::get_if<TupleExpr>(&expr.kind)) {
        std::vector<FeExprPtr> elements;
        for (const auto& element : tuple->elements) {
            auto lowered = count_arrow_stage_sites(*element) > 0 ? lowerArrowStageExpr(*element, current)
                                                                 : lowerExpr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
        }
        auto type = requiredSemanticTypeForExpr(expr, "arrow tuple expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::tuple(*arena_, std::move(elements), std::get<FeType>(std::move(type)));
    }
    if (const auto* list = std::get_if<ListExpr>(&expr.kind)) {
        std::vector<FeExprPtr> elements;
        for (const auto& element : list->elements) {
            auto lowered = count_arrow_stage_sites(*element) > 0 ? lowerArrowStageExpr(*element, current)
                                                                 : lowerExpr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                return *diagnostic;
            }
            elements.push_back(std::get<FeExprPtr>(std::move(lowered)));
        }
        auto type = requiredSemanticTypeForExpr(expr, "arrow list expression");
        if (const auto* diagnostic = std::get_if<Diagnostic>(&type)) {
            return *diagnostic;
        }
        return FeExpr::list(*arena_, std::move(elements), std::get<FeType>(std::move(type)));
    }
    if (count_arrow_stage_sites(expr) == 0) {
        return lowerExpr(expr);
    }
    return errorSpan(expr.span, "Unsupported compound arrow stage in frontend lowering");
}

std::variant<std::vector<FeStmt>, Diagnostic> FrontendLowerer::lowerScope(const Stmt& stmt) {
    const auto* scope = std::get_if<ScopeStmt>(&stmt.kind);
    if (scope == nullptr) {
        return errorSpan(stmt.span, "Expected scope statement while lowering");
    }
    auto saved = currentSymbols_;
    std::vector<FeStmt> lowered;
    for (const auto& child : scope->statements) {
        auto stmt_result = lowerStmt(child);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&stmt_result)) {
            return *diagnostic;
        }
        lowered.push_back(std::get<FeStmt>(std::move(stmt_result)));
    }
    currentSymbols_ = saved;
    return lowered;
}

std::variant<FeStmt, Diagnostic> FrontendLowerer::lowerStmt(const Stmt& stmt) {
    return std::visit(
        [&](const auto& value) -> std::variant<FeStmt, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, ReturnStmt>) {
                auto lowered = lowerExpr(*value.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                return FeStmt{FeReturnStmt{std::get<FeExprPtr>(std::move(lowered))}};
            } else if constexpr (std::is_same_v<T, ExprStmt>) {
                auto lowered = lowerExpr(*value.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                return FeStmt{FeExprStmt{std::get<FeExprPtr>(std::move(lowered))}};
            } else if constexpr (std::is_same_v<T, VarDecl>) {
                auto declaration = semanticDeclarationForStmt(stmt, value.name);
                if (!declaration) {
                    return errorSpan(stmt.span, "Frontend lowering missing semantic declaration info for '" + value.name + "'");
                }
                FeExprPtr lowered_value = nullptr;
                bool hasValue = false;
                if (value.init) {
                    auto lowered = lowerExpr(*value.init);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                        return *diagnostic;
                    }
                    lowered_value = std::get<FeExprPtr>(std::move(lowered));
                    hasValue = true;
                }
                FeType type = lowerType(declaration->finalType);
                bindSymbol(value.name, type);
                return FeStmt{FeVarDeclStmt{value.name, type, lowered_value, hasValue}};
            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                auto assignment = semanticAssignmentForStmt(stmt, value.name);
                if (!assignment) {
                    return errorSpan(stmt.span, "Frontend lowering missing semantic assignment info for '" + value.name + "'");
                }
                auto lowered = lowerExpr(*value.value);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered)) {
                    return *diagnostic;
                }
                bindSymbol(value.name, lowerType(assignment->targetType));
                return FeStmt{FeAssignStmt{value.name, std::get<FeExprPtr>(std::move(lowered))}};
            } else if constexpr (std::is_same_v<T, ScopeStmt>) {
                return errorSpan(stmt.span, "Nested standalone scope statements are not supported in FE lowering");
            } else if constexpr (std::is_same_v<T, IfStmt>) {
                auto condition = lowerExpr(*value.condition);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                auto thenBody = lowerScope(*value.thenStmt);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&thenBody)) {
                    return *diagnostic;
                }
                std::vector<FeElifBody> elifs;
                for (const auto& branch : value.elifs) {
                    auto branch_condition = lowerExpr(*branch.condition);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&branch_condition)) {
                        return *diagnostic;
                    }
                    auto body = lowerScope(*branch.body);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&body)) {
                        return *diagnostic;
                    }
                    elifs.push_back(FeElifBody{
                        std::get<FeExprPtr>(std::move(branch_condition)),
                        std::get<std::vector<FeStmt>>(std::move(body)),
                    });
                }
                std::vector<FeStmt> elseBody;
                if (value.elseStmt) {
                    auto lowered_else = lowerScope(*value.elseStmt);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&lowered_else)) {
                        return *diagnostic;
                    }
                    elseBody = std::get<std::vector<FeStmt>>(std::move(lowered_else));
                }
                return FeStmt{FeIfStmt{
                    std::get<FeExprPtr>(std::move(condition)),
                    std::get<std::vector<FeStmt>>(std::move(thenBody)),
                    std::move(elifs),
                    std::move(elseBody),
                }};
            }
        },
        stmt.kind
    );
}

std::variant<FeFunction, Diagnostic> FrontendLowerer::lowerFunction(const Function& function) {
    FeFunction lowered;
    lowered.name = function.name;
    lowered.returnType = lowerType(function.returnType);
    auto saved_symbols = currentSymbols_;
    auto saved_owner = currentOwner_;
    currentSymbols_.clear();
    currentOwner_ = function.name;
    for (const auto& arg : function.args) {
        FeType type = lowerType(arg.type);
        lowered.params.push_back({arg.name, type});
        bindSymbol(arg.name, type);
    }
    auto body = lowerScope(function.body);
    currentSymbols_ = saved_symbols;
    currentOwner_ = saved_owner;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&body)) {
        return *diagnostic;
    }
    lowered.body = std::get<std::vector<FeStmt>>(std::move(body));
    return lowered;
}

std::variant<FeLayer, Diagnostic> FrontendLowerer::lowerLayer(const Layer& layer) {
    FeLayer lowered;
    lowered.name = layer.name;
    lowered.returnType = lowerType(layer.returnType);
    auto saved_symbols = currentSymbols_;
    auto saved_owner = currentOwner_;
    currentSymbols_.clear();
    currentOwner_ = layer.name;
    for (const auto& arg : layer.args) {
        FeType type = lowerType(arg.type);
        lowered.params.push_back({arg.name, type});
        bindSymbol(arg.name, type);
    }
    auto body = lowerScope(layer.body);
    currentSymbols_ = saved_symbols;
    currentOwner_ = saved_owner;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&body)) {
        return *diagnostic;
    }
    lowered.body = std::get<std::vector<FeStmt>>(std::move(body));
    for (const auto& stmt : lowered.body) {
        if (const auto* ret = std::get_if<FeReturnStmt>(&stmt.kind)) {
            if (const auto* var = std::get_if<FeVarExpr>(&ret->value->kind)) {
                lowered.namedOutputs.push_back({var->symbol, ret->value->type});
            } else if (const auto* tuple = std::get_if<FeTupleExpr>(&ret->value->kind)) {
                for (const auto& element : tuple->elements) {
                    if (const auto* var = std::get_if<FeVarExpr>(&element->kind)) {
                        lowered.namedOutputs.push_back({var->symbol, element->type});
                    }
                }
            }
        }
    }
    return lowered;
}

std::variant<FeConfig, Diagnostic> FrontendLowerer::lowerConfig(const Config& config) {
    FeConfig lowered;
    lowered.name = config.name;
    for (const auto& field : config.fields) {
        auto value = evalConfigField(config.name, field.name, config.span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
            return *diagnostic;
        }
        lowered.fields[field.name] = std::get<FeValue>(std::move(value));
    }
    return lowered;
}

std::variant<FeTrain, Diagnostic> FrontendLowerer::lowerTrainConfig(const Config& config) {
    FeTrain lowered;
    lowered.name = config.name;
    lowered.variantCount = 1;
    for (const auto& field : config.fields) {
        std::vector<FeValue> values;
        if (field.init) {
            auto evaluated = evalConstantFieldValues(*field.init);
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
        } else if (field.name == "lr" || field.name == "learningRate") {
            lowered.learningRates = values;
        } else if (field.name == "iteration") {
            lowered.iterations = values;
        } else if (field.name == "objective") {
            lowered.objectiveSymbols.clear();
            for (const auto& value : values) {
                const auto* stringValue = std::get_if<std::string>(&value.value);
                if (stringValue == nullptr) {
                    return errorSpan(config.span, "Training config field 'objective' must reference a named tensor root");
                }
                lowered.objectiveSymbols.push_back(*stringValue);
            }
        } else {
            assigned = false;
        }
        if (assigned) {
            lowered.variantCount = std::max(lowered.variantCount, values.size());
        } else {
            if (values.size() != 1) {
                return errorSpan(config.span, "Training config field '" + field.name + "' does not support tuple variants; use a scalar value");
            }
            lowered.extraProperties[field.name] = values.front();
        }
    }
    return lowered;
}

std::variant<FeExecutionPlan, Diagnostic> FrontendLowerer::buildExecutionPlan(const LoweredModule& module) {
    auto model = std::find_if(module.layers.begin(), module.layers.end(), [](const FeLayer& layer) {
        return layer.name == "model";
    });
    auto train = std::find_if(module.trains.begin(), module.trains.end(), [](const FeTrain& item) {
        return item.name == "model";
    });
    if (model == module.layers.end() || train == module.trains.end()) {
        return error("Training config lowering requires layer model and config model");
    }
    FeExecutionPlan plan;
    plan.modelEntry = model->name;
    for (std::size_t index = 0; index < train->variantCount; ++index) {
        FeExecutionRun run;
        run.runName = train->variantCount == 1 ? "model" : "model_" + std::to_string(index + 1);
        run.modelName = model->name;
        run.trainName = train->name;
        run.backend = pick_broadcast_value(train->backends, index);
        run.optimizer = pick_broadcast_value(train->optimizers, index);
        run.learningRate = pick_broadcast_value(train->learningRates, index);
        run.objectiveSymbol = pick_broadcast_string(train->objectiveSymbols, index);
        run.iteration = pick_broadcast_value(train->iterations, index);
        if (run.objectiveSymbol) {
            for (const auto& output : model->namedOutputs) {
                if (output.first == *run.objectiveSymbol) {
                    run.objectiveSource = ObjectiveSource::Output;
                    run.objectiveType = output.second;
                }
            }
            if (run.objectiveSource == ObjectiveSource::Unknown) {
                for (const auto& param : model->params) {
                    if (param.first == *run.objectiveSymbol) {
                        run.objectiveSource = ObjectiveSource::Param;
                        run.objectiveType = param.second;
                    }
                }
            }
            if (run.objectiveSource == ObjectiveSource::Unknown) {
                for (const auto& stmt : model->body) {
                    if (resolveObjectiveStmt(stmt, *run.objectiveSymbol, run)) {
                        break;
                    }
                }
            }
        }
        plan.runs.push_back(std::move(run));
    }
    return plan;
}

bool FrontendLowerer::resolveObjectiveStmt(
    const FeStmt& stmt,
    const std::string& objectiveSymbol,
    FeExecutionRun& run
) const {
    return std::visit(
        [&](const auto& value) -> bool {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, FeVarDeclStmt>) {
                if (value.name != objectiveSymbol) {
                    return false;
                }
                run.objectiveSource = ObjectiveSource::Local;
                run.objectiveType = value.type;
                return true;
            } else if constexpr (std::is_same_v<T, FeAssignStmt>) {
                if (value.name != objectiveSymbol) {
                    return false;
                }
                run.objectiveSource = ObjectiveSource::Local;
                run.objectiveType = value.value->type;
                return true;
            } else if constexpr (std::is_same_v<T, FeIfStmt>) {
                for (const auto& child : value.thenBody) {
                    if (resolveObjectiveStmt(child, objectiveSymbol, run)) {
                        return true;
                    }
                }
                for (const auto& elif : value.elifBodies) {
                    for (const auto& child : elif.body) {
                        if (resolveObjectiveStmt(child, objectiveSymbol, run)) {
                            return true;
                        }
                    }
                }
                for (const auto& child : value.elseBody) {
                    if (resolveObjectiveStmt(child, objectiveSymbol, run)) {
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

std::variant<FeValue, Diagnostic> FrontendLowerer::evalConstantExpr(const Expr& expr) {
    return std::visit(
        [&](const auto& value) -> std::variant<FeValue, Diagnostic> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, IntLiteral>) {
                return FeValue::intValue(value.value);
            } else if constexpr (std::is_same_v<T, FloatLiteral>) {
                return FeValue::floatValue(value.value);
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                return FeValue::boolValue(value.value);
            } else if constexpr (std::is_same_v<T, StringLiteral>) {
                return FeValue::stringValue(value.value);
            } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
                if (value.name == "None") {
                    return FeValue::none();
                }
                return FeValue::stringValue(value.name);
            } else if constexpr (std::is_same_v<T, TupleExpr>) {
                std::vector<FeValue> values;
                for (const auto& element : value.elements) {
                    auto evaluated = evalConstantExpr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                        return *diagnostic;
                    }
                    values.push_back(std::get<FeValue>(std::move(evaluated)));
                }
                return FeValue::tupleValue(std::move(values));
            } else if constexpr (std::is_same_v<T, ListExpr>) {
                std::vector<FeValue> values;
                for (const auto& element : value.elements) {
                    auto evaluated = evalConstantExpr(*element);
                    if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                        return *diagnostic;
                    }
                    values.push_back(std::get<FeValue>(std::move(evaluated)));
                }
                return FeValue::listValue(std::move(values));
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                auto operand = evalConstantExpr(*value.operand);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&operand)) {
                    return *diagnostic;
                }
                return evalUnary(value.op, std::get<FeValue>(std::move(operand)), expr.span);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                if (value.op == TokenType::Dot) {
                    auto access = semanticConfigFieldAccessForExpr(expr);
                    if (!access) {
                        return errorSpan(expr.span, "Frontend lowering missing semantic config-field access info");
                    }
                    return evalConfigField(access->configName, access->fieldName, expr.span);
                }
                auto lhs = evalConstantExpr(*value.lhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&lhs)) {
                    return *diagnostic;
                }
                auto rhs = evalConstantExpr(*value.rhs);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&rhs)) {
                    return *diagnostic;
                }
                return evalBinary(value.op, std::get<FeValue>(std::move(lhs)), std::get<FeValue>(std::move(rhs)), expr.span);
            } else if constexpr (std::is_same_v<T, TernaryExpr>) {
                auto condition = evalConstantExpr(*value.condition);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&condition)) {
                    return *diagnostic;
                }
                auto bool_condition = as_bool(std::get<FeValue>(condition), value.condition->span);
                if (const auto* diagnostic = std::get_if<Diagnostic>(&bool_condition)) {
                    return *diagnostic;
                }
                return std::get<bool>(bool_condition) ? evalConstantExpr(*value.thenExpr)
                                                      : evalConstantExpr(*value.elseExpr);
            } else {
                return errorSpan(expr.span, "Expression is not compile-time constant");
            }
        },
        expr.kind
    );
}

std::variant<std::vector<FeValue>, Diagnostic> FrontendLowerer::evalConstantFieldValues(const Expr& expr) {
    if (const auto* tuple = std::get_if<TupleExpr>(&expr.kind)) {
        std::vector<FeValue> values;
        for (const auto& element : tuple->elements) {
            auto evaluated = evalConstantExpr(*element);
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
            auto evaluated = evalConstantExpr(*element);
            if (const auto* diagnostic = std::get_if<Diagnostic>(&evaluated)) {
                return *diagnostic;
            }
            values.push_back(std::get<FeValue>(std::move(evaluated)));
        }
        return values;
    }
    auto value = evalConstantExpr(expr);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
        return *diagnostic;
    }
    return std::vector<FeValue>{std::get<FeValue>(std::move(value))};
}

std::variant<FeValue, Diagnostic> FrontendLowerer::evalConfigField(
    const std::string& configName,
    const std::string& fieldName,
    const SourceSpan& span
) {
    // Evaluates a config field into a constant FeValue.
    // Uses configFieldCache_ to memoize results and detect cyclic dependencies
    // across config references during compile-time evaluation.
    auto config = configDefs_.find(configName);
    if (config == configDefs_.end()) {
        return errorSpan(span, "Unknown config '" + configName + "'");
    }
    auto& cache = configFieldCache_[configName][fieldName];
    if (cache.computed) {
        return cache.value;
    }
    if (cache.inProgress) {
        return errorSpan(span, "Cycle detected while evaluating config field '" + configName + "." + fieldName + "'");
    }
    const Field* field = nullptr;
    for (const auto& candidate : config->second->fields) {
        if (candidate.name == fieldName) {
            field = &candidate;
            break;
        }
    }
    if (field == nullptr) {
        return errorSpan(span, "Config field '" + configName + "." + fieldName + "' does not exist");
    }
    if (!field->init) {
        return errorSpan(span, "Config field '" + configName + "." + fieldName + "' does not have a compile-time value");
    }
    cache.inProgress = true;
    auto value = evalConstantExpr(*field->init);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
        cache.inProgress = false;
        return *diagnostic;
    }
    cache.value = std::get<FeValue>(std::move(value));
    cache.computed = true;
    cache.inProgress = false;
    return cache.value;
}

std::variant<FeValue, Diagnostic> FrontendLowerer::evalBinary(TokenType op, const FeValue& lhs, const FeValue& rhs, const SourceSpan& span) {
    auto lhs_num = [&]() -> std::variant<double, Diagnostic> { return as_double(lhs, span); };
    auto rhs_num = [&]() -> std::variant<double, Diagnostic> { return as_double(rhs, span); };
    const bool both_int = std::holds_alternative<std::int64_t>(lhs.value) && std::holds_alternative<std::int64_t>(rhs.value);
    if (op == TokenType::Plus || op == TokenType::Minus || op == TokenType::Star || op == TokenType::DoubleSlash) {
        if (both_int) {
            const auto left = std::get<std::int64_t>(lhs.value);
            const auto right = std::get<std::int64_t>(rhs.value);
            if (op == TokenType::Plus) {
                return FeValue::intValue(left + right);
            }
            if (op == TokenType::Minus) {
                return FeValue::intValue(left - right);
            }
            if (op == TokenType::Star) {
                return FeValue::intValue(left * right);
            }
            return FeValue::intValue(static_cast<std::int64_t>(std::floor(static_cast<double>(left) / static_cast<double>(right))));
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
                return FeValue::floatValue(l + r);
            case TokenType::Minus:
                return FeValue::floatValue(l - r);
            case TokenType::Star:
                return FeValue::floatValue(l * r);
            case TokenType::Slash:
                return FeValue::floatValue(l / r);
            case TokenType::DoubleSlash:
                return FeValue::floatValue(std::floor(l / r));
            case TokenType::Lt:
                return FeValue::boolValue(l < r);
            case TokenType::Gt:
                return FeValue::boolValue(l > r);
            case TokenType::LtEq:
                return FeValue::boolValue(l <= r);
            case TokenType::GtEq:
                return FeValue::boolValue(l >= r);
            default:
                break;
        }
    }
    if (op == TokenType::EqEq) {
        if (is_none(lhs) || is_none(rhs)) {
            return FeValue::boolValue(is_none(lhs) && is_none(rhs));
        }
        if (std::holds_alternative<bool>(lhs.value) && std::holds_alternative<bool>(rhs.value)) {
            return FeValue::boolValue(std::get<bool>(lhs.value) == std::get<bool>(rhs.value));
        }
        auto left = as_double(lhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&left)) {
            return *diagnostic;
        }
        auto right = as_double(rhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&right)) {
            return *diagnostic;
        }
        return FeValue::boolValue(std::abs(std::get<double>(left) - std::get<double>(right)) < 1e-12);
    }
    if (op == TokenType::Neq) {
        auto equal = evalBinary(TokenType::EqEq, lhs, rhs, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&equal)) {
            return *diagnostic;
        }
        return FeValue::boolValue(!std::get<bool>(std::get<FeValue>(equal).value));
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
        return FeValue::boolValue(op == TokenType::AmpAmp ? (std::get<bool>(left) && std::get<bool>(right))
                                                           : (std::get<bool>(left) || std::get<bool>(right)));
    }
    return errorSpan(span, "Constant evaluation failed: unsupported binary operator");
}

std::variant<FeValue, Diagnostic> FrontendLowerer::evalUnary(TokenType op, const FeValue& operand, const SourceSpan& span) {
    if (op == TokenType::Minus) {
        if (std::holds_alternative<std::int64_t>(operand.value)) {
            return FeValue::intValue(-std::get<std::int64_t>(operand.value));
        }
        auto value = as_double(operand, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
            return *diagnostic;
        }
        return FeValue::floatValue(-std::get<double>(value));
    }
    if (op == TokenType::Bang) {
        auto value = as_bool(operand, span);
        if (const auto* diagnostic = std::get_if<Diagnostic>(&value)) {
            return *diagnostic;
        }
        return FeValue::boolValue(!std::get<bool>(value));
    }
    return errorSpan(span, "Constant evaluation failed: unsupported unary operator");
}

std::optional<SemanticCallInfo> FrontendLowerer::semanticCallForExpr(const Expr& expr, const std::string& callee) const {
    for (auto call = semanticInfo_.calls.rbegin(); call != semanticInfo_.calls.rend(); ++call) {
        if (call->callee == callee && same_span(call->span, expr.span) && !call->arrowStage &&
            owner_matches(call->owner, currentOwner_)) {
            return *call;
        }
    }
    return std::nullopt;
}

std::optional<SemanticCallInfo> FrontendLowerer::semanticCallForArrowStage(const std::string& callee, const SourceSpan& span) const {
    for (auto call = semanticInfo_.calls.rbegin(); call != semanticInfo_.calls.rend(); ++call) {
        if (call->callee == callee && same_span(call->span, span) && call->arrowStage &&
            owner_matches(call->owner, currentOwner_)) {
            return *call;
        }
    }
    return std::nullopt;
}

std::optional<SemanticIdentifierInfo> FrontendLowerer::semanticIdentifierForExpr(const Expr& expr, const std::string& name) const {
    for (auto identifier = semanticInfo_.identifiers.rbegin(); identifier != semanticInfo_.identifiers.rend(); ++identifier) {
        if (identifier->name == name && same_span(identifier->span, expr.span) &&
            owner_matches(identifier->owner, currentOwner_)) {
            return *identifier;
        }
    }
    return std::nullopt;
}

std::optional<SemanticAssignmentInfo> FrontendLowerer::semanticAssignmentForStmt(const Stmt& stmt, const std::string& name) const {
    for (auto assignment = semanticInfo_.assignments.rbegin(); assignment != semanticInfo_.assignments.rend(); ++assignment) {
        if (assignment->targetName == name && same_span(assignment->span, stmt.span) &&
            owner_matches(assignment->owner, currentOwner_)) {
            return *assignment;
        }
    }
    return std::nullopt;
}

std::optional<SemanticConfigFieldAccessInfo> FrontendLowerer::semanticConfigFieldAccessForExpr(const Expr& expr) const {
    for (auto access = semanticInfo_.configFieldAccesses.rbegin(); access != semanticInfo_.configFieldAccesses.rend(); ++access) {
        if (same_span(access->span, expr.span) && owner_matches(access->owner, currentOwner_)) {
            return *access;
        }
    }
    return std::nullopt;
}

std::optional<SemanticDeclarationInfo> FrontendLowerer::semanticDeclarationForStmt(const Stmt& stmt, const std::string& name) const {
    for (auto declaration = semanticInfo_.declarations.rbegin(); declaration != semanticInfo_.declarations.rend(); ++declaration) {
        if (declaration->name == name && same_span(declaration->span, stmt.span) &&
            owner_matches(declaration->owner, currentOwner_)) {
            return *declaration;
        }
    }
    return std::nullopt;
}

std::optional<FeType> FrontendLowerer::semanticTypeForExpr(const Expr& expr) const {
    for (auto info = semanticInfo_.exprs.rbegin(); info != semanticInfo_.exprs.rend(); ++info) {
        if (same_span(info->span, expr.span) && owner_matches(info->owner, currentOwner_)) {
            return lowerType(info->type);
        }
    }
    return std::nullopt;
}

std::variant<FeType, Diagnostic> FrontendLowerer::requiredSemanticTypeForExpr(const Expr& expr, const std::string& context) {
    auto type = semanticTypeForExpr(expr);
    if (!type) {
        return errorSpan(expr.span, "Frontend lowering missing semantic expression type info for " + context);
    }
    return *type;
}

void FrontendLowerer::bindSymbol(const std::string& name, FeType type) {
    currentSymbols_[name] = std::move(type);
}

std::optional<FeType> FrontendLowerer::findSymbol(const std::string& name) const {
    auto local = currentSymbols_.find(name);
    if (local != currentSymbols_.end()) {
        return local->second;
    }
    auto global = globalSymbols_.find(name);
    if (global != globalSymbols_.end()) {
        return global->second;
    }
    return std::nullopt;
}

Diagnostic FrontendLowerer::error(const std::string& message) {
    Diagnostic diagnostic = Diagnostic::error(DiagnosticCode::FrontendIrError, message);
    lastDiagnostic_ = diagnostic;
    return diagnostic;
}

Diagnostic FrontendLowerer::errorSpan(const SourceSpan& span, const std::string& message) {
    Diagnostic diagnostic = Diagnostic::error(DiagnosticCode::FrontendIrError, message).withSourceSpan(span);
    lastDiagnostic_ = diagnostic;
    return diagnostic;
}

std::string loweredModuleSummary(const LoweredModule& module) {
    std::ostringstream out;
    out << "lowered=configs:" << module.configs.size()
        << " trains:" << module.trains.size()
        << " functions:" << module.functions.size()
        << " globals:" << module.globals.size()
        << " executionPlan:" << (module.executionPlan ? "yes" : "no");
    return out.str();
}

std::string frontendIrToString(const LoweredModule& module) {
    std::ostringstream out;
    out << loweredModuleSummary(module) << '\n';
    for (const auto& config : module.configs) {
        out << "config " << config.name << " fields=" << config.fields.size() << '\n';
    }
    for (const auto& train : module.trains) {
        out << "train " << train.name << " variants=" << train.variantCount << '\n';
    }
    for (const auto& layer : module.layers) {
        out << "layer " << layer.name << '(';
        for (std::size_t index = 0; index < layer.params.size(); ++index) {
            if (index != 0) {
                out << ", ";
            }
            out << layer.params[index].first << ": " << fe_type_to_string(layer.params[index].second);
        }
        out << ") -> " << fe_type_to_string(layer.returnType) << '\n';
        for (const auto& stmt : layer.body) {
            append_stmt(out, stmt, 2);
        }
    }
    for (const auto& function : module.functions) {
        out << "fn " << function.name << '(';
        for (std::size_t index = 0; index < function.params.size(); ++index) {
            if (index != 0) {
                out << ", ";
            }
            out << function.params[index].first << ": " << fe_type_to_string(function.params[index].second);
        }
        out << ") -> " << fe_type_to_string(function.returnType) << '\n';
        for (const auto& stmt : function.body) {
            append_stmt(out, stmt, 2);
        }
    }
    for (const auto& stmt : module.globals) {
        append_stmt(out, stmt, 0);
    }
    return out.str();
}
