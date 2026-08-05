#include "semantic_analyzer.h"

#include "ops.h"
#include "parser.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace {

bool isCallable(const Type &type) { return type.base == TypeBase::Callable; }

Type tensor_any() {
  return Type::tensor(std::nullopt, std::nullopt, std::nullopt);
}

/**
 * @brief Infers the result type of a function or layer call based on its
 * declared type and arguments.
 *
 * Special-cases builtins (reshape, sum, mean, matmul, cross_entropy, etc.) to
 * compute tensor ranks and shapes from the inputs when possible.
 *
 * @param callee The name of the function or layer being called.
 * @param declared_type The statically declared return type.
 * @param argTypes The inferred types of the arguments passed to the call.
 * @return The inferred return type of the call.
 */
Type infer_call_result_type(const std::string &callee,
                            const Type &declared_type,
                            const std::vector<Type> &argTypes) {
  if (declared_type.base == TypeBase::Callable && !argTypes.empty()) {
    if (declared_type.callableReturn &&
        declared_type.callableReturn->base == TypeBase::Tensor &&
        argTypes[0].base == TypeBase::Tensor) {
      return Type::callable(
          Type::tensor(argTypes[0].tensorDtype,
                       declared_type.callableReturn->tensorShapeExpr,
                       argTypes[0].tensorRank));
    }
    return declared_type;
  }

  if (declared_type.base != TypeBase::Tensor || argTypes.empty() ||
      argTypes[0].base != TypeBase::Tensor) {
    return declared_type;
  }

  const Type &first = argTypes[0];
  if (callee == "reshape") {
    return Type::tensor(first.tensorDtype, std::nullopt, std::nullopt);
  }
  if (callee == "sum" || callee == "mean") {
    return Type::tensor(first.tensorDtype, std::nullopt, 1);
  }
  if (callee == "matmul") {
    return Type::tensor(first.tensorDtype, declared_type.tensorShapeExpr,
                        first.tensorRank ? first.tensorRank
                                         : declared_type.tensorRank);
  }
  if (callee == "cross_entropy") {
    return Type::tensor(first.tensorDtype, declared_type.tensorShapeExpr,
                        declared_type.tensorRank);
  }
  if (preservesFirstTensorArg(callee)) {
    return first;
  }
  return Type::tensor(first.tensorDtype, declared_type.tensorShapeExpr,
                      first.tensorRank);
}

Type default_unconstrained_numeric_type(Type type) {
  if (type.base == TypeBase::Int && !type.scalarDtype) {
    return Type::int32();
  }
  if (type.base == TypeBase::Float && !type.scalarDtype) {
    return Type::float64();
  }
  if (type.base == TypeBase::List || type.base == TypeBase::Tuple) {
    for (auto &element : type.elements) {
      element = default_unconstrained_numeric_type(std::move(element));
    }
  }
  return type;
}

/**
 * @brief Merges two float types during binary operations, picking the highest
 * precision.
 *
 * E.g., float16 + float64 -> float64.
 */
Type merge_scalar_float_types(const Type &lhs, const Type &rhs) {
  auto rank = [](const Type &type) {
    if (type.scalarDtype == "float64") {
      return 3;
    }
    if (type.scalarDtype == "float16") {
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
  if (!lhs.scalarDtype && !rhs.scalarDtype) {
    return Type::floatType();
  }
  return Type::float32();
}

Type merge_scalar_int_types(const Type &lhs, const Type &rhs) {
  auto rank = [](const Type &type) {
    if (type.scalarDtype == "int64") {
      return 3;
    }
    if (type.scalarDtype == "int16") {
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
  if (!lhs.scalarDtype && !rhs.scalarDtype) {
    return Type::intType();
  }
  return Type::int32();
}

int count_stage_sites(const Expr &expr);

int count_expr_list(const std::vector<ExprPtr> &exprs) {
  int count = 0;
  for (const auto &expr : exprs) {
    count += count_stage_sites(*expr);
  }
  return count;
}

int count_stage_sites(const Expr &expr) {
  return std::visit(
      [](const auto &value) -> int {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CallExpr>) {
          return 1;
        } else if constexpr (std::is_same_v<T, RepeatExpr>) {
          return count_stage_sites(*value.stage) +
                 count_stage_sites(*value.count);
        } else if constexpr (std::is_same_v<T, UnaryExpr>) {
          return count_stage_sites(*value.operand);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
          return count_stage_sites(*value.lhs) + count_stage_sites(*value.rhs);
        } else if constexpr (std::is_same_v<T, TernaryExpr>) {
          return count_stage_sites(*value.thenExpr) +
                 count_stage_sites(*value.condition) +
                 count_stage_sites(*value.elseExpr);
        } else if constexpr (std::is_same_v<T, TupleExpr>) {
          return count_expr_list(value.elements);
        } else if constexpr (std::is_same_v<T, ListExpr>) {
          return count_expr_list(value.elements);
        } else {
          return 0;
        }
      },
      expr.kind);
}

std::optional<std::size_t> tuple_arity(const Expr &expr) {
  if (const auto *tuple = std::get_if<TupleExpr>(&expr.kind)) {
    return tuple->elements.size();
  }
  return std::nullopt;
}

void collect_expr_identifiers(const Expr &expr, std::set<std::string> &symbols);

void collect_expr_list_identifiers(const std::vector<ExprPtr> &exprs,
                                   std::set<std::string> &symbols) {
  for (const auto &expr : exprs) {
    collect_expr_identifiers(*expr, symbols);
  }
}

void collect_expr_identifiers(const Expr &expr,
                              std::set<std::string> &symbols) {
  std::visit(
      [&](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, IdentifierExpr>) {
          if (value.name != "None") {
            symbols.insert(value.name);
          }
        } else if constexpr (std::is_same_v<T, CallExpr>) {
          for (const auto &arg : value.args) {
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
          collect_expr_identifiers(*value.thenExpr, symbols);
          collect_expr_identifiers(*value.condition, symbols);
          collect_expr_identifiers(*value.elseExpr, symbols);
        } else if constexpr (std::is_same_v<T, TupleExpr>) {
          collect_expr_list_identifiers(value.elements, symbols);
        } else if constexpr (std::is_same_v<T, ListExpr>) {
          collect_expr_list_identifiers(value.elements, symbols);
        } else if constexpr (std::is_same_v<T, ArrowExpr>) {
          collect_expr_identifiers(*value.source, symbols);
          collect_expr_list_identifiers(value.stages, symbols);
        }
      },
      expr.kind);
}

void collect_stmt_symbols(const Stmt &stmt, std::set<std::string> &symbols) {
  std::visit(
      [&](const auto &value) {
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
          for (const auto &inner : value.statements) {
            collect_stmt_symbols(inner, symbols);
          }
        } else if constexpr (std::is_same_v<T, IfStmt>) {
          collect_expr_identifiers(*value.condition, symbols);
          collect_stmt_symbols(*value.thenStmt, symbols);
          for (const auto &branch : value.elifs) {
            collect_expr_identifiers(*branch.condition, symbols);
            collect_stmt_symbols(*branch.body, symbols);
          }
          if (value.elseStmt) {
            collect_stmt_symbols(*value.elseStmt, symbols);
          }
        }
      },
      stmt.kind);
}

// Reviewed
bool is_train_config_field(const std::string &name) {
  return name == "backend" || name == "target" || name == "device" ||
         name == "optimizer" || name == "lr" || name == "learningRate" ||
         name == "objective" || name == "iteration";
}

// Reviewed
bool is_train_config(const Config &config) {
  return std::any_of(
      config.fields.begin(), config.fields.end(), [](const Field &field) {
        return field.type.base == TypeBase::Unknown &&
               (is_train_config_field(field.name) || field.name == "subtrain");
      });
}

} // namespace

SemanticAnalyzer::SemanticAnalyzer() { registerBuiltins(); }

std::optional<Diagnostic> SemanticAnalyzer::takeLastDiagnostic() {
  auto diagnostic = lastDiagnostic_;
  lastDiagnostic_.reset();
  return diagnostic;
}

SemanticResult SemanticAnalyzer::analyzeWithInfo(const Program &program) {
  beginAnalysis();
  if (auto diagnostic = analyze(program)) {
    return *diagnostic;
  }
  return semanticInfo_;
}

void SemanticAnalyzer::beginAnalysis() {
  scopes_.clear();
  functions_.clear();
  layers_.clear();
  configs_.clear();
  registerBuiltins();
  lastExprType_ = Type::noneType();
  currentReturnType_.reset();
  currentCallableHasReturn_ = false;
  currentCallableKind_ = CallableKind::None;
  currentCallableName_.reset();
  semanticInfo_ = SemanticInfo{};
  lastDiagnostic_.reset();

  for (const auto &spec : allBuiltinSignatures()) {
    recordSymbol(SemanticSymbol{
        spec.name,
        SemanticSymbolKind::BuiltinFunction,
        spec.returnType,
        0,
        std::nullopt,
        std::nullopt,
    });
  }
}

void SemanticAnalyzer::registerBuiltins() {
  for (const auto &spec : allBuiltinSignatures()) {
    functions_[spec.name] = Signature{spec.name, spec.returnType, spec.argTypes,
                                      spec.minArity, spec.maxArity};
  }
}
// U-Review
std::optional<Diagnostic> SemanticAnalyzer::analyze(const Program &program) {
  if (auto diagnostic = collectConfigs(program)) {
    return diagnostic;
  }
  if (auto diagnostic = collectLayers(program)) {
    return diagnostic;
  }
  if (auto diagnostic = collectFunctions(program)) {
    return diagnostic;
  }

  pushScope();
  for (const auto &stmt : program.globals) {
    if (auto diagnostic = analyzeStmt(stmt, program)) {
      return diagnostic;
    }
  }
  for (const auto &layer : program.layers) {
    if (auto diagnostic =
            visitCallable(layer.args, layer.returnType, layer.body, layer.span,
                          layer.name, CallableKind::Layer, "Layer", program)) {
      return diagnostic;
    }
  }
  for (const auto &function : program.functions) {
    if (auto diagnostic = visitCallable(
            function.args, function.returnType, function.body, function.span,
            function.name, CallableKind::Function, "Function", program)) {
      return diagnostic;
    }
  }
  popScope();
  return std::nullopt;
}

// Reviewed
std::optional<Diagnostic>
SemanticAnalyzer::collectConfigs(const Program &program) {
  for (const auto &config : program.configs) {
    if (configs_.count(config.name) != 0) {
      return error(config.span, "Duplicate config '" + config.name + "'");
    }
    std::map<std::string, Type> fields;
    for (const auto &field : config.fields) {
      if (auto diagnostic = validateDeclaredType(field.type, config.span)) {
        return diagnostic;
      }
      if (fields.count(field.name) != 0) {
        return error(config.span, "Duplicate field '" + field.name +
                                      "' in config '" + config.name + "'");
      }
      fields[field.name] = field.type;
    }
    configs_[config.name] = fields;
    recordSymbol(SemanticSymbol{config.name, SemanticSymbolKind::Config,
                                Type::noneType(), 0, std::nullopt,
                                config.span});
    for (const auto &field : config.fields) {
      recordSymbol(SemanticSymbol{field.name, SemanticSymbolKind::ConfigField,
                                  field.type, 0, config.name, config.span});
    }
    if (is_train_config(config)) {
      if (auto diagnostic = validateTrainConfig(config, program)) {
        return diagnostic;
      }
    }
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::collectLayers(const Program &program) {
  for (const auto &layer : program.layers) {
    if (auto diagnostic = validateDeclaredType(layer.returnType, layer.span)) {
      return diagnostic;
    }
    for (const auto &arg : layer.args) {
      if (auto diagnostic = validateDeclaredType(arg.type, layer.span)) {
        return diagnostic;
      }
    }
    if (layers_.count(layer.name) != 0) {
      return error(layer.span, "Duplicate layer '" + layer.name + "'");
    }
    // need attention
    if (functions_.count(layer.name) != 0) {
      return error(layer.span,
                   "Layer '" + layer.name +
                       "' conflicts with an existing function or builtin");
    }
    const auto minArity = static_cast<std::size_t>(
        std::count_if(layer.args.begin(), layer.args.end(),
                      [](const Arg &arg) { return !arg.defaultValue; }));
    std::vector<Type> argTypes;
    for (const auto &arg : layer.args) {
      argTypes.push_back(arg.type);
    }
    layers_[layer.name] = Signature{layer.name, layer.returnType, argTypes,
                                    minArity, layer.args.size()};
    recordSymbol(SemanticSymbol{layer.name, SemanticSymbolKind::Layer,
                                layer.returnType, 0, std::nullopt, layer.span});
  }
  return std::nullopt;
}
// Reviewed
std::optional<Diagnostic>
SemanticAnalyzer::collectFunctions(const Program &program) {
  for (const auto &function : program.functions) {
    if (auto diagnostic =
            validateDeclaredType(function.returnType, function.span)) {
      return diagnostic;
    }
    for (const auto &arg : function.args) {
      if (auto diagnostic = validateDeclaredType(arg.type, function.span)) {
        return diagnostic;
      }
    }
    if (functions_.count(function.name) != 0 ||
        layers_.count(function.name) != 0) {
      return error(function.span,
                   "Function '" + function.name +
                       "' conflicts with an existing declaration");
    }
    const auto minArity = static_cast<std::size_t>(
        std::count_if(function.args.begin(), function.args.end(),
                      [](const Arg &arg) { return !arg.defaultValue; }));
    std::vector<Type> argTypes;
    for (const auto &arg : function.args) {
      argTypes.push_back(arg.type);
    }
    functions_[function.name] =
        Signature{function.name, function.returnType, argTypes, minArity,
                  function.args.size()};
    recordSymbol(SemanticSymbol{function.name, SemanticSymbolKind::Function,
                                function.returnType, 0, std::nullopt,
                                function.span});
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeReturnStmt(const ReturnStmt &value,
                                    const SourceSpan &span,
                                    const Program &program) {
  auto analyzed = analyzeExpr(*value.value, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
    return *diagnostic;
  }
  Type valueType = std::get<Type>(std::move(analyzed));
  currentCallableHasReturn_ = true;
  if (currentReturnType_ &&
      !isCompatible(*currentReturnType_, valueType)) {
    return error(span, "Return type mismatch. Expected " +
                           typeToString(*currentReturnType_) +
                           ", got " + typeToString(valueType));
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeExprStmt(const ExprStmt &value,
                                  const Program &program) {
  auto analyzed = analyzeExpr(*value.value, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
    return *diagnostic;
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeVarDeclStmt(const VarDecl &value,
                                     std::uint32_t nodeId,
                                     const SourceSpan &span,
                                     const Program &program) {
  Type finalType = value.type;
  if (value.init) {
    auto analyzed = analyzeExpr(*value.init, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
      return *diagnostic;
    }
    Type init_type = std::get<Type>(std::move(analyzed));
    if (value.type.base == TypeBase::Unknown) {
      finalType =
          default_unconstrained_numeric_type(std::move(init_type));
    } else {
      if (auto diagnostic =
              validateDeclaredType(value.type, span)) {
        return diagnostic;
      }
      if (!isCompatible(value.type, init_type)) {
        return error(span, "Initialization type mismatch for '" +
                               value.name + "'");
      }
    }
  } else if (auto diagnostic =
                 validateDeclaredType(value.type, span)) {
    return diagnostic;
  }
  const auto kind = currentCallableName_ ? SemanticSymbolKind::Local
                                         : SemanticSymbolKind::Global;
  recordDeclaration(nodeId, span, value.name, kind, finalType);
  return declareVar(value.name, std::move(finalType), kind, span);
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeAssignStmt(const AssignStmt &value,
                                    std::uint32_t nodeId,
                                    const SourceSpan &span,
                                    const Program &program) {
  auto analyzed = analyzeExpr(*value.value, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
    return *diagnostic;
  }
  Type valueType = std::get<Type>(std::move(analyzed));
  const Symbol *symbol = findVar(value.name);
  if (symbol == nullptr) {
    // Variable not found, implicitly declare it in the current scope
    const auto kind = currentCallableName_ ? SemanticSymbolKind::Local
                                           : SemanticSymbolKind::Global;
    recordDeclaration(nodeId, span, value.name, kind, valueType);
    if (auto diagnostic =
            declareVar(value.name, valueType, kind, span)) {
      return diagnostic;
    }
    recordAssignment(nodeId, span, value.name, kind, valueType, valueType);
  } else {
    if (!isCompatible(symbol->type, valueType)) {
      return error(span,
                   "Assignment type mismatch for '" + value.name + "'");
    }
    recordAssignment(nodeId, span, value.name, symbol->kind, symbol->type,
                     valueType);
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeScopeStmt(const ScopeStmt &value,
                                   const Program &program) {
  pushScope();
  for (const auto &inner : value.statements) {
    if (auto diagnostic = analyzeStmt(inner, program)) {
      return diagnostic;
    }
  }
  popScope();
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeIfStmt(const IfStmt &value,
                                const Program &program) {
  auto condition = analyzeExpr(*value.condition, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&condition)) {
    return *diagnostic;
  }
  if (auto diagnostic =
          ensureConditionType(std::get<Type>(condition),
                              value.condition->span, "If condition")) {
    return diagnostic;
  }
  if (auto diagnostic = analyzeStmt(*value.thenStmt, program)) {
    return diagnostic;
  }
  for (const auto &branch : value.elifs) {
    auto branch_condition = analyzeExpr(*branch.condition, program);
    if (const auto *diagnostic =
            std::get_if<Diagnostic>(&branch_condition)) {
      return *diagnostic;
    }
    if (auto diagnostic = ensureConditionType(
            std::get<Type>(branch_condition), branch.condition->span,
            "Elif condition")) {
      return diagnostic;
    }
    if (auto diagnostic = analyzeStmt(*branch.body, program)) {
      return diagnostic;
    }
  }
  if (value.elseStmt) {
    return analyzeStmt(*value.elseStmt, program);
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::analyzeStmt(const Stmt &stmt, const Program &program) {
  return std::visit(
      [&](const auto &value) -> std::optional<Diagnostic> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ReturnStmt>) {
          return analyzeReturnStmt(value, stmt.span, program);
        } else if constexpr (std::is_same_v<T, ExprStmt>) {
          return analyzeExprStmt(value, program);
        } else if constexpr (std::is_same_v<T, VarDecl>) {
          return analyzeVarDeclStmt(value, stmt.id, stmt.span, program);
        } else if constexpr (std::is_same_v<T, AssignStmt>) {
          return analyzeAssignStmt(value, stmt.id, stmt.span, program);
        } else if constexpr (std::is_same_v<T, ScopeStmt>) {
          return analyzeScopeStmt(value, program);
        } else if constexpr (std::is_same_v<T, IfStmt>) {
          return analyzeIfStmt(value, program);
        }
        return std::nullopt;
      },
      stmt.kind);
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::analyzeExpr(const Expr &expr, const Program &program) {
  auto result = std::visit(
      [&](const auto &value) -> std::variant<Type, Diagnostic> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, IntLiteral>) { // the compiler keeps
                                                       // only the IntLiteral
                                                       // branch. why we are
                                                       // using the constexpr
          return Type::intType();
        } else if constexpr (std::is_same_v<T, FloatLiteral>) {
          return Type::floatType();
        } else if constexpr (std::is_same_v<T, BoolLiteral>) {
          return Type::boolType();
        } else if constexpr (std::is_same_v<T, StringLiteral>) {
          return Type::strType();
        } else if constexpr (std::is_same_v<T, IdentifierExpr>) {
          return visitIdentifier(expr, value.name, expr.span);
        } else if constexpr (std::is_same_v<T, CallExpr>) {
          return visitCall(expr, value.callee, value.args, expr.span, program);
        } else if constexpr (std::is_same_v<T, RepeatExpr>) {
          return error(
              expr.span,
              "Repeat suffix '[n]' is only valid inside arrow pipeline stages");
        } else if constexpr (std::is_same_v<T, UnaryExpr>) {
          return visitUnary(*value.operand, value.op, expr.span, program);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
          return visitBinary(expr, *value.lhs, *value.rhs, value.op, expr.span,
                             program);
        } else if constexpr (std::is_same_v<T, TernaryExpr>) {
          return visitTernary(*value.thenExpr, *value.condition,
                              *value.elseExpr, expr.span, program);
        } else if constexpr (std::is_same_v<T, TupleExpr>) {
          std::vector<Type> types;
          for (const auto &element : value.elements) {
            auto analyzed = analyzeExpr(*element, program);
            if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
              return *diagnostic;
            }
            types.push_back(std::get<Type>(std::move(analyzed)));
          }
          return Type::tuple(std::move(types));
        } else if constexpr (std::is_same_v<T, ListExpr>) {
          std::vector<Type> types;
          for (const auto &element : value.elements) {
            auto analyzed = analyzeExpr(*element, program);
            if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
              return *diagnostic;
            }
            Type element_type = std::get<Type>(std::move(analyzed));
            const bool known = std::any_of(
                types.begin(), types.end(), [&](const Type &known_type) {
                  return isCompatible(known_type, element_type);
                });
            if (!known) {
              types.push_back(std::move(element_type));
            }
          }
          return Type::list(std::move(types));
        } else if constexpr (std::is_same_v<T, ArrowExpr>) {
          return visitArrow(*value.source, value.stages, program);
        }
      },
      expr.kind);
  if (const auto *type = std::get_if<Type>(&result)) {
    lastExprType_ = *type;
    recordExprType(expr, *type);
  }
  return result;
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::visitIdentifier(const Expr &expr, const std::string &name,
                                  const SourceSpan &span) {
  if (name == "None") {
    return Type::noneType();
  }
  if (const Symbol *symbol = findVar(name)) {
    recordIdentifier(expr.id, span, name, symbol->kind, symbol->type);
    return symbol->type;
  }
  if (configs_.count(name) != 0) {
    recordIdentifier(expr.id, span, name, SemanticSymbolKind::Config, Type::noneType());
    return Type::noneType();
  }
  return error(span, "Undefined variable '" + name + "'");
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::visitCall(const Expr &expr, const std::string &callee,
                            const std::vector<CallArgument> &args,
                            const SourceSpan &span, const Program &program) {
  auto function = functions_.find(callee);
  if (function != functions_.end()) {
    if (auto diagnostic = ensureCallAllowed(callee, false, span)) {
      return *diagnostic;
    }
    if (auto diagnostic =
            validateSignatureArity(function->second, args.size(), span)) {
      return *diagnostic;
    }
    std::vector<Type> argTypes;
    for (std::size_t index = 0; index < args.size(); ++index) {
      auto analyzed = analyzeExpr(*args[index].value, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      Type actual = std::get<Type>(std::move(analyzed));
      argTypes.push_back(actual);
      if (index < function->second.argTypes.size() &&
          !isCompatible(function->second.argTypes[index], actual)) {
        return error(span, "Argument " + std::to_string(index + 1) + " to '" +
                               callee + "' has incompatible type. Expected " +
                               typeToString(function->second.argTypes[index]) +
                               ", got " + typeToString(actual));
      }
    }
    Type result =
        infer_call_result_type(callee, function->second.returnType, argTypes);
    recordCall(expr.id, span, callee,
               isBuiltinOp(callee) ? SemanticCallTargetKind::BuiltinFunction
                                   : SemanticCallTargetKind::Function,
               result, false);
    return result;
  }

  auto layer = layers_.find(callee);
  if (layer != layers_.end()) {
    if (auto diagnostic = ensureCallAllowed(callee, true, span)) {
      return *diagnostic;
    }
    if (auto diagnostic =
            validateSignatureArity(layer->second, args.size(), span)) {
      return *diagnostic;
    }
    std::vector<Type> argTypes;
    for (std::size_t index = 0; index < args.size(); ++index) {
      auto analyzed = analyzeExpr(*args[index].value, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      Type actual = std::get<Type>(std::move(analyzed));
      argTypes.push_back(actual);
      if (index < layer->second.argTypes.size() &&
          !isCompatible(layer->second.argTypes[index], actual)) {
        return error(span, "Argument " + std::to_string(index + 1) + " to '" +
                               callee + "' has incompatible type");
      }
    }
    Type result =
        infer_call_result_type(callee, layer->second.returnType, argTypes);
    recordCall(expr.id, span, callee, SemanticCallTargetKind::Layer, result, false);
    return result;
  }

  if (const Symbol *symbol = findVar(callee)) {
    if (!isCallable(symbol->type)) {
      return error(span, "Variable '" + callee + "' is not callable");
    }
    if (args.size() != 1) {
      return error(
          span,
          "Callable value '" + callee +
              "' must be invoked with exactly one pipeline/input argument");
    }
    auto input = analyzeExpr(*args[0].value, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&input)) {
      return *diagnostic;
    }
    Type input_type = std::get<Type>(std::move(input));
    Type result = tensor_any();
    if (symbol->type.callableReturn) {
      if (symbol->type.callableReturn->base == TypeBase::Tensor &&
          input_type.base == TypeBase::Tensor) {
        result = Type::tensor(input_type.tensorDtype,
                               symbol->type.callableReturn->tensorShapeExpr,
                               input_type.tensorRank);
      } else {
        result = *symbol->type.callableReturn;
      }
    }
    recordCall(expr.id, span, callee, SemanticCallTargetKind::CallableLocal, result,
               false);
    return result;
  }

  return error(span, "Undefined function or layer '" + callee + "'");
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::visitUnary(const Expr &operand, TokenType op,
                             const SourceSpan &span, const Program &program) {
  auto analyzed = analyzeExpr(operand, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
    return *diagnostic;
  }
  Type operand_type = std::get<Type>(std::move(analyzed));
  if (op == TokenType::Bang) {
    if (auto diagnostic =
            ensureConditionType(operand_type, span, "Unary '!'")) {
      return *diagnostic;
    }
    return Type::boolType();
  }
  if (op == TokenType::Minus && !(operand_type.base == TypeBase::Int ||
                                  operand_type.base == TypeBase::Float ||
                                  operand_type.base == TypeBase::Tensor ||
                                  operand_type.base == TypeBase::Unknown)) {
    return error(span, "Unary '-' expects int, float, or tensor operand");
  }
  return operand_type;
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::visitBinary(const Expr &expr, const Expr &lhs, const Expr &rhs, TokenType op,
                              const SourceSpan &span, const Program &program) {
  if (op == TokenType::Dot) {
    const auto *lhs_id = std::get_if<IdentifierExpr>(&lhs.kind);
    const auto *rhs_id = std::get_if<IdentifierExpr>(&rhs.kind);
    if (lhs_id && rhs_id) {
      auto config = configs_.find(lhs_id->name);
      if (config != configs_.end()) {
        auto field = config->second.find(rhs_id->name);
        if (field != config->second.end()) {
          recordConfigFieldAccess(expr.id, span, lhs_id->name, rhs_id->name,
                                  field->second);
          return field->second;
        }
        return error(span, "Unknown field '" + rhs_id->name + "' on config '" +
                               lhs_id->name + "'");
      }
    }
    return error(
        span,
        "Only config field access of the form configName.field is supported");
  }

  auto left = analyzeExpr(lhs, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&left)) {
    return *diagnostic;
  }
  auto right = analyzeExpr(rhs, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&right)) {
    return *diagnostic;
  }
  Type lhs_type = std::get<Type>(std::move(left));
  Type rhs_type = std::get<Type>(std::move(right));

  if (op == TokenType::Plus || op == TokenType::Minus ||
      op == TokenType::Star || op == TokenType::Slash ||
      op == TokenType::DoubleSlash) {
    if (lhs_type.base == TypeBase::Tensor ||
        rhs_type.base == TypeBase::Tensor) {
      return mergeTensorTypes(lhs_type, rhs_type);
    }
    if (lhs_type.base == TypeBase::Unknown ||
        rhs_type.base == TypeBase::Unknown) {
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
    if (!isCompatible(lhs_type, rhs_type)) {
      return error(span, "Comparison expects compatible operand types");
    }
    return Type::boolType();
  }
  if (op == TokenType::Lt || op == TokenType::Gt || op == TokenType::LtEq ||
      op == TokenType::GtEq) {
    if (!((lhs_type.base == TypeBase::Int || lhs_type.base == TypeBase::Float ||
           lhs_type.base == TypeBase::Unknown) &&
          (rhs_type.base == TypeBase::Int || rhs_type.base == TypeBase::Float ||
           rhs_type.base == TypeBase::Unknown))) {
      return error(span, "Ordered comparisons require int or float operands");
    }
    return Type::boolType();
  }
  if (op == TokenType::AmpAmp || op == TokenType::PipePipe) {
    if (auto diagnostic =
            ensureConditionType(lhs_type, lhs.span, "Logical operand")) {
      return *diagnostic;
    }
    if (auto diagnostic =
            ensureConditionType(rhs_type, rhs.span, "Logical operand")) {
      return *diagnostic;
    }
    return Type::boolType();
  }
  return error(span, "Unsupported binary operator in semantic analysis");
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::visitTernary(const Expr &thenExpr, const Expr &condition,
                               const Expr &elseExpr, const SourceSpan &span,
                               const Program &program) {
  auto condition_type = analyzeExpr(condition, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&condition_type)) {
    return *diagnostic;
  }
  if (auto diagnostic =
          ensureConditionType(std::get<Type>(condition_type), condition.span,
                              "Ternary condition")) {
    return *diagnostic;
  }
  auto left = analyzeExpr(thenExpr, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&left)) {
    return *diagnostic;
  }
  auto right = analyzeExpr(elseExpr, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&right)) {
    return *diagnostic;
  }
  Type then_type = std::get<Type>(std::move(left));
  Type else_type = std::get<Type>(std::move(right));
  if (!isCompatible(then_type, else_type) &&
      !isCompatible(else_type, then_type)) {
    return error(span, "Ternary branches must have compatible types");
  }
  return then_type.base == TypeBase::None ? else_type : then_type;
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::visitArrow(const Expr &source,
                             const std::vector<ExprPtr> &stages,
                             const Program &program) {
  auto current = analyzeExpr(source, program);
  if (const auto *diagnostic = std::get_if<Diagnostic>(&current)) {
    return *diagnostic;
  }
  Type current_type = std::get<Type>(std::move(current));
  for (const auto &stage : stages) {
    auto next = analyzeStage(*stage, current_type, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&next)) {
      return *diagnostic;
    }
    current_type = std::get<Type>(std::move(next));
    recordExprType(*stage, current_type);
  }
  return current_type;
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::analyzeStage(const Expr &expr, const Type &input_type,
                               const Program &program) {
  if (const auto *call = std::get_if<CallExpr>(&expr.kind)) {
    return analyzeArrowCall(expr, call->callee, call->args, expr.span, input_type,
                            program);
  }
  if (const auto *repeat = std::get_if<RepeatExpr>(&expr.kind)) {
    if (!std::holds_alternative<CallExpr>(repeat->stage->kind)) {
      return error(expr.span, "Repeated arrow stage must begin with a call");
    }
    auto count = analyzeExpr(*repeat->count, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&count)) {
      return *diagnostic;
    }
    if (std::get<Type>(count).base != TypeBase::Int) {
      return error(repeat->count->span,
                   "Repeated arrow stage count must have type int");
    }
    const auto &call = std::get<CallExpr>(repeat->stage->kind);
    return analyzeArrowCall(*repeat->stage, call.callee, call.args, repeat->stage->span,
                            input_type, program);
  }
  if (const auto *unary = std::get_if<UnaryExpr>(&expr.kind)) {
    Type operand_type;
    if (count_stage_sites(*unary->operand) > 0) {
      auto analyzed = analyzeStage(*unary->operand, input_type, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      operand_type = std::get<Type>(std::move(analyzed));
    } else {
      auto analyzed = analyzeExpr(*unary->operand, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      operand_type = std::get<Type>(std::move(analyzed));
    }
    if (unary->op == TokenType::Bang) {
      if (auto diagnostic =
              ensureConditionType(operand_type, expr.span, "Unary '!'")) {
        return *diagnostic;
      }
      return Type::boolType();
    }
    if (unary->op == TokenType::Minus &&
        !(operand_type.base == TypeBase::Int ||
          operand_type.base == TypeBase::Float ||
          operand_type.base == TypeBase::Tensor ||
          operand_type.base == TypeBase::Unknown)) {
      return error(expr.span,
                   "Unary '-' expects int, float, or tensor operand");
    }
    return operand_type;
  }
  if (const auto *binary = std::get_if<BinaryExpr>(&expr.kind)) {
    Type lhs_type;
    if (count_stage_sites(*binary->lhs) > 0) {
      auto analyzed = analyzeStage(*binary->lhs, input_type, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      lhs_type = std::get<Type>(std::move(analyzed));
    } else {
      auto analyzed = analyzeExpr(*binary->lhs, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      lhs_type = std::get<Type>(std::move(analyzed));
    }

    Type rhs_type;
    if (count_stage_sites(*binary->rhs) > 0) {
      auto analyzed = analyzeStage(*binary->rhs, input_type, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      rhs_type = std::get<Type>(std::move(analyzed));
    } else {
      auto analyzed = analyzeExpr(*binary->rhs, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      rhs_type = std::get<Type>(std::move(analyzed));
    }

    if (binary->op == TokenType::Plus || binary->op == TokenType::Minus ||
        binary->op == TokenType::Star || binary->op == TokenType::Slash ||
        binary->op == TokenType::DoubleSlash) {
      if (lhs_type.base == TypeBase::Tensor ||
          rhs_type.base == TypeBase::Tensor) {
        return mergeTensorTypes(lhs_type, rhs_type);
      }
      if (lhs_type.base == TypeBase::Unknown ||
          rhs_type.base == TypeBase::Unknown) {
        return Type::unknown();
      }
      if (lhs_type.base == TypeBase::Float ||
          rhs_type.base == TypeBase::Float) {
        return merge_scalar_float_types(lhs_type, rhs_type);
      }
      if (lhs_type.base == TypeBase::Int && rhs_type.base == TypeBase::Int) {
        return merge_scalar_int_types(lhs_type, rhs_type);
      }
      return error(expr.span,
                   "Arithmetic operation has incompatible operand types");
    }
    if (binary->op == TokenType::EqEq || binary->op == TokenType::Neq) {
      if (!isCompatible(lhs_type, rhs_type)) {
        return error(expr.span, "Comparison expects compatible operand types");
      }
      return Type::boolType();
    }
    if (binary->op == TokenType::Lt || binary->op == TokenType::Gt ||
        binary->op == TokenType::LtEq || binary->op == TokenType::GtEq) {
      if (!((lhs_type.base == TypeBase::Int ||
             lhs_type.base == TypeBase::Float ||
             lhs_type.base == TypeBase::Unknown) &&
            (rhs_type.base == TypeBase::Int ||
             rhs_type.base == TypeBase::Float ||
             rhs_type.base == TypeBase::Unknown))) {
        return error(expr.span,
                     "Ordered comparisons require int or float operands");
      }
      return Type::boolType();
    }
    if (binary->op == TokenType::AmpAmp || binary->op == TokenType::PipePipe) {
      if (auto diagnostic = ensureConditionType(lhs_type, binary->lhs->span,
                                                "Logical operand")) {
        return *diagnostic;
      }
      if (auto diagnostic = ensureConditionType(rhs_type, binary->rhs->span,
                                                "Logical operand")) {
        return *diagnostic;
      }
      return Type::boolType();
    }
    return analyzeExpr(expr, program);
  }
  if (const auto *ternary = std::get_if<TernaryExpr>(&expr.kind)) {
    auto condition =
        count_stage_sites(*ternary->condition) > 0
            ? analyzeStage(*ternary->condition, input_type, program)
            : analyzeExpr(*ternary->condition, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&condition)) {
      return *diagnostic;
    }
    if (auto diagnostic = ensureConditionType(std::get<Type>(condition),
                                              ternary->condition->span,
                                              "Ternary condition")) {
      return *diagnostic;
    }

    auto then_type = count_stage_sites(*ternary->thenExpr) > 0
                         ? analyzeStage(*ternary->thenExpr, input_type, program)
                         : analyzeExpr(*ternary->thenExpr, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&then_type)) {
      return *diagnostic;
    }
    auto else_type = count_stage_sites(*ternary->elseExpr) > 0
                         ? analyzeStage(*ternary->elseExpr, input_type, program)
                         : analyzeExpr(*ternary->elseExpr, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&else_type)) {
      return *diagnostic;
    }
    Type lhs = std::get<Type>(std::move(then_type));
    Type rhs = std::get<Type>(std::move(else_type));
    if (!isCompatible(lhs, rhs) && !isCompatible(rhs, lhs)) {
      return error(expr.span, "Ternary branches must have compatible types");
    }
    return lhs.base == TypeBase::None ? rhs : lhs;
  }
  if (const auto *tuple = std::get_if<TupleExpr>(&expr.kind)) {
    std::vector<Type> types;
    for (const auto &element : tuple->elements) {
      auto analyzed = count_stage_sites(*element) > 0
                          ? analyzeStage(*element, input_type, program)
                          : analyzeExpr(*element, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      types.push_back(std::get<Type>(std::move(analyzed)));
    }
    return Type::tuple(std::move(types));
  }
  if (const auto *list = std::get_if<ListExpr>(&expr.kind)) {
    std::vector<Type> types;
    for (const auto &element : list->elements) {
      auto analyzed = count_stage_sites(*element) > 0
                          ? analyzeStage(*element, input_type, program)
                          : analyzeExpr(*element, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
        return *diagnostic;
      }
      Type element_type = std::get<Type>(std::move(analyzed));
      const bool known =
          std::any_of(types.begin(), types.end(), [&](const Type &known_type) {
            return isCompatible(known_type, element_type);
          });
      if (!known) {
        types.push_back(std::move(element_type));
      }
    }
    return Type::list(std::move(types));
  }
  if (count_stage_sites(expr) == 0) {
    return analyzeExpr(expr, program);
  }
  return error(expr.span,
               "Unsupported compound arrow stage in semantic analysis");
}

std::variant<Type, Diagnostic> SemanticAnalyzer::analyzeArrowCall(
    const Expr &expr, const std::string &callee, const std::vector<CallArgument> &args,
    const SourceSpan &span, const Type &input_type, const Program &program) {
  std::vector<Type> argTypes{input_type};
  for (const auto &arg : args) {
    auto analyzed = analyzeExpr(*arg.value, program);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&analyzed)) {
      return *diagnostic;
    }
    argTypes.push_back(std::get<Type>(std::move(analyzed)));
  }

  auto function = functions_.find(callee);
  if (function != functions_.end()) {
    Type stage_type;
    if (function->second.returnType.base == TypeBase::Callable) {
      if (currentCallableKind_ == CallableKind::Function) {
        return error(span, "Arrow stage '" + callee +
                               "' resolves to a callable/layer-like stage and "
                               "cannot be used inside fn");
      }
      std::vector<Type> ctor_arg_types(argTypes.begin() + 1, argTypes.end());
      if (auto diagnostic = validateSignatureArity(
              function->second, ctor_arg_types.size(), span)) {
        return *diagnostic;
      }
      stage_type = infer_call_result_type(callee, function->second.returnType,
                                          ctor_arg_types);
    } else {
      if (auto diagnostic =
              validateSignatureArity(function->second, argTypes.size(), span)) {
        return *diagnostic;
      }
      stage_type =
          infer_call_result_type(callee, function->second.returnType, argTypes);
    }
    auto unwrapped = unwrapCallableStage(stage_type, input_type);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&unwrapped)) {
      return *diagnostic;
    }
    Type result = std::get<Type>(std::move(unwrapped));
    recordCall(expr.id, span, callee,
               isBuiltinOp(callee) ? SemanticCallTargetKind::BuiltinFunction
                                   : SemanticCallTargetKind::Function,
               result, true);
    return result;
  }

  auto layer = layers_.find(callee);
  if (layer != layers_.end()) {
    if (auto diagnostic = ensureCallAllowed(callee, true, span)) {
      return *diagnostic;
    }
    Type result = layer->second.returnType;
    recordCall(expr.id, span, callee, SemanticCallTargetKind::Layer, result, true);
    return result;
  }

  if (const Symbol *symbol = findVar(callee)) {
    if (!isCallable(symbol->type)) {
      return error(span, "Arrow stage '" + callee + "' is not callable");
    }
    if (currentCallableKind_ == CallableKind::Function) {
      return error(span,
                   "Arrow stage '" + callee +
                       "' is a callable/layer value and cannot be used inside "
                       "fn; fn arrow stages must stay function-only");
    }
    auto unwrapped = unwrapCallableStage(symbol->type, input_type);
    if (const auto *diagnostic = std::get_if<Diagnostic>(&unwrapped)) {
      return *diagnostic;
    }
    Type result = std::get<Type>(std::move(unwrapped));
    recordCall(expr.id, span, callee, SemanticCallTargetKind::CallableLocal, result,
               true);
    return result;
  }
  return error(span, "Arrow stage '" + callee + "' is not callable");
}

std::variant<Type, Diagnostic>
SemanticAnalyzer::unwrapCallableStage(const Type &stage_type,
                                      const Type &input_type) {
  if (stage_type.base != TypeBase::Callable) {
    return stage_type;
  }
  if (!stage_type.callableReturn) {
    return tensor_any();
  }
  if (stage_type.callableReturn->base == TypeBase::Tensor &&
      input_type.base == TypeBase::Tensor) {
    return Type::tensor(input_type.tensorDtype,
                        stage_type.callableReturn->tensorShapeExpr,
                        input_type.tensorRank);
  }
  return *stage_type.callableReturn;
}

std::optional<Diagnostic>
SemanticAnalyzer::validateTrainConfig(const Config &config,
                                      const Program &program) {
  std::optional<std::size_t> variantCount;
  for (const auto &field : config.fields) {
    if (field.name == "subtrain") {
      const SourceSpan span = field.init ? field.init->span : config.span;
      return error(span, "Field 'subtrain' is no longer supported; use "
                         "tuple-valued fields on the training config instead");
    }
    if (!is_train_config_field(field.name) || !field.init) {
      continue;
    }
    auto arity = tuple_arity(*field.init);
    if (!arity) {
      continue;
    }
    if (*arity == 0) {
      return error(field.init->span, "Training config field '" + field.name +
                                         "' cannot use an empty tuple");
    }
    if (variantCount && *variantCount != *arity) {
      return error(field.init->span,
                   "Tuple-valued training config fields must have the same "
                   "length; use scalar values to broadcast");
    }
    if (!variantCount) {
      variantCount = *arity;
    }
  }

  std::vector<std::string> model_symbols = collectModelSymbols(program);
  for (const auto &field : config.fields) {
    if (field.name != "objective" || !field.init) {
      continue;
    }

    std::vector<const Expr *> objective_exprs;
    if (const auto *tuple = std::get_if<TupleExpr>(&field.init->kind)) {
      for (const auto &element : tuple->elements) {
        objective_exprs.push_back(element);
      }
    } else {
      objective_exprs.push_back(field.init);
    }

    for (const Expr *expr : objective_exprs) {
      const auto *identifier = std::get_if<IdentifierExpr>(&expr->kind);
      if (identifier == nullptr) {
        return error(expr->span,
                     "Field 'objective' must reference a named tensor root, "
                     "not a string literal or arbitrary expression");
      }
      if (!model_symbols.empty() &&
          std::find(model_symbols.begin(), model_symbols.end(),
                    identifier->name) == model_symbols.end()) {
        return error(expr->span,
                     "Field 'objective' references unknown model root '" +
                         identifier->name + "'");
      }
    }
  }

  return std::nullopt;
}

std::vector<std::string>
SemanticAnalyzer::collectModelSymbols(const Program &program) const {
  const Layer *model = nullptr;
  for (const auto &layer : program.layers) {
    if (layer.name == "model") {
      model = &layer;
      break;
    }
  }
  if (model == nullptr) {
    return {};
  }

  std::set<std::string> symbols;
  for (const auto &arg : model->args) {
    symbols.insert(arg.name);
  }
  collect_stmt_symbols(model->body, symbols);
  return {symbols.begin(), symbols.end()};
}

std::optional<Diagnostic> SemanticAnalyzer::visitCallable(
    const std::vector<Arg> &args, const Type &returnType, const Stmt &body,
    const SourceSpan &span, const std::string &name, CallableKind kind,
    const char *label, const Program &program) {
  const auto previous_return = currentReturnType_;
  const auto previous_has_return = currentCallableHasReturn_;
  const auto previous_kind = currentCallableKind_;
  const auto previous_name = currentCallableName_;
  currentReturnType_ = returnType;
  currentCallableHasReturn_ = false;
  currentCallableKind_ = kind;
  currentCallableName_ = name;

  pushScope();
  for (const auto &arg : args) {
    if (arg.defaultValue) {
      auto default_type = analyzeExpr(*arg.defaultValue, program);
      if (const auto *diagnostic = std::get_if<Diagnostic>(&default_type)) {
        return *diagnostic;
      }
      if (!isCompatible(arg.type, std::get<Type>(default_type))) {
        return error(span, "Default value for argument '" + arg.name +
                               "' has incompatible type");
      }
    }
    if (auto diagnostic = declareVar(arg.name, arg.type,
                                     SemanticSymbolKind::Parameter, span)) {
      return diagnostic;
    }
  }
  if (auto diagnostic = analyzeStmt(body, program)) {
    return diagnostic;
  }
  popScope();

  if (returnType.base != TypeBase::None && !currentCallableHasReturn_) {
    return error(span, std::string(label) + " '" + name +
                           "' is missing a return statement");
  }

  currentReturnType_ = previous_return;
  currentCallableHasReturn_ = previous_has_return;
  currentCallableKind_ = previous_kind;
  currentCallableName_ = previous_name;
  return std::nullopt;
}

std::optional<Diagnostic> SemanticAnalyzer::declareVar(const std::string &name,
                                                       Type type,
                                                       SemanticSymbolKind kind,
                                                       const SourceSpan &span) {
  if (scopes_.empty()) {
    pushScope();
  }
  auto &scope = scopes_.back();
  if (scope.count(name) != 0) {
    return error(span,
                 "Variable '" + name + "' already declared in this scope");
  }
  Symbol symbol;
  symbol.type = type;
  symbol.isCallable = isCallable(type);
  if (type.callableReturn) {
    symbol.callableReturnType = *type.callableReturn;
  }
  symbol.kind = kind;
  scope[name] = symbol;
  recordSymbol(SemanticSymbol{name, kind, type, scopes_.size(),
                              currentCallableName_, span});
  return std::nullopt;
}

const Symbol *SemanticAnalyzer::findVar(const std::string &name) const {
  for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); ++scope) {
    auto found = scope->find(name);
    if (found != scope->end()) {
      return &found->second;
    }
  }
  return nullptr;
}
// i think here we are checking to type are compatible
bool SemanticAnalyzer::isCompatible(const Type &target,
                                    const Type &source) const {
  if (target.base == TypeBase::Unknown || source.base == TypeBase::Unknown) {
    return true;
  }
  if (target.base != source.base) {
    return false;
  }
  if ((target.base == TypeBase::Float || target.base == TypeBase::Int) &&
      target.scalarDtype && source.scalarDtype &&
      target.scalarDtype != source.scalarDtype) {
    return false;
  }
  if (target.base == TypeBase::Tensor) {
    if (target.tensorDtype && source.tensorDtype &&
        target.tensorDtype != source.tensorDtype) {
      return false;
    }
    if (target.tensorShapeExpr && source.tensorShapeExpr &&
        target.tensorShapeExpr != source.tensorShapeExpr) {
      return false;
    }
    if (target.tensorRank && source.tensorRank &&
        target.tensorRank != source.tensorRank) {
      return false;
    }
  }
  if (target.base == TypeBase::Tuple) {
    if (target.elements.size() != source.elements.size()) {
      return false;
    }
    for (std::size_t index = 0; index < target.elements.size(); ++index) {
      if (!isCompatible(target.elements[index], source.elements[index])) {
        return false;
      }
    }
  }
  if (target.base == TypeBase::List) {
    if (target.elements.empty() || source.elements.empty()) {
      return true;
    }
    return std::all_of(
        source.elements.begin(), source.elements.end(), [&](const Type &rhs) {
          return std::any_of(
              target.elements.begin(), target.elements.end(),
              [&](const Type &lhs) { return isCompatible(lhs, rhs); });
        });
  }
  if (target.base == TypeBase::Callable) {
    // here if we do 'c : callable' it doesn't have callableReturn, we need to
    // fix this if (!target.callableReturn || !source.callableReturn) {
    //     return !target.callableReturn && !source.callableReturn;
    // }
    // return isCompatible(*target.callableReturn, *source.callableReturn);
    return true;
  }
  return true;
}

Type SemanticAnalyzer::mergeTensorTypes(const Type &lhs,
                                        const Type &rhs) const {
  if (lhs.base != TypeBase::Tensor) {
    return rhs;
  }
  if (rhs.base != TypeBase::Tensor) {
    return lhs;
  }
  return Type::tensor(lhs.tensorDtype ? lhs.tensorDtype : rhs.tensorDtype,
                      lhs.tensorShapeExpr ? lhs.tensorShapeExpr
                                          : rhs.tensorShapeExpr,
                      lhs.tensorRank ? lhs.tensorRank : rhs.tensorRank);
}

// reviewed
std::optional<Diagnostic>
SemanticAnalyzer::validateDeclaredType(const Type &type,
                                       const SourceSpan &span) {
  // static for this vector persist between calls.
  static const std::vector<std::string> tensor_dtypes = {
      "float16", "float32", "float64", "bfloat16", "int16", "int32", "int64",
  };
  if (type.base == TypeBase::Int && type.scalarDtype &&
      !(*type.scalarDtype == "int16" || *type.scalarDtype == "int32" ||
        *type.scalarDtype == "int64")) {
    return error(span,
                 "Unsupported scalar integer type '" + *type.scalarDtype + "'");
  }
  if (type.base == TypeBase::Float && type.scalarDtype &&
      !(*type.scalarDtype == "float16" || *type.scalarDtype == "float32" ||
        *type.scalarDtype == "float64")) {
    return error(span,
                 "Unsupported scalar float type '" + *type.scalarDtype + "'");
  }
  if (type.base == TypeBase::Tensor && type.tensorDtype &&
      std::find(tensor_dtypes.begin(), tensor_dtypes.end(),
                *type.tensorDtype) == tensor_dtypes.end()) {
    return error(span, "Unsupported tensor dtype '" + *type.tensorDtype + "'");
  }
  // not sure here
  for (const auto &element : type.elements) {
    if (auto diagnostic = validateDeclaredType(element, span)) {
      return diagnostic;
    }
  }
  if (type.callableReturn) {
    return validateDeclaredType(*type.callableReturn, span);
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::validateSignatureArity(const Signature &signature,
                                         std::size_t actual_arity,
                                         const SourceSpan &span) {
  if (actual_arity < signature.minArity || actual_arity > signature.maxArity) {
    std::ostringstream expected;
    if (signature.minArity == signature.maxArity) {
      expected << signature.minArity;
    } else {
      expected << signature.minArity << " to " << signature.maxArity;
    }
    return error(span, "Call to '" + signature.name + "' expects " +
                           expected.str() + " argument(s), but got " +
                           std::to_string(actual_arity));
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::ensureConditionType(const Type &type, const SourceSpan &span,
                                      const std::string &context) {
  if (!(type.base == TypeBase::Bool || type.base == TypeBase::Unknown)) {
    return error(span, context + " must have type bool, but got " +
                           typeToString(type));
  }
  return std::nullopt;
}

std::optional<Diagnostic>
SemanticAnalyzer::ensureCallAllowed(const std::string &callee, bool is_layer,
                                    const SourceSpan &span) {
  if (is_layer && currentCallableKind_ == CallableKind::Function) {
    return error(span, "Function '" + callee +
                           "' is a layer and cannot be called from fn");
  }
  return std::nullopt;
}

Diagnostic SemanticAnalyzer::error(const SourceSpan &span,
                                   const std::string &message) {
  Diagnostic diagnostic =
      Diagnostic::error(DiagnosticCode::SemanticError, message)
          .withSourceSpan(span);
  lastDiagnostic_ = diagnostic;
  return diagnostic;
}
// this func pushing scope
void SemanticAnalyzer::pushScope() { scopes_.push_back({}); }

void SemanticAnalyzer::popScope() {
  if (!scopes_.empty()) {
    scopes_.pop_back();
  }
}
// reviewed
void SemanticAnalyzer::recordSymbol(SemanticSymbol symbol) {
  semanticInfo_.symbols.push_back(std::move(symbol));
}

void SemanticAnalyzer::recordExprType(const Expr &expr, Type type) {
  std::size_t idx = semanticInfo_.exprs.size();
  semanticInfo_.exprs.push_back(
      SemanticExprInfo{expr.id, expr.span, std::move(type), currentCallableName_});
  if (expr.id != 0) {
    semanticInfo_.expr_index[expr.id] = idx;
  }
}

void SemanticAnalyzer::recordIdentifier(std::uint32_t nodeId, const SourceSpan &span,
                                        const std::string &name,
                                        SemanticSymbolKind target, Type type) {
  std::size_t idx = semanticInfo_.identifiers.size();
  semanticInfo_.identifiers.push_back(SemanticIdentifierInfo{
      nodeId, span, name, target, std::move(type), currentCallableName_});
  if (nodeId != 0) {
    semanticInfo_.identifier_index[nodeId] = idx;
  }
}

void SemanticAnalyzer::recordAssignment(std::uint32_t nodeId, const SourceSpan &span,
                                        const std::string &name,
                                        SemanticSymbolKind target,
                                        Type targetType, Type valueType) {
  std::size_t idx = semanticInfo_.assignments.size();
  semanticInfo_.assignments.push_back(SemanticAssignmentInfo{
      nodeId,
      span,
      name,
      target,
      std::move(targetType),
      std::move(valueType),
      currentCallableName_,
  });
  if (nodeId != 0) {
    semanticInfo_.assignment_index[nodeId] = idx;
  }
}

void SemanticAnalyzer::recordConfigFieldAccess(std::uint32_t nodeId, const SourceSpan &span,
                                               const std::string &configName,
                                               const std::string &fieldName,
                                               Type fieldType) {
  std::size_t idx = semanticInfo_.configFieldAccesses.size();
  semanticInfo_.configFieldAccesses.push_back(SemanticConfigFieldAccessInfo{
      nodeId, span, configName, fieldName, std::move(fieldType), currentCallableName_});
  if (nodeId != 0) {
    semanticInfo_.config_access_index[nodeId] = idx;
  }
}

void SemanticAnalyzer::recordDeclaration(std::uint32_t nodeId, const SourceSpan &span,
                                         const std::string &name,
                                         SemanticSymbolKind kind,
                                         Type finalType) {
  std::size_t idx = semanticInfo_.declarations.size();
  semanticInfo_.declarations.push_back(SemanticDeclarationInfo{
      nodeId, span, name, kind, std::move(finalType), currentCallableName_});
  if (nodeId != 0) {
    semanticInfo_.declaration_index[nodeId] = idx;
  }
}

void SemanticAnalyzer::recordCall(std::uint32_t nodeId, const SourceSpan &span,
                                  const std::string &callee,
                                  SemanticCallTargetKind target,
                                  Type resultType, bool arrowStage) {
  std::size_t idx = semanticInfo_.calls.size();
  semanticInfo_.calls.push_back(
      SemanticCallInfo{nodeId, span, callee, target, std::move(resultType),
                       currentCallableName_, arrowStage});
  if (nodeId != 0) {
    semanticInfo_.call_index[nodeId] = idx;
  }
}

const char *semanticSymbolKindName(SemanticSymbolKind kind) {
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

const char *semanticCallTargetKindName(SemanticCallTargetKind kind) {
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

std::string semanticInfoSummary(const SemanticInfo &info,
                                const Program &program) {
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
  out << "configFieldAccesses=" << info.configFieldAccesses.size() << '\n';
  out << "declarations=" << info.declarations.size() << '\n';
  out << "calls=" << info.calls.size();
  return out.str();
}
