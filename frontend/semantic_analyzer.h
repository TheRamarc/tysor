#pragma once

#include "diagnostic.h"
#include "parser.h"
#include "source.h"

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

/**
 * @brief Identifies the nature of a resolved symbol.
 *
 * Symbol kind is kept in diagnostics and SemanticInfo so lowering can tell a
 * local variable from a layer/function/config access without re-resolving
 * names.
 */
enum class SemanticSymbolKind {
  BuiltinFunction,
  Function,
  Layer,
  Config,
  ConfigField,
  Global,
  Parameter,
  Local,
};

/**
 * @brief Identifies the kind of callable being invoked.
 */
enum class SemanticCallTargetKind {
  BuiltinFunction,
  Function,
  Layer,
  CallableLocal,
};

/**
 * @brief Information about one declared object (a variable, parameter,
 * function, etc.) in the scope.
 */
struct Symbol {
  /**
   * @brief The resolved type of the symbol.
   *
   * Why it exists: Used to enforce type safety when the symbol is referenced.
   * What it tracks: The Type assigned during declaration.
   * What mutates/updates it: Populated upon declaration in the scope and
   * remains immutable.
   */
  Type type;

  /**
   * @brief Whether the symbol can be invoked like a function.
   *
   * Why it exists: Prevents attempting to call non-callable symbols.
   * What it tracks: A boolean flag indicating callability.
   * What mutates/updates it: Determined and set during symbol resolution.
   */
  bool isCallable = false;

  /**
   * @brief If callable, the expected return type.
   *
   * Why it exists: Helps infer the type of a call expression involving this
   * symbol. What it tracks: An optional return Type. What mutates/updates it:
   * Populated for callables during declaration; empty otherwise.
   */
  std::optional<Type> callableReturnType;

  /**
   * @brief The origin kind of the symbol.
   *
   * Why it exists: Tells downstream passes (e.g. lowering) how to load/store
   * this value. What it tracks: Whether it is a local, parameter, global,
   * layer, etc. What mutates/updates it: Set based on the context of its
   * declaration (e.g., function arg vs variable).
   */
  SemanticSymbolKind kind = SemanticSymbolKind::Local;
};

/**
 * @brief Information about how to call a function or layer.
 */
struct Signature {
  /**
   * @brief The name of the function or layer.
   *
   * Why it exists: Identifies the callable for resolution.
   * What it tracks: The identifier string.
   * What mutates/updates it: Set once when parsing the definition.
   */
  std::string name;

  /**
   * @brief The return type of the callable.
   *
   * Why it exists: Validates what calling this returns.
   * What it tracks: The Type returned on invocation.
   * What mutates/updates it: Set during definition analysis.
   */
  Type returnType;

  /**
   * @brief Expected types for the arguments.
   *
   * Why it exists: Validates the arguments passed at call sites.
   * What it tracks: A list of parameter Types.
   * What mutates/updates it: Set during definition analysis.
   */
  std::vector<Type> argTypes;

  /**
   * @brief Minimum number of arguments required.
   *
   * Why it exists: Validates call arity (considering default args).
   * What it tracks: The minimum valid argument count.
   * What mutates/updates it: Computed from the argument list during definition
   * analysis.
   */
  std::size_t minArity = 0;

  /**
   * @brief Maximum number of arguments allowed.
   *
   * Why it exists: Prevents over-passing arguments.
   * What it tracks: The total number of parameters.
   * What mutates/updates it: Computed from the argument list during definition
   * analysis.
   */
  std::size_t maxArity = 0;
};

/**
 * @brief Extended symbol information retained for downstream passes (e.g.
 * lowering).
 */
struct SemanticSymbol {
  /**
   * @brief The name of the symbol.
   *
   * Why it exists: Serves as the primary identifier in the semantic info
   * side-table. What it tracks: The string name of the symbol. What
   * mutates/updates it: Initialized during scope registration.
   */
  std::string name;

  /**
   * @brief The origin kind of the symbol.
   *
   * Why it exists: Tells downstream passes how to implement the symbol.
   * What it tracks: The symbol's category (local, param, etc.).
   * What mutates/updates it: Initialized during scope registration.
   */
  SemanticSymbolKind kind = SemanticSymbolKind::Local;

  /**
   * @brief The resolved type of the symbol.
   *
   * Why it exists: Needed by IR generation to allocate correctly sized storage.
   * What it tracks: The validated Type.
   * What mutates/updates it: Initialized during scope registration.
   */
  Type type;

  /**
   * @brief How deeply nested the scope is.
   *
   * Why it exists: Resolves shadowing conflicts by picking the deepest symbol.
   * What it tracks: Integer depth (0 = global).
   * What mutates/updates it: Set based on current scope nesting level during
   * analysis.
   */
  std::size_t scopeDepth = 0;

  /**
   * @brief Name of the function/layer that owns this symbol (if any).
   *
   * Why it exists: Distinguishes local symbols with identical names in
   * different functions. What it tracks: The parent function/layer name. What
   * mutates/updates it: Set based on the active callable context during
   * analysis.
   */
  std::optional<std::string> owner;

  /**
   * @brief Source location of the symbol's declaration.
   *
   * Why it exists: Useful for debugging and advanced error reporting later in
   * lowering. What it tracks: The SourceSpan where the symbol was defined. What
   * mutates/updates it: Initialized from the declaration AST node.
   */
  std::optional<SourceSpan> span;
};

/**
 * @brief Type information deduced for a specific expression span.
 */
struct SemanticExprInfo {
  std::uint32_t nodeId = 0;
  SourceSpan span{};
  Type type;
  std::optional<std::string> owner;
};

struct SemanticIdentifierInfo {
  std::uint32_t nodeId = 0;
  SourceSpan span{};
  std::string name;
  SemanticSymbolKind target = SemanticSymbolKind::Local;
  Type type;
  std::optional<std::string> owner;
};

struct SemanticAssignmentInfo {
  std::uint32_t nodeId = 0;
  SourceSpan span{};
  std::string targetName;
  SemanticSymbolKind targetKind = SemanticSymbolKind::Local;
  Type targetType;
  Type valueType;
  std::optional<std::string> owner;
};

struct SemanticConfigFieldAccessInfo {
  std::uint32_t nodeId = 0;
  SourceSpan span{};
  std::string configName;
  std::string fieldName;
  Type fieldType;
  std::optional<std::string> owner;
};

struct SemanticDeclarationInfo {
  std::uint32_t nodeId = 0;
  SourceSpan span{};
  std::string name;
  SemanticSymbolKind kind = SemanticSymbolKind::Local;
  Type finalType;
  std::optional<std::string> owner;
};

struct SemanticCallInfo {
  std::uint32_t nodeId = 0;
  SourceSpan span{};
  std::string callee;
  SemanticCallTargetKind target = SemanticCallTargetKind::Function;
  Type resultType;
  std::optional<std::string> owner;
  bool arrowStage = false;
};

/**
 * @brief Side-table produced by semantic analysis.
 *
 * Frontend lowering consults this by span/owner instead of repeating scope
 * lookup and type inference.
 */
struct SemanticInfo {
  /**
   * @brief Extended symbol information retained for downstream passes.
   *
   * Why it exists: Acts as a symbol table for lowering.
   * What it tracks: All variables, functions, and configs defined.
   * What mutates/updates it: Populated during the collection phase.
   */
  std::vector<SemanticSymbol> symbols;

  /**
   * @brief Type information deduced for specific expression spans.
   *
   * Why it exists: Caches type checking results.
   * What it tracks: Expression type mappings.
   * What mutates/updates it: Appended during AST traversal.
   */
  std::vector<SemanticExprInfo> exprs;

  /**
   * @brief Resolution information for identifier expressions.
   *
   * Why it exists: Caches variable lookups.
   * What it tracks: Identifier resolutions.
   * What mutates/updates it: Appended during AST traversal.
   */
  std::vector<SemanticIdentifierInfo> identifiers;

  /**
   * @brief Validation info for assignment statements.
   *
   * Why it exists: Caches store target information.
   * What it tracks: Assignment resolutions.
   * What mutates/updates it: Appended during AST traversal.
   */
  std::vector<SemanticAssignmentInfo> assignments;

  /**
   * @brief Validation info for config field accesses.
   *
   * Why it exists: Caches config resolution.
   * What it tracks: Configuration member access types.
   * What mutates/updates it: Appended during AST traversal.
   */
  std::vector<SemanticConfigFieldAccessInfo> configFieldAccesses;

  /**
   * @brief Type and scope info for variable declarations.
   *
   * Why it exists: Caches inferred declaration types.
   * What it tracks: Let binding types.
   * What mutates/updates it: Appended during AST traversal.
   */
  std::vector<SemanticDeclarationInfo> declarations;

  /**
   * @brief Target resolution for function/layer calls.
   *
   * Why it exists: Caches dispatch targets.
   * What it tracks: Call resolutions.
   * What mutates/updates it: Appended during AST traversal.
   */
  std::vector<SemanticCallInfo> calls;

  std::unordered_map<std::uint32_t, std::size_t> expr_index;
  std::unordered_map<std::uint32_t, std::size_t> identifier_index;
  std::unordered_map<std::uint32_t, std::size_t> assignment_index;
  std::unordered_map<std::uint32_t, std::size_t> declaration_index;
  std::unordered_map<std::uint32_t, std::size_t> call_index;
  std::unordered_map<std::uint32_t, std::size_t> config_access_index;

  const SemanticExprInfo* findExpr(std::uint32_t nodeId, const SourceSpan& span) const {
    if (nodeId != 0) {
      auto it = expr_index.find(nodeId);
      if (it != expr_index.end() && it->second < exprs.size()) return &exprs[it->second];
    }
    for (const auto& item : exprs) {
      if (item.span.line == span.line && item.span.column == span.column) return &item;
    }
    return nullptr;
  }

  const SemanticIdentifierInfo* findIdentifier(std::uint32_t nodeId, const SourceSpan& span) const {
    if (nodeId != 0) {
      auto it = identifier_index.find(nodeId);
      if (it != identifier_index.end() && it->second < identifiers.size()) return &identifiers[it->second];
    }
    for (const auto& item : identifiers) {
      if (item.span.line == span.line && item.span.column == span.column) return &item;
    }
    return nullptr;
  }

  const SemanticCallInfo* findCall(std::uint32_t nodeId, const SourceSpan& span) const {
    if (nodeId != 0) {
      auto it = call_index.find(nodeId);
      if (it != call_index.end() && it->second < calls.size()) return &calls[it->second];
    }
    for (const auto& item : calls) {
      if (item.span.line == span.line && item.span.column == span.column) return &item;
    }
    return nullptr;
  }

  const SemanticDeclarationInfo* findDeclaration(std::uint32_t nodeId, const SourceSpan& span) const {
    if (nodeId != 0) {
      auto it = declaration_index.find(nodeId);
      if (it != declaration_index.end() && it->second < declarations.size()) return &declarations[it->second];
    }
    for (const auto& item : declarations) {
      if (item.span.line == span.line && item.span.column == span.column) return &item;
    }
    return nullptr;
  }

  const SemanticAssignmentInfo* findAssignment(std::uint32_t nodeId, const SourceSpan& span) const {
    if (nodeId != 0) {
      auto it = assignment_index.find(nodeId);
      if (it != assignment_index.end() && it->second < assignments.size()) return &assignments[it->second];
    }
    for (const auto& item : assignments) {
      if (item.span.line == span.line && item.span.column == span.column) return &item;
    }
    return nullptr;
  }
};

using SemanticResult = std::variant<SemanticInfo, Diagnostic>;

/**
 * @brief Performs type checking and scope resolution over the AST.
 *
 * Performs declaration collection first, then statement/expression validation.
 * That supports forward calls to functions/layers while still enforcing local
 * scope, call arity, and tensor type compatibility.
 */
class SemanticAnalyzer {
public:
  SemanticAnalyzer();

  /**
   * @brief Analyzes the program and produces SemanticInfo or a Diagnostic
   * error.
   */
  SemanticResult analyzeWithInfo(const Program &program);

  /**
   * @brief Consumes and returns the last recorded diagnostic.
   */
  [[nodiscard]] std::optional<Diagnostic> takeLastDiagnostic();

private:
  enum class CallableKind {
    None,
    Function,
    Layer,
  };

  /**
   * @brief Stack of lexical scopes for variable resolution.
   *
   * Why it exists: Provides an ordered lookup mechanism (from most nested to
   * global) to resolve identifier references (`IdentifierExpr`) to their typed
   * `Symbol` counterparts and handle variable shadowing in nested block scopes
   * (e.g. `IfStmt`, `ForStmt`). What it tracks: A dynamic stack where each
   * element represents a lexical block. The `std::map` maps string identifiers
   * directly to their AST `Symbol` definitions (type, callability, etc.). Index
   * 0 always represents the global module scope. What mutates/updates it:
   * `pushScope()` appends a new empty map when entering a block. `popScope()`
   * destroys the deepest block on exit. `declareVar()` inserts new symbols into
   * `scopes_.back()`.
   */
  std::vector<std::map<std::string, Symbol>> scopes_;

  /**
   * @brief Global registry of user-defined functions.
   *
   * Why it exists: Enables forward and mutually recursive calls across the
   * module by establishing all callable targets before descending into function
   * bodies. What it tracks: A flat dictionary mapping top-level `Function` AST
   * node names to their parsed `Signature` (return type, argument types, and
   * arity bounds). What mutates/updates it: Exclusively populated by
   * `collectFunctions()` during the initial top-level scan of the `Program` AST
   * prior to statement analysis.
   */
  std::map<std::string, Signature> functions_;

  /**
   * @brief Global registry of user-defined layers.
   *
   * Why it exists: Serves as the authoritative source for layer validation,
   * distinguishing neural network modules (which hold state) from pure
   * mathematical functions during instantiation and invocation. What it tracks:
   * Maps `Layer` AST node identifiers to their explicit `Signature`. Unlike
   * functions, layers enforce different runtime constraints (like parameter
   * initialization). What mutates/updates it: Exclusively populated by
   * `collectLayers()` during the initial top-level scan of the `Program` AST.
   */
  std::map<std::string, Signature> layers_;

  /**
   * @brief Global registry of configuration blocks.
   *
   * Why it exists: Allows strict type-checking of hyperparameter and
   * configuration field accesses (e.g., `train.learningRate`) without requiring
   * dynamic dictionaries at runtime. What it tracks: A nested map. The outer
   * map keys on the config block name (e.g., "train"). The inner map associates
   * specific field names with their deduced static `Type`. What mutates/updates
   * it: Exclusively populated by `collectConfigs()` during the initial scan of
   * `Config` AST nodes.
   */
  std::map<std::string, std::map<std::string, Type>> configs_;

  /**
   * @brief The deduced type of the last evaluated expression.
   *
   * Why it exists: Acts as a side-channel to propagate type information up the
   * AST, especially useful for implicit returns or arrow pipeline (`->`) stage
   * inference where a parent node needs the exact type of its child. What it
   * tracks: The `Type` object computed for the most recently traversed `Expr`
   * node. What mutates/updates it: Overwritten at the end of every
   * `analyzeExpr()` call and `analyzeStmt()` expression evaluation. Reset to
   * `Type::noneType()` when exiting scopes that don't produce values.
   */
  Type lastExprType_ = Type::noneType();

  /**
   * @brief The expected return type of the active callable.
   *
   * Why it exists: Provides the reference type to validate all `ReturnStmt`
   * nodes against. Without this, nested `return` statements would not know if
   * they violate the function's strict signature. What it tracks: An
   * `std::optional<Type>`. When `std::nullopt`, it indicates we are in the
   * global scope (where returns are invalid). Otherwise, it holds the
   * `returnType` of the enclosing function/layer. What mutates/updates it: Set
   * by `visitCallable()` when descending into a function/layer body, and
   * restored to its previous state (usually `nullopt`) upon returning up the
   * call stack.
   */
  std::optional<Type> currentReturnType_;

  /**
   * @brief Flag indicating if the current callable has returned a value on all
   * paths.
   *
   * Why it exists: Powers the control-flow validation that ensures functions
   * declaring a non-Void return type don't accidentally fall off the end of the
   * body without returning a value. What it tracks: A strict boolean flag
   * indicating if a valid `ReturnStmt` was unconditionally hit in the current
   * execution block. What mutates/updates it: Set to `false` at the start of
   * `visitCallable()`. Flipped to `true` inside `analyzeStmt()` when
   * encountering a valid `ReturnStmt`.
   */
  bool currentCallableHasReturn_ = false;

  /**
   * @brief The kind of the callable currently being analyzed.
   *
   * Why it exists: Enforces contextual rules. For example, layers can
   * instantiate other layers, but pure functions cannot hold state or
   * instantiate layers. This tracks the context for those semantic checks. What
   * it tracks: A `CallableKind` enum (`None` for global scope, `Function` for
   * pure logic, `Layer` for stateful modules). What mutates/updates it:
   * Overwritten by `visitCallable()` when entering a function/layer, and
   * reverted to `CallableKind::None` upon exit.
   */
  CallableKind currentCallableKind_ = CallableKind::None;

  /**
   * @brief The name of the callable currently being analyzed.
   *
   * Why it exists: Enables the IR lowerer to disambiguate identically named
   * local variables that exist in different functions by associating an 'owner'
   * namespace to every resolved expression. What it tracks: The raw string name
   * of the enclosing function or layer. What mutates/updates it: Set by
   * `visitCallable()` and used to stamp the `owner` field in the `record_*`
   * side-table helpers.
   */
  std::optional<std::string> currentCallableName_;

  /**
   * @brief The accumulated semantic information side-table.
   *
   * Why it exists: Completely decouples AST validation from IR generation. By
   * building this side-table, the `FrontendLowerer` can perform O(1) lookups
   * for type and target info without needing to understand scope rules or
   * perform type inference. What it tracks: The `SemanticInfo` structure,
   * containing parallel arrays (`exprs`, `identifiers`, `calls`, etc.) mapping
   * `SourceSpan` coordinates to resolved backend facts. What mutates/updates
   * it: Appended to sequentially by the `record_*` helpers (e.g.,
   * `recordExprType`, `recordCall`) throughout the entire AST traversal
   * process.
   */
  SemanticInfo semanticInfo_;

  /**
   * @brief The last encountered error diagnostic.
   *
   * Why it exists: Caches semantic compilation errors (like type mismatches or
   * undefined variables) to halt compilation gracefully and report
   * human-readable errors back to the CLI. What it tracks: An
   * `std::optional<Diagnostic>` object holding the error message, severity, and
   * exact `SourceSpan` where the semantic violation occurred. What
   * mutates/updates it: Assigned by the `error()` helper method. Once set, the
   * analyzer immediately aborts further traversal and bubbles the error up.
   */
  std::optional<Diagnostic> lastDiagnostic_;

  void beginAnalysis();
  void registerBuiltins();
  std::optional<Diagnostic> analyze(const Program &program);
  std::optional<Diagnostic> collectConfigs(const Program &program);
  std::optional<Diagnostic> collectLayers(const Program &program);
  std::optional<Diagnostic> collectFunctions(const Program &program);
  std::optional<Diagnostic> analyzeStmt(const Stmt &stmt,
                                        const Program &program);
  std::optional<Diagnostic> analyzeReturnStmt(const ReturnStmt &value,
                                              const SourceSpan &span,
                                              const Program &program);
  std::optional<Diagnostic> analyzeExprStmt(const ExprStmt &value,
                                            const Program &program);
  std::optional<Diagnostic> analyzeVarDeclStmt(const VarDecl &value,
                                               std::uint32_t nodeId,
                                               const SourceSpan &span,
                                               const Program &program);
  std::optional<Diagnostic> analyzeAssignStmt(const AssignStmt &value,
                                              std::uint32_t nodeId,
                                              const SourceSpan &span,
                                              const Program &program);
  std::optional<Diagnostic> analyzeScopeStmt(const ScopeStmt &value,
                                             const Program &program);
  std::optional<Diagnostic> analyzeIfStmt(const IfStmt &value,
                                          const Program &program);
  std::variant<Type, Diagnostic> analyzeExpr(const Expr &expr,
                                             const Program &program);
  std::variant<Type, Diagnostic> visitIdentifier(const Expr &expr, const std::string &name,
                                                 const SourceSpan &span);
  std::variant<Type, Diagnostic>
  visitCall(const Expr &expr, const std::string &callee, const std::vector<CallArgument> &args,
            const SourceSpan &span, const Program &program);
  std::variant<Type, Diagnostic> visitUnary(const Expr &operand, TokenType op,
                                            const SourceSpan &span,
                                            const Program &program);
  std::variant<Type, Diagnostic> visitBinary(const Expr &expr, const Expr &lhs, const Expr &rhs,
                                             TokenType op,
                                             const SourceSpan &span,
                                             const Program &program);
  std::variant<Type, Diagnostic> visitTernary(const Expr &thenExpr,
                                              const Expr &condition,
                                              const Expr &elseExpr,
                                              const SourceSpan &span,
                                              const Program &program);
  std::variant<Type, Diagnostic> visitArrow(const Expr &source,
                                            const std::vector<ExprPtr> &stages,
                                            const Program &program);
  std::variant<Type, Diagnostic> analyzeStage(const Expr &expr,
                                              const Type &input_type,
                                              const Program &program);
  std::variant<Type, Diagnostic> analyzeArrowCall(
      const Expr &expr, const std::string &callee, const std::vector<CallArgument> &args,
      const SourceSpan &span, const Type &input_type, const Program &program);
  std::variant<Type, Diagnostic> unwrapCallableStage(const Type &stage_type,
                                                     const Type &input_type);
  std::optional<Diagnostic> validateTrainConfig(const Config &config,
                                                const Program &program);
  std::vector<std::string> collectModelSymbols(const Program &program) const;
  std::optional<Diagnostic>
  visitCallable(const std::vector<Arg> &args, const Type &returnType,
                const Stmt &body, const SourceSpan &span,
                const std::string &name, CallableKind kind, const char *label,
                const Program &program);
  std::optional<Diagnostic> declareVar(const std::string &name, Type type,
                                       SemanticSymbolKind kind,
                                       const SourceSpan &span);
  const Symbol *findVar(const std::string &name) const;
  bool isCompatible(const Type &target, const Type &source) const;
  Type mergeTensorTypes(const Type &lhs, const Type &rhs) const;
  std::optional<Diagnostic> validateDeclaredType(const Type &type,
                                                 const SourceSpan &span);
  std::optional<Diagnostic> validateSignatureArity(const Signature &signature,
                                                   std::size_t actual_arity,
                                                   const SourceSpan &span);
  std::optional<Diagnostic> ensureConditionType(const Type &type,
                                                const SourceSpan &span,
                                                const std::string &context);
  std::optional<Diagnostic> ensureCallAllowed(const std::string &callee,
                                              bool is_layer,
                                              const SourceSpan &span);
  Diagnostic error(const SourceSpan &span, const std::string &message);
  void pushScope();
  void popScope();
  void recordSymbol(SemanticSymbol symbol);
  void recordExprType(const Expr &expr, Type type);
  void recordIdentifier(std::uint32_t nodeId, const SourceSpan &span, const std::string &name,
                        SemanticSymbolKind target, Type type);
  void recordAssignment(std::uint32_t nodeId, const SourceSpan &span, const std::string &name,
                        SemanticSymbolKind target, Type targetType,
                        Type valueType);
  void recordConfigFieldAccess(std::uint32_t nodeId, const SourceSpan &span,
                               const std::string &configName,
                               const std::string &fieldName, Type fieldType);
  void recordDeclaration(std::uint32_t nodeId, const SourceSpan &span, const std::string &name,
                         SemanticSymbolKind kind, Type finalType);
  void recordCall(std::uint32_t nodeId, const SourceSpan &span, const std::string &callee,
                  SemanticCallTargetKind target, Type resultType,
                  bool arrowStage);
};

const char *semanticSymbolKindName(SemanticSymbolKind kind);
const char *semanticCallTargetKindName(SemanticCallTargetKind kind);
std::string semanticInfoSummary(const SemanticInfo &info,
                                const Program &program);
