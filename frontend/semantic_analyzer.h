#pragma once

#include "diagnostic.h"
#include "parser.h"
#include "source.h"

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

/**
 * @brief Identifies the nature of a resolved symbol.
 * 
 * Symbol kind is kept in diagnostics and SemanticInfo so lowering can tell a
 * local variable from a layer/function/config access without re-resolving names.
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
 * @brief Information about one declared object (a variable, parameter, function, etc.) in the scope.
 */
struct Symbol {
    /**
     * @brief The resolved type of the symbol.
     * 
     * Why it exists: Used to enforce type safety when the symbol is referenced.
     * What it tracks: The Type assigned during declaration.
     * What mutates/updates it: Populated upon declaration in the scope and remains immutable.
     */
    Type type;

    /**
     * @brief Whether the symbol can be invoked like a function.
     * 
     * Why it exists: Prevents attempting to call non-callable symbols.
     * What it tracks: A boolean flag indicating callability.
     * What mutates/updates it: Determined and set during symbol resolution.
     */
    bool is_callable = false;

    /**
     * @brief If callable, the expected return type.
     * 
     * Why it exists: Helps infer the type of a call expression involving this symbol.
     * What it tracks: An optional return Type.
     * What mutates/updates it: Populated for callables during declaration; empty otherwise.
     */
    std::optional<Type> callable_return_type;

    /**
     * @brief The origin kind of the symbol.
     * 
     * Why it exists: Tells downstream passes (e.g. lowering) how to load/store this value.
     * What it tracks: Whether it is a local, parameter, global, layer, etc.
     * What mutates/updates it: Set based on the context of its declaration (e.g., function arg vs variable).
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
    Type return_type;

    /**
     * @brief Expected types for the arguments.
     * 
     * Why it exists: Validates the arguments passed at call sites.
     * What it tracks: A list of parameter Types.
     * What mutates/updates it: Set during definition analysis.
     */
    std::vector<Type> arg_types;

    /**
     * @brief Minimum number of arguments required.
     * 
     * Why it exists: Validates call arity (considering default args).
     * What it tracks: The minimum valid argument count.
     * What mutates/updates it: Computed from the argument list during definition analysis.
     */
    std::size_t min_arity = 0;

    /**
     * @brief Maximum number of arguments allowed.
     * 
     * Why it exists: Prevents over-passing arguments.
     * What it tracks: The total number of parameters.
     * What mutates/updates it: Computed from the argument list during definition analysis.
     */
    std::size_t max_arity = 0;
};

/**
 * @brief Extended symbol information retained for downstream passes (e.g. lowering).
 */
struct SemanticSymbol {
    /**
     * @brief The name of the symbol.
     * 
     * Why it exists: Serves as the primary identifier in the semantic info side-table.
     * What it tracks: The string name of the symbol.
     * What mutates/updates it: Initialized during scope registration.
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
     * What mutates/updates it: Set based on current scope nesting level during analysis.
     */
    std::size_t scope_depth = 0;

    /**
     * @brief Name of the function/layer that owns this symbol (if any).
     * 
     * Why it exists: Distinguishes local symbols with identical names in different functions.
     * What it tracks: The parent function/layer name.
     * What mutates/updates it: Set based on the active callable context during analysis.
     */
    std::optional<std::string> owner;

    /**
     * @brief Source location of the symbol's declaration.
     * 
     * Why it exists: Useful for debugging and advanced error reporting later in lowering.
     * What it tracks: The SourceSpan where the symbol was defined.
     * What mutates/updates it: Initialized from the declaration AST node.
     */
    std::optional<SourceSpan> span;
};

/**
 * @brief Type information deduced for a specific expression span.
 */
struct SemanticExprInfo {
    /**
     * @brief The source location of the expression.
     * 
     * Why it exists: Used as a key to look up type information for a specific AST node.
     * What it tracks: The SourceSpan where the expression is located.
     * What mutates/updates it: Populated during AST traversal.
     */
    SourceSpan span{};

    /**
     * @brief The inferred type of the expression.
     * 
     * Why it exists: Prevents re-running type inference during lowering.
     * What it tracks: The deduced Type of the expression.
     * What mutates/updates it: Populated during type analysis.
     */
    Type type;

    /**
     * @brief The function/layer owning this expression.
     * 
     * Why it exists: Disambiguates identical spans across multiple parsed files (if concatenated) or inline contexts.
     * What it tracks: The name of the surrounding callable context.
     * What mutates/updates it: Populated during AST traversal.
     */
    std::optional<std::string> owner;
};

/**
 * @brief Resolution information for an identifier expression.
 */
struct SemanticIdentifierInfo {
    /**
     * @brief The source location of the identifier.
     * 
     * Why it exists: Maps the resolution info back to the specific IdentifierExpr AST node.
     * What it tracks: The SourceSpan of the identifier token.
     * What mutates/updates it: Populated during AST traversal.
     */
    SourceSpan span{};

    /**
     * @brief The string name of the identifier.
     * 
     * Why it exists: Provides the name to look up in the current scope.
     * What it tracks: The identifier text.
     * What mutates/updates it: Populated from the AST node.
     */
    std::string name;

    /**
     * @brief The resolved target kind (local, global, layer, etc.).
     * 
     * Why it exists: Informs the backend whether to emit a local load, function pointer, or parameter access.
     * What it tracks: The SemanticSymbolKind determined from scope lookup.
     * What mutates/updates it: Populated after resolving the name.
     */
    SemanticSymbolKind target = SemanticSymbolKind::Local;

    /**
     * @brief The resolved type of the identifier.
     * 
     * Why it exists: Caches type information so lowering doesn't re-resolve the name.
     * What it tracks: The Type from the corresponding Symbol.
     * What mutates/updates it: Populated during type analysis.
     */
    Type type;

    /**
     * @brief The function/layer owning this identifier.
     * 
     * Why it exists: Contextual disambiguation.
     * What it tracks: The enclosing callable name.
     * What mutates/updates it: Populated during AST traversal.
     */
    std::optional<std::string> owner;
};

/**
 * @brief Validation info for an assignment statement.
 */
struct SemanticAssignmentInfo {
    /**
     * @brief The source location of the assignment.
     * 
     * Why it exists: Used to map this info to the corresponding AssignStmt AST node.
     * What it tracks: The SourceSpan of the assignment.
     * What mutates/updates it: Populated during AST traversal.
     */
    SourceSpan span{};

    /**
     * @brief The name of the variable being assigned to.
     * 
     * Why it exists: Identifies the target of the store operation.
     * What it tracks: The LHS identifier text.
     * What mutates/updates it: Populated from the AST.
     */
    std::string target_name;

    /**
     * @brief The resolved target kind of the variable.
     * 
     * Why it exists: Helps determine the correct store instruction (e.g., local vs global).
     * What it tracks: The SemanticSymbolKind of the target.
     * What mutates/updates it: Resolved by looking up the target name in the current scope.
     */
    SemanticSymbolKind target_kind = SemanticSymbolKind::Local;

    /**
     * @brief The expected type of the target variable.
     * 
     * Why it exists: Needed to validate that the assigned value matches.
     * What it tracks: The target's registered Type.
     * What mutates/updates it: Found during scope lookup.
     */
    Type target_type;

    /**
     * @brief The inferred type of the assigned value.
     * 
     * Why it exists: Ensures the RHS evaluates to an appropriate type.
     * What it tracks: The Type of the expression being assigned.
     * What mutates/updates it: Populated after analyzing the RHS expression.
     */
    Type value_type;

    /**
     * @brief The function/layer owning this assignment.
     * 
     * Why it exists: Contextual disambiguation.
     * What it tracks: The enclosing callable name.
     * What mutates/updates it: Populated during AST traversal.
     */
    std::optional<std::string> owner;
};

/**
 * @brief Validation info for a config field access (e.g. `cfg.learning_rate`).
 */
struct SemanticConfigFieldAccessInfo {
    /**
     * @brief The source location of the field access.
     * 
     * Why it exists: Maps info back to the member access AST node.
     * What it tracks: The SourceSpan of the access expression.
     * What mutates/updates it: Populated during AST traversal.
     */
    SourceSpan span{};

    /**
     * @brief The name of the config object.
     * 
     * Why it exists: Identifies which configuration is being read.
     * What it tracks: The text of the base identifier.
     * What mutates/updates it: Extracted from the LHS of the member access.
     */
    std::string config_name;

    /**
     * @brief The name of the specific field within the config.
     * 
     * Why it exists: Identifies the precise value being accessed.
     * What it tracks: The text of the field identifier.
     * What mutates/updates it: Extracted from the RHS of the member access.
     */
    std::string field_name;

    /**
     * @brief The resolved type of the config field.
     * 
     * Why it exists: Needed for downstream operations consuming the field value.
     * What it tracks: The type registered for this field in the configuration.
     * What mutates/updates it: Looked up from the config definition during analysis.
     */
    Type field_type;

    /**
     * @brief The function/layer owning this access.
     * 
     * Why it exists: Contextual disambiguation.
     * What it tracks: The enclosing callable name.
     * What mutates/updates it: Populated during AST traversal.
     */
    std::optional<std::string> owner;
};

/**
 * @brief Type and scope info for a variable declaration.
 */
struct SemanticDeclarationInfo {
    /**
     * @brief The source location of the declaration.
     * 
     * Why it exists: Used to map info back to the VarDecl AST node.
     * What it tracks: The SourceSpan of the declaration.
     * What mutates/updates it: Populated during AST traversal.
     */
    SourceSpan span{};

    /**
     * @brief The name of the declared variable.
     * 
     * Why it exists: Associates the declared name for IR generation.
     * What it tracks: The identifier text.
     * What mutates/updates it: Extracted from the AST.
     */
    std::string name;

    /**
     * @brief The symbol kind (e.g. local).
     * 
     * Why it exists: Specifies how the variable should be allocated.
     * What it tracks: The SemanticSymbolKind.
     * What mutates/updates it: Set based on declaration context.
     */
    SemanticSymbolKind kind = SemanticSymbolKind::Local;

    /**
     * @brief The deduced or explicitly stated type of the variable.
     * 
     * Why it exists: Used for allocating appropriately sized storage.
     * What it tracks: The final resolved Type.
     * What mutates/updates it: Inferred from the initializer or type annotation.
     */
    Type final_type;

    /**
     * @brief The function/layer owning this declaration.
     * 
     * Why it exists: Contextual disambiguation.
     * What it tracks: The enclosing callable name.
     * What mutates/updates it: Populated during AST traversal.
     */
    std::optional<std::string> owner;
};

/**
 * @brief Target resolution for a function/layer call.
 */
struct SemanticCallInfo {
    /**
     * @brief The source location of the call expression.
     * 
     * Why it exists: Used to map info back to the CallExpr AST node.
     * What it tracks: The SourceSpan of the call.
     * What mutates/updates it: Populated during AST traversal.
     */
    SourceSpan span{};

    /**
     * @brief The name of the callable being invoked.
     * 
     * Why it exists: Indicates which function/layer to dispatch to.
     * What it tracks: The callee identifier text.
     * What mutates/updates it: Extracted from the AST.
     */
    std::string callee;

    /**
     * @brief The resolved kind of the callable target.
     * 
     * Why it exists: Differentiates between built-in ops, user functions, and layers.
     * What it tracks: The SemanticCallTargetKind enum.
     * What mutates/updates it: Looked up from the symbol/function/layer tables.
     */
    SemanticCallTargetKind target = SemanticCallTargetKind::Function;

    /**
     * @brief The inferred return type of the call.
     * 
     * Why it exists: Required for further expression type-checking.
     * What it tracks: The Type returned by the callable.
     * What mutates/updates it: Looked up from the callable's signature.
     */
    Type result_type;

    /**
     * @brief The function/layer owning this call.
     * 
     * Why it exists: Contextual disambiguation.
     * What it tracks: The enclosing callable name.
     * What mutates/updates it: Populated during AST traversal.
     */
    std::optional<std::string> owner;

    /**
     * @brief Whether this call is part of an Arrow pipeline expression (`->`).
     * 
     * Why it exists: Arrow pipeline stages might need special lowering semantics.
     * What it tracks: A boolean flag indicating pipeline involvement.
     * What mutates/updates it: Set when traversing ArrowExpr branches.
     */
    bool arrow_stage = false;
};

/**
 * @brief Side-table produced by semantic analysis.
 * 
 * Frontend lowering consults this by span/owner instead of repeating scope lookup
 * and type inference.
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
    std::vector<SemanticConfigFieldAccessInfo> config_field_accesses;

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
     * @brief Analyzes the program and produces SemanticInfo or a Diagnostic error.
     */
    SemanticResult analyze_with_info(const Program& program);
    
    /**
     * @brief Consumes and returns the last recorded diagnostic.
     */
    [[nodiscard]] std::optional<Diagnostic> take_last_diagnostic();

private:
    enum class CallableKind {
        None,
        Function,
        Layer,
    };

    // Stack of lexical scopes. The current function/layer owner is recorded in
    // SemanticInfo so identical spans in different owners remain distinguishable.
    std::vector<std::map<std::string, Symbol>> scopes_;
    std::map<std::string, Signature> functions_;
    std::map<std::string, Signature> layers_;
    std::map<std::string, std::map<std::string, Type>> configs_;
    Type last_expr_type_ = Type::None_type();
    std::optional<Type> current_return_type_;
    bool current_callable_has_return_ = false;
    CallableKind current_callable_kind_ = CallableKind::None;
    std::optional<std::string> current_callable_name_;
    SemanticInfo semantic_info_;
    std::optional<Diagnostic> last_diagnostic_;

    void begin_analysis();
    void register_builtins();
    std::optional<Diagnostic> analyze(const Program& program);
    std::optional<Diagnostic> collect_configs(const Program& program);
    std::optional<Diagnostic> collect_layers(const Program& program);
    std::optional<Diagnostic> collect_functions(const Program& program);
    std::optional<Diagnostic> analyze_stmt(const Stmt& stmt, const Program& program);
    std::variant<Type, Diagnostic> analyze_expr(const Expr& expr, const Program& program);
    std::variant<Type, Diagnostic> visit_identifier(const std::string& name, const SourceSpan& span);
    std::variant<Type, Diagnostic> visit_call(
        const std::string& callee,
        const std::vector<CallArgument>& args,
        const SourceSpan& span,
        const Program& program
    );
    std::variant<Type, Diagnostic> visit_unary(
        const Expr& operand,
        TokenType op,
        const SourceSpan& span,
        const Program& program
    );
    std::variant<Type, Diagnostic> visit_binary(
        const Expr& lhs,
        const Expr& rhs,
        TokenType op,
        const SourceSpan& span,
        const Program& program
    );
    std::variant<Type, Diagnostic> visit_ternary(
        const Expr& then_expr,
        const Expr& condition,
        const Expr& else_expr,
        const SourceSpan& span,
        const Program& program
    );
    std::variant<Type, Diagnostic> visit_arrow(
        const Expr& source,
        const std::vector<ExprPtr>& stages,
        const Program& program
    );
    std::variant<Type, Diagnostic> analyze_stage(
        const Expr& expr,
        const Type& input_type,
        const Program& program
    );
    std::variant<Type, Diagnostic> analyze_arrow_call(
        const std::string& callee,
        const std::vector<CallArgument>& args,
        const SourceSpan& span,
        const Type& input_type,
        const Program& program
    );
    std::variant<Type, Diagnostic> unwrap_callable_stage(const Type& stage_type, const Type& input_type);
    std::optional<Diagnostic> validate_train_config(const Config& config, const Program& program);
    std::vector<std::string> collect_model_symbols(const Program& program) const;
    std::optional<Diagnostic> visit_callable(
        const std::vector<Arg>& args,
        const Type& return_type,
        const Stmt& body,
        const SourceSpan& span,
        const std::string& name,
        CallableKind kind,
        const char* label,
        const Program& program
    );
    std::optional<Diagnostic> declare_var(
        const std::string& name,
        Type type,
        SemanticSymbolKind kind,
        const SourceSpan& span
    );
    const Symbol* find_var(const std::string& name) const;
    bool is_compatible(const Type& target, const Type& source) const;
    Type merge_tensor_types(const Type& lhs, const Type& rhs) const;
    std::optional<Diagnostic> validate_declared_type(const Type& type, const SourceSpan& span);
    std::optional<Diagnostic> validate_signature_arity(
        const Signature& signature,
        std::size_t actual_arity,
        const SourceSpan& span
    );
    std::optional<Diagnostic> ensure_condition_type(
        const Type& type,
        const SourceSpan& span,
        const std::string& context
    );
    std::optional<Diagnostic> ensure_call_allowed(
        const std::string& callee,
        bool is_layer,
        const SourceSpan& span
    );
    Diagnostic error(const SourceSpan& span, const std::string& message);
    void push_scope();
    void pop_scope();
    void record_symbol(SemanticSymbol symbol);
    void record_expr_type(const Expr& expr, Type type);
    void record_identifier(
        const SourceSpan& span,
        const std::string& name,
        SemanticSymbolKind target,
        Type type
    );
    void record_assignment(
        const SourceSpan& span,
        const std::string& name,
        SemanticSymbolKind target,
        Type target_type,
        Type value_type
    );
    void record_config_field_access(
        const SourceSpan& span,
        const std::string& config_name,
        const std::string& field_name,
        Type field_type
    );
    void record_declaration(
        const SourceSpan& span,
        const std::string& name,
        SemanticSymbolKind kind,
        Type final_type
    );
    void record_call(
        const SourceSpan& span,
        const std::string& callee,
        SemanticCallTargetKind target,
        Type result_type,
        bool arrow_stage
    );
};

const char* semantic_symbol_kind_name(SemanticSymbolKind kind);
const char* semantic_call_target_kind_name(SemanticCallTargetKind kind);
std::string semantic_info_summary(const SemanticInfo& info, const Program& program);
