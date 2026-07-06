#pragma once

#include "diagnostic.h"
#include "parser.h"
#include "source.h"

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Symbol kind is kept in diagnostics and SemanticInfo so lowering can tell a
// local variable from a layer/function/config access without re-resolving names.
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

enum class SemanticCallTargetKind {
    BuiltinFunction,
    Function,
    Layer,
    CallableLocal,
};

// Symbol = information about one declared object (a variable, parameter, function, etc.).
struct Symbol {
    Type type;
    bool mutable_symbol = false;
    bool is_callable = false;
    std::optional<Type> callable_return_type;
    SemanticSymbolKind kind = SemanticSymbolKind::Local;
};

// Signature = information about how to call a function.
struct Signature {
    std::string name;
    Type return_type;
    std::vector<Type> arg_types;
    std::size_t min_arity = 0;
    std::size_t max_arity = 0;
};

struct SemanticSymbol {
    std::string name;
    SemanticSymbolKind kind = SemanticSymbolKind::Local;
    Type type;
    bool mutable_symbol = false;
    std::size_t scope_depth = 0;
    std::optional<std::string> owner;
    std::optional<SourceSpan> span;
};

struct SemanticExprInfo {
    SourceSpan span{};
    Type type;
    std::optional<std::string> owner;
};

struct SemanticIdentifierInfo {
    SourceSpan span{};
    std::string name;
    SemanticSymbolKind target = SemanticSymbolKind::Local;
    Type type;
    bool mutable_symbol = false;
    std::optional<std::string> owner;
};

struct SemanticAssignmentInfo {
    SourceSpan span{};
    std::string target_name;
    SemanticSymbolKind target_kind = SemanticSymbolKind::Local;
    Type target_type;
    Type value_type;
    bool mutable_symbol = false;
    std::optional<std::string> owner;
};

struct SemanticConfigFieldAccessInfo {
    SourceSpan span{};
    std::string config_name;
    std::string field_name;
    Type field_type;
    std::optional<std::string> owner;
};

struct SemanticDeclarationInfo {
    SourceSpan span{};
    std::string name;
    SemanticSymbolKind kind = SemanticSymbolKind::Local;
    Type final_type;
    bool mutable_symbol = false;
    std::optional<std::string> owner;
};

struct SemanticCallInfo {
    SourceSpan span{};
    std::string callee;
    SemanticCallTargetKind target = SemanticCallTargetKind::Function;
    Type result_type;
    std::optional<std::string> owner;
    bool arrow_stage = false;
};

// Side-table produced by semantic analysis. Frontend lowering consults this by
// span/owner instead of repeating scope lookup and type inference.
struct SemanticInfo {
    std::vector<SemanticSymbol> symbols;
    std::vector<SemanticExprInfo> exprs;
    std::vector<SemanticIdentifierInfo> identifiers;
    std::vector<SemanticAssignmentInfo> assignments;
    std::vector<SemanticConfigFieldAccessInfo> config_field_accesses;
    std::vector<SemanticDeclarationInfo> declarations;
    std::vector<SemanticCallInfo> calls;
};

using SemanticResult = std::variant<SemanticInfo, Diagnostic>;

// Performs declaration collection first, then statement/expression validation.
// That supports forward calls to functions/layers while still enforcing local
// scope, mutability, call arity, and tensor type compatibility.
class SemanticAnalyzer {
public:
    SemanticAnalyzer();

    SemanticResult analyze_with_info(const Program& program);
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
        bool mutable_symbol,
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
        Type type,
        bool mutable_symbol
    );
    void record_assignment(
        const SourceSpan& span,
        const std::string& name,
        SemanticSymbolKind target,
        Type target_type,
        Type value_type,
        bool mutable_symbol
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
        Type final_type,
        bool mutable_symbol
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
