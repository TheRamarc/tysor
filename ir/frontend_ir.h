#pragma once

#include "diagnostic.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

enum class FeTypeKind {
    Unknown,
    Int,
    Float,
    Bool,
    Tensor,
    Tuple,
    List,
    Callable,
    Void,
    None,
};

struct FeType {
    FeTypeKind kind = FeTypeKind::Void;
    std::vector<FeType> elements;
    std::shared_ptr<FeType> callable_return;
    std::optional<std::string> scalar_dtype;
    std::optional<std::string> tensor_dtype;
    std::optional<std::string> tensor_shape_expr;
    std::optional<std::size_t> tensor_rank;

    static FeType unknown();
    static FeType int_type();
    static FeType int16();
    static FeType int32();
    static FeType int64();
    static FeType float_type();
    static FeType float16();
    static FeType float32();
    static FeType float64();
    static FeType bool_type();
    static FeType tensor(
        std::optional<std::string> dtype,
        std::optional<std::string> shape_expr,
        std::optional<std::size_t> rank
    );
    static FeType tuple(std::vector<FeType> elements);
    static FeType list(std::vector<FeType> elements);
    static FeType callable(FeType return_type);
    static FeType void_type();
    static FeType none();
};

struct FeValue;
struct FeTupleValue {
    std::vector<FeValue> values;
};
struct FeListValue {
    std::vector<FeValue> values;
};

struct FeValue {
    using Storage = std::variant<std::monostate, std::int64_t, double, bool, std::string, FeTupleValue, FeListValue>;
    Storage value;

    static FeValue none();
    static FeValue int_value(std::int64_t value);
    static FeValue float_value(double value);
    static FeValue bool_value(bool value);
    static FeValue string_value(std::string value);
    static FeValue tuple_value(std::vector<FeValue> values);
    static FeValue list_value(std::vector<FeValue> values);
};

enum class FeBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Not,
};

struct FeExpr;
using FeExprPtr = std::shared_ptr<FeExpr>;

struct FeCallArg {
    std::optional<std::string> name;
    FeExprPtr value;
};

struct FeConstantExpr {
    FeValue value;
};

struct FeVarExpr {
    std::string symbol;
};

struct FeCallExpr {
    std::string callee;
    std::vector<FeCallArg> args;
};

struct FeLayerCtorExpr {
    std::string callee;
    std::vector<FeCallArg> args;
};

struct FeApplyExpr {
    FeExprPtr callee;
    std::vector<FeCallArg> args;
};

struct FeTupleExpr {
    std::vector<FeExprPtr> elements;
};

struct FeListExpr {
    std::vector<FeExprPtr> elements;
};

struct FeBinaryExpr {
    FeBinaryOp op = FeBinaryOp::Add;
    FeExprPtr lhs;
    FeExprPtr rhs;
};

struct FeIfThenElseExpr {
    FeExprPtr condition;
    FeExprPtr then_expr;
    FeExprPtr else_expr;
};

using FeExprKind = std::variant<
    FeConstantExpr,
    FeVarExpr,
    FeCallExpr,
    FeLayerCtorExpr,
    FeApplyExpr,
    FeTupleExpr,
    FeListExpr,
    FeBinaryExpr,
    FeIfThenElseExpr
>;

struct FeExpr {
    FeType type;
    FeExprKind kind;

    static FeExprPtr constant(FeValue value, FeType type);
    static FeExprPtr var(std::string symbol, FeType type);
    static FeExprPtr call(std::string callee, std::vector<FeCallArg> args, FeType type);
    static FeExprPtr layer_ctor(std::string callee, std::vector<FeCallArg> args, FeType type);
    static FeExprPtr apply(FeExprPtr callee, std::vector<FeCallArg> args, FeType type);
    static FeExprPtr tuple(std::vector<FeExprPtr> elements, FeType type);
    static FeExprPtr list(std::vector<FeExprPtr> elements, FeType type);
    static FeExprPtr binary(FeBinaryOp op, FeExprPtr lhs, FeExprPtr rhs, FeType type);
    static FeExprPtr if_then_else(FeExprPtr condition, FeExprPtr then_expr, FeExprPtr else_expr, FeType type);
};

struct FeStmt;

struct FeVarDeclStmt {
    std::string name;
    FeType type;
    FeExprPtr value;
    bool has_value = false;
    bool mutable_symbol = false;
};

struct FeAssignStmt {
    std::string name;
    FeExprPtr value;
};

struct FeReturnStmt {
    FeExprPtr value;
};

struct FeExprStmt {
    FeExprPtr value;
};

struct FeElifBody {
    FeExprPtr condition;
    std::vector<FeStmt> body;
};

struct FeIfStmt {
    FeExprPtr condition;
    std::vector<FeStmt> then_body;
    std::vector<FeElifBody> elif_bodies;
    std::vector<FeStmt> else_body;
};

using FeStmtKind = std::variant<FeVarDeclStmt, FeAssignStmt, FeReturnStmt, FeExprStmt, FeIfStmt>;

struct FeStmt {
    FeStmtKind kind;
};

struct FeFunction {
    std::string name;
    bool is_layer = false;
    FeType return_type;
    std::vector<std::pair<std::string, FeType>> params;
    std::vector<std::pair<std::string, FeType>> named_outputs;
    std::vector<FeStmt> body;
};

struct FeConfig {
    std::string name;
    std::map<std::string, FeValue> fields;
};

struct FeTrain {
    std::string name;
    std::vector<FeValue> backends;
    std::vector<FeValue> optimizers;
    std::vector<FeValue> learning_rates;
    std::vector<std::string> objective_symbols;
    std::vector<FeValue> iterations;
    std::size_t variant_count = 1;
    std::map<std::string, FeValue> extra_properties;
};

enum class ObjectiveSource {
    Param,
    Output,
    Local,
    Unknown,
};

struct FeExecutionRun {
    std::string run_name;
    std::string model_name;
    std::string train_name;
    std::optional<FeValue> backend;
    std::optional<FeValue> optimizer;
    std::optional<FeValue> learning_rate;
    std::optional<std::string> objective_symbol;
    ObjectiveSource objective_source = ObjectiveSource::Unknown;
    FeType objective_type;
    std::optional<FeValue> iteration;
};

struct FeExecutionPlan {
    std::string model_entry;
    std::vector<FeExecutionRun> runs;
};

struct LoweredModule {
    std::vector<FeConfig> configs;
    std::vector<FeTrain> trains;
    std::vector<FeFunction> functions;
    std::vector<FeStmt> globals;
    std::optional<FeExecutionPlan> execution_plan;
};

using FrontendResult = std::variant<LoweredModule, Diagnostic>;

class FrontendLowerer {
public:
    FrontendLowerer(const Program& program, const SemanticInfo& semantic_info);

    FrontendResult lower();
    [[nodiscard]] std::optional<Diagnostic> take_last_diagnostic();

private:
    struct EvaluatedConfigField {
        bool in_progress = false;
        bool computed = false;
        FeValue value = FeValue::none();
    };

    const Program& program_;
    const SemanticInfo& semantic_info_;
    std::map<std::string, const Config*> config_defs_;
    std::map<std::string, std::map<std::string, EvaluatedConfigField>> config_field_cache_;
    std::map<std::string, FeType> global_symbols_;
    std::map<std::string, FeType> current_symbols_;
    std::optional<std::string> current_owner_;
    std::optional<Diagnostic> last_diagnostic_;

    std::variant<FeExprPtr, Diagnostic> lower_expr(const Expr& expr);
    std::variant<FeExprPtr, Diagnostic> lower_arrow_expr(const Expr& expr);
    std::variant<FeExprPtr, Diagnostic> lower_arrow_stage_expr(const Expr& expr, FeExprPtr current);
    std::variant<FeExprPtr, Diagnostic> lower_arrow_call_stage(
        const std::string& callee,
        const std::vector<CallArgument>& args,
        const SourceSpan& span,
        FeExprPtr current
    );
    std::variant<FeExprPtr, Diagnostic> lower_semantic_arrow_call_stage(
        const SemanticCallInfo& call,
        const std::string& callee,
        const std::vector<CallArgument>& args,
        FeExprPtr current
    );
    std::variant<std::vector<FeStmt>, Diagnostic> lower_scope(const Stmt& stmt);
    std::variant<FeStmt, Diagnostic> lower_stmt(const Stmt& stmt);
    std::variant<FeFunction, Diagnostic> lower_function(const Function& function);
    std::variant<FeFunction, Diagnostic> lower_layer(const Layer& layer);
    std::variant<FeConfig, Diagnostic> lower_config(const Config& config);
    std::variant<FeTrain, Diagnostic> lower_train_config(const Config& config);
    std::variant<FeExecutionPlan, Diagnostic> build_execution_plan(const LoweredModule& module);
    bool resolve_objective_stmt(const FeStmt& stmt, const std::string& objective_symbol, FeExecutionRun& run) const;
    std::variant<FeValue, Diagnostic> eval_constant_expr(const Expr& expr);
    std::variant<std::vector<FeValue>, Diagnostic> eval_constant_field_values(const Expr& expr);
    std::variant<FeValue, Diagnostic> eval_config_field(
        const std::string& config_name,
        const std::string& field_name,
        const SourceSpan& span
    );
    std::variant<FeValue, Diagnostic> eval_binary(TokenType op, const FeValue& lhs, const FeValue& rhs, const SourceSpan& span);
    std::variant<FeValue, Diagnostic> eval_unary(TokenType op, const FeValue& operand, const SourceSpan& span);
    std::optional<SemanticCallInfo> semantic_call_for_expr(const Expr& expr, const std::string& callee) const;
    std::optional<SemanticCallInfo> semantic_call_for_arrow_stage(const std::string& callee, const SourceSpan& span) const;
    std::optional<SemanticIdentifierInfo> semantic_identifier_for_expr(const Expr& expr, const std::string& name) const;
    std::optional<SemanticAssignmentInfo> semantic_assignment_for_stmt(const Stmt& stmt, const std::string& name) const;
    std::optional<SemanticConfigFieldAccessInfo> semantic_config_field_access_for_expr(const Expr& expr) const;
    std::optional<SemanticDeclarationInfo> semantic_declaration_for_stmt(const Stmt& stmt, const std::string& name) const;
    std::optional<FeType> semantic_type_for_expr(const Expr& expr) const;
    std::variant<FeType, Diagnostic> required_semantic_type_for_expr(const Expr& expr, const std::string& context);
    void bind_symbol(const std::string& name, FeType type);
    std::optional<FeType> find_symbol(const std::string& name) const;
    Diagnostic error(const std::string& message);
    Diagnostic error_span(const SourceSpan& span, const std::string& message);
};

FeType lower_type(const Type& type);
std::variant<FeBinaryOp, Diagnostic> lower_binary_op(TokenType token, const SourceSpan& span);
std::string lowered_module_summary(const LoweredModule& module);
std::string frontend_ir_to_string(const LoweredModule& module);
