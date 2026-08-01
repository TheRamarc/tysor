#pragma once

#include "arena.h"
#include "diagnostic.h"
#include "frontend_ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic_analyzer.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

enum class MTypeKind {
  Unknown,
  Int,
  Float,
  Bool,
  Tensor,
  Tuple,
  List,
  Callable,
  None,
  Str,
};

struct MType {
  MTypeKind Kind = MTypeKind::None;
  std::vector<MType> elements;
  std::shared_ptr<MType> callableReturn;
  std::optional<std::string> scalarDtype;
  std::optional<std::string> tensorDtype;

  static MType unknown();
  static MType intType();
  static MType int16();
  static MType int32();
  static MType int64();
  static MType floatType();
  static MType float16();
  static MType float32();
  static MType float64();
  static MType boolType();
  static MType strType();
  static MType tensor(std::optional<std::string> dtype);
  static MType tuple(std::vector<MType> elements);
  static MType list(std::vector<MType> elements);
  static MType callable(MType returnType);
  static MType voidType();
  static MType none();
  static MType str();
};

struct MValue;
struct MTupleValue {
  std::vector<MValue> values;
};

struct MListValue {
  std::vector<MValue> values;
};

struct MValue {
  using Storage = std::variant<std::monostate, int, std::int64_t, float, double,
                               bool, std::string, MTupleValue, MListValue>;

  Storage value;

  static MValue none();
  static MValue intValue(std::int64_t value);
  static MValue floatValue(double value);
  static MValue boolValue(bool value);
  static MValue stringValue(std::string value);
  static MValue tupleValue(std::vector<MValue> values);
  static MValue listValue(std::vector<MValue> values);
};

enum class MBinaryOp {
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

struct MExpr;
using MExprPtr = MExpr *;

struct MCallArg {
  std::optional<std::string> name;
  MExprPtr value;
};

struct MConstantExpr {
  MValue value;
};

struct MVarExpr {
  std::string symbol;
};

struct MCallExpr {
  std::string callee;
  std::vector<MCallArg> args;
};

struct MApplyExpr {
  MExprPtr callee;
  std::vector<MCallArg> args;
};

struct MTupleExpr {
  std::vector<MExprPtr> elemetns;
};

struct MListExpr {
  std::vector<MExprPtr> elements;
};

struct MBinaryExpr {
  MBinaryOp op = MBinaryOp::Add;
  MExprPtr lhs;
  MExprPtr rhs;
};

struct MIfThenElseExpr {
  MExprPtr condition;
  MExprPtr thenExpr;
  MExprPtr elseExpr;
};

using MExprKind = std::variant<MConstantExpr, MVarExpr, MCallExpr, MTupleExpr,
                               MListExpr, MBinaryExpr, MIfThenElseExpr>;

struct MExpr {
  MType type;
  MExprKind kind;

  static MExprPtr constant(Arena &arena, MValue value, MType type);
  static MExprPtr var(Arena &arena, std::string symbol, MType type);
  static MExprPtr call(Arena &arena, std::string callee, MType type);
  static MExprPtr apply(Arena &arena, MExprPtr callee,
                        std::vector<FeCallExpr> args, MType type);
  static MExprPtr tuple(Arena &arena, std::vector<MExprPtr> elements,
                        MType type);
  static MExprPtr list(Arena &arena, std::vector<MExprPtr> elements,
                       MType type);
  static MExprPtr binary(Arena &arena, MBinaryOp op, MExprPtr lhs, MExprPtr rhs,
                         MType type);
  static MExprPtr ifThenElse(Arena &arena, MExprPtr condition,
                             MExprPtr thenExpr, MExprPtr elseExpr, MType type);
};

struct MStmt;

struct MVarDeclStmt {
  std::string name;
  MType type;
  MExprPtr value;
  bool hasValue = false;
};

struct MAssignStmt {
  std::string name;
  MExprPtr value;
};

struct MReturnStmt {
  MExprPtr value;
};

struct MExprStmt {
  MExprPtr value;
};

struct MElifBody {
  MExprPtr condition;
  std::vector<MStmt> body;
};

struct MIfStmt {
  MExprPtr condition;
  std::vector<MStmt> thenBody;
  std::vector<MElifBody> elifBodies;
  std::vector<MStmt> elseBody;
};

using MStmtKind =
    std::variant<MVarDeclStmt, MAssignStmt, MReturnStmt, MExprStmt, MIfStmt>;

struct MStmt {
  MStmtKind kind;
};

struct MFunction {
  std::string name;
  MType returnType;
  std::vector<std::pair<std::string, MType>> params;
  std::vector<std::pair<std::string, MType>> namedOutputs;
  std::vector<MStmt> body;
};

struct MLayer {
  std::string name;
  MType returnType;
  std::vector<std::pair<std::string, MType>> params;
  std::vector<std::pair<std::string, MType>> namedOutputs;
  std::vector<MStmt> body;
};

struct MConfig {
  std::string name;
  std::map<std::string, MValue> fields;
  std::vector<MValue> backends;
  std::vector<MValue> optimizers;
  std::vector<MValue> learningRates;
  std::vector<std::string> objectiveSymbols;
  std::vector<MValue> iterations;
  std::size_t variantCount = 1;
  std::map<std::string, MValue> extraProperties;
};

enum class ObjectiveSource {
  Param,
  Output,
  Local,
  Unknown,
};

struct MExecutionRun {
  std::string runName;
  std::string modelName;
  std::string trainName;
  std::optional<MValue> backend;
  std::optional<MValue> optimizer;
  std::optional<MValue> learningRate;
  std::optional<std::string> ojectiveSymbol;
  ObjectiveSource objectiveSource = ObjectiveSource::Unknown;
  MType objectiveType;
  std::optional<MValue> iteration;
};
// this different config running together
struct MExecutionPlan {
  std::string modelEntry;
  std::vector<MExecutionRun> runs;
};

struct LoweredModule {
  // Arena containing all Frontend IR nodes
  std::unique_ptr<Arena> arena;
  // To store global configurations for use by the runtime or model and To
  // define how to train the models in the module.
  std::vector<MConfig> configs;
  // To hold all executable function definitions.
  std::vector<MFunction> functions;
  // TO hold all stateful layer definitions.
  std::vector<MLayer> layers;
  // To handle module-level variable declarations or assignments.
  std::vector<MStmt> globals;
  // To provide a recipe for executing the program.
  std::optional<MExecutionPlan> executionplan;
};

using MirResult = std::variant<LoweredModule, Diagnostic>;

class MirLowerer {
public:
  MirLowerer(const Program &program, const SemanticInfo &semantic_info);
  MirResult lower();
  [[nodiscard]] std::optional<Diagnostic> takeLastDiagnostic();

private:
  struct EvaluatedConfigField {
    // To detect circular dependencies in config field evaluation.
    bool inProgress = false;
    // To memoize evalutd config fields.
    bool computed = false;
    // To hold a runtieme or compile-time evaluated constant.
    MValue value = MValue::none();
  }

  const Program &program_;
  const SemanticInfo &semanticInfo_;
  std::map<std::string, const Config *> configDefs_;
  std::unique_ptr<Arena> arena_;
  std::map<std::string, std::map<std::string, EvaluatedConfigField>>
      configFieldCache_;
  std::map<std::string, MType> globalSymbols_;
  std::map<std::string, MType> currentSymbols_;
  std::map<std::string, MType> currentOwner_;
  std::optional<Diagnositc> lastDignostic_;

  std::variant<MExprPtr, Diagnositc> lowerExpr(const Expr &expr);
  std::variant<MExprPtr, Diagnositc> lowerArrow(const Expr &expr);
  std::variant<MExprPtr, Diagnositc> lowerArrowStageExpr(const Expr &expr,
                                                         MExprPtr current);
  std::variant<MExprPtr, Diagnositc>
  lowerSemanticArrowCallStage(const SemanticCallInfo &call,
                              const std::string &callee,
                              const std::vector<CallArgument> &args, MExprPtr);
};
