enum class Ops {
  // ** 1 -- defines/special **

  // This is a RANGE for GPU dimensions, similar to symbolic shapes but not
  // exactly
  SPECIAL,

  // BUFFER allocates global/local/register storage depending on its addrspace
  BUFFER,

  // ** 2 -- non op uops **

  // uops that aren't rendered
  NOOP,
  REWRITE_ERROR,

  // CALL is a kernel invocation; calls with RETURNED inputs are value-producing
  // (and gradient-able), the rest are opaque
  PARAM,
  CALL,

  // renderer
  // LINEAR is a list of UOps, SOURCE has a str arg that's human readable,
  // BINARY has bytes arg that's compiled
  PROGRAM,
  LINEAR,
  SOURCE,
  BINARY,

  // AFTER passes src[0] through and promises in the toposort that any consumers
  // of the AFTER run after src[1:]
  // GROUP is a NOOP that just merges things together
  SINK,
  AFTER,
  GROUP,

  // vector creation / item selection
  STACK,

  // hcq specific
  GETADDR,

  // ** 3 -- load/store **

  // INDEX is a BinaryOp similar to ADD, but it operates on pointers
  INDEX,
  SHRINK,

  // load/store before math
  LOAD,
  STORE,

  // ** 4 -- math **

  // tensor core math op, not elementwise
  WMMA,

  // UnaryOps
  CAST,
  BITCAST,
  EXP2,
  LOG2,
  SIN,
  SQRT,
  RECIPROCAL,
  NEG,
  TRUNC,

  // BinaryOps
  ADD,
  MUL,
  SHL,
  SHR,
  CDIV,
  MAX,
  CMOD,
  CMPLT,
  CMPNE,
  CMPEQ,
  XOR,
  OR,
  AND,
  THREEFRY,
  SUB,
  FDIV,
  POW,
  FLOORDIV,
  FLOORMOD,

  // TernaryOps
  WHERE,
  MULACC,

  // ** 5 -- control flow / consts / custom **

  // control flow ops
  BARRIER,
  RANGE,
  IF,
  END,
  ENDIF,

  // const.
  CONST,

  // CUSTOM/CUSTOMI are used to output strings into codegen.
  // The I makes the string inline.
  CUSTOM,
  CUSTOMI,

  // INS is a machine instruction
  INS,

  // ** 6 -- ops that don't exist in programs **

  // ops that adjust the behavior of the scheduler
  CONTIGUOUS,
  CONTIGUOUS_BACKWARD,
  DETACH,

  // buffer ops
  STAGE,
  COPY,
  MSELECT,
  MSTACK,
  CUSTOM_FUNCTION,

  // the core 6 movement ops!
  // these only exist in the tensor graph
  RESHAPE,
  PERMUTE,
  EXPAND,
  PAD,
  FLIP,

  // UNSHARD is really a movement op
  UNSHARD,

  // reduce
  REDUCE,
  ALLREDUCE,

  // ** 7 -- pattern compiler IR (used in upat.py) **

  // PYLITERAL carries a Python literal as an arg for CUSTOM predicates
  PYLITERAL
};

#include <unordered_set>

struct GroupOp {
  using Set = std::unordered_set<Ops>;

  // Unary operations
  inline static const Set Unary = {Ops::EXP2, Ops::LOG2,       Ops::SIN,
                                   Ops::SQRT, Ops::RECIPROCAL, Ops::NEG,
                                   Ops::TRUNC};

  // Binary operations
  inline static const Set Binary = {
      Ops::ADD,   Ops::MUL,   Ops::CDIV,     Ops::MAX,      Ops::CMOD,
      Ops::CMPLT, Ops::CMPNE, Ops::CMPEQ,    Ops::XOR,      Ops::SHL,
      Ops::SHR,   Ops::OR,    Ops::AND,      Ops::THREEFRY, Ops::SUB,
      Ops::FDIV,  Ops::POW,   Ops::FLOORDIV, Ops::FLOORMOD};

  // Ternary operations
  inline static const Set Ternary = {Ops::WHERE, Ops::MULACC};

  // ALU = Unary + Binary + Ternary
  inline static const Set ALU = [] {
    Set result = Unary;
    result.insert(Binary.begin(), Binary.end());
    result.insert(Ternary.begin(), Ternary.end());
    return result;
  }();

  // Operations that support broadcasting
  inline static const Set Broadcastable = [] {
    Set result = Binary;
    result.insert(Ternary.begin(), Ternary.end());
    return result;
  }();

  // TODO: Is BITCAST always Elementwise if it's shape changing?
  inline static const Set Elementwise = [] {
    Set result = ALU;
    result.insert(Ops::CAST);
    result.insert(Ops::BITCAST);
    return result;
  }();

  // Operations that define values
  inline static const Set Defines = {Ops::PARAM, Ops::BUFFER};

  // Operations that cannot be decomposed further
  inline static const Set Irreducible = {Ops::CONST, Ops::SPECIAL, Ops::RANGE,
                                         Ops::PARAM, Ops::GETADDR};

  // Tensor movement operations
  inline static const Set Movement = {Ops::RESHAPE, Ops::EXPAND, Ops::PERMUTE,
                                      Ops::PAD,     Ops::SHRINK, Ops::FLIP};

  // Binary operations that can be flipped
  inline static const Set Commutative = {Ops::ADD,   Ops::MUL,   Ops::MAX,
                                         Ops::CMPNE, Ops::CMPEQ, Ops::XOR,
                                         Ops::AND,   Ops::OR};

  // Binary operations where f(f(a,b),c) = f(a,f(b,c))
  inline static const Set Associative = {Ops::ADD, Ops::MUL, Ops::AND, Ops::OR,
                                         Ops::MAX};

  // Binary operations satisfying f(x,x) = x
  // https://en.wikipedia.org/wiki/Idempotence
  inline static const Set Idempotent = {Ops::OR, Ops::AND, Ops::MAX};

  // ALU operations valid as REDUCE/ALLREDUCE operations
  inline static const Set Reduce = {Ops::ADD, Ops::MUL, Ops::MAX};

  // Operations that can change dtype to bool
  inline static const Set Comparison = {Ops::CMPLT, Ops::CMPNE, Ops::CMPEQ};

  // All operations
  inline static const Set All = [] {
    Set result;
    for (int i = static_cast<int>(Ops::SPECIAL);
         i <= static_cast<int>(Ops::PYLITERAL); ++i) {
      result.insert(static_cast<Ops>(i));
    }
    return result;
  }();
};
