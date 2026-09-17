#include "dtype.h"
#include "ops.h"
#include <any>
#include <vector>

class Tir {
public:
  Ops op;
  std::vector<Tir> src{};
  std::any arg{};
  std::any tag{};

  DType dtype() { return dtype_from_tir() }
};

DType dtype_from_uop(Ops op, std::vector<Tir> src, std::any arg);
