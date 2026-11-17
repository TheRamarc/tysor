#include "dtype.h"
#include "op.h"
#include <any>
#include <vector>

class Tir;

DType dtype_from_tir(
    Ops op,
    const std::vector<Tir>& src,
    const std::any& arg
);

class Tir {
public:
    Ops op;
    std::vector<Tir> src{};
    std::any arg{};
    std::any tag{};

    DType dtype() const {
        return dtype_from_tir(op, src, arg);
    }
};