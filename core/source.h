#pragma once

#include <cstddef>

struct SourceSpan {
    std::size_t line;
    std::size_t column;
};
