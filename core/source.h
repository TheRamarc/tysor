#pragma once

#include <cstddef>

/**
 * @brief Represents a specific location within a source file.
 * 
 * This structure is typically used for error reporting and debugging, 
 * to pinpoint where a particular token, AST node, or error originated.
 */
struct SourceSpan {
    /// The line number in the source code (1-indexed).
    std::size_t line;

    /// The column number in the source code (1-indexed).
    std::size_t column;
};
