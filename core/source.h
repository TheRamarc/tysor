#pragma once

#include <cstddef>

/**
 * @brief Represents a specific location within a source file.
 * 
 * This structure is typically used for error reporting and debugging, 
 * to pinpoint where a particular token, AST node, or error originated.
 */
struct SourceSpan {
    /**
     * @brief The line number in the source code (1-indexed).
     * 
     * Why it exists: To provide vertical localization for diagnostics and debugging.
     * What it tracks: The exact line where a specific token or AST node begins.
     * What mutates/updates it: Set by the Lexer during tokenization based on newline characters; remains immutable thereafter.
     */
    std::size_t line;

    /**
     * @brief The column number in the source code (1-indexed).
     * 
     * Why it exists: To provide horizontal localization for precise error pointers.
     * What it tracks: The horizontal offset of a token or AST node on its respective line.
     * What mutates/updates it: Set by the Lexer during tokenization based on character offsets; remains immutable thereafter.
     */
    std::size_t column;
};
