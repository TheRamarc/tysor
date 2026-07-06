#pragma once

#include "source.h"

#include <iosfwd>
#include <optional>
#include <string>

/**
 * @brief Represents the severity level of a diagnostic message.
 */
enum class DiagnosticSeverity {
    /// A fatal error that typically halts compilation or execution.
    Error,
    /// A warning indicating a potential issue that doesn't halt execution.
    Warning,
    /// An informational note providing additional context to another diagnostic.
    Note,
};

/**
 * @brief Converts a DiagnosticSeverity enum value to its string representation.
 * 
 * @param severity The severity level to convert.
 * @return A constant character pointer representing the severity name.
 */
const char* diagnostic_severity_name(DiagnosticSeverity severity);

/**
 * @brief Represents a diagnostic message generated during various compilation stages.
 * 
 * This structure holds all necessary information to report errors, warnings,
 * and notes to the user, including the stage it occurred, a specific error code,
 * the message itself, and optionally the location in the source code and a help message.
 */
struct Diagnostic {
    /**
     * @brief A unique code identifying the specific type of diagnostic (e.g., "E001").
     * 
     * Why it exists: Allows programmatic filtering and easy referencing of known issues.
     * What it tracks: A fixed string representation of the error category.
     * What mutates/updates it: Initialized on creation by static factory methods (e.g., error(), warning()) and usually unchanged.
     */
    std::string code;
    
    /**
     * @brief The compilation stage where the diagnostic was generated (e.g., "lexer", "parser").
     * 
     * Why it exists: Helps developers isolate which phase of compilation failed.
     * What it tracks: The name of the subsystem that emitted this diagnostic.
     * What mutates/updates it: Set during instantiation and immutable afterwards.
     */
    std::string stage;
    
    /**
     * @brief The severity of the diagnostic, defaulting to Error.
     * 
     * Why it exists: Determines how the compiler should react (fail, print warning, etc.).
     * What it tracks: The criticality level (Error, Warning, Note).
     * What mutates/updates it: Defined at creation via the respective factory method (error, warning, note).
     */
    DiagnosticSeverity severity = DiagnosticSeverity::Error;
    
    /**
     * @brief The detailed diagnostic message.
     * 
     * Why it exists: Provides a human-readable explanation of the issue.
     * What it tracks: The specific contextual error text.
     * What mutates/updates it: Set at creation time based on the exact failure condition.
     */
    std::string message;
    
    /**
     * @brief An optional source location indicating where the issue occurred.
     * 
     * Why it exists: Crucial for pointing users to the exact place in their code causing the issue.
     * What it tracks: The line and column spanning the erroneous syntax or semantics.
     * What mutates/updates it: Initially empty or set at creation; can be updated by `with_span()` or `with_source_span()` to add context as the error bubbles up.
     */
    std::optional<SourceSpan> span;
    
    /**
     * @brief An optional help message suggesting a fix or providing more details.
     * 
     * Why it exists: Improves developer experience by offering actionable solutions.
     * What it tracks: A secondary string giving guidance to resolve the diagnostic.
     * What mutates/updates it: Initially empty; typically mutated/added via the `with_help()` method.
     */
    std::optional<std::string> help;


    // [[nodiscard]] do not ignore the returned Diagnostic
    
    /**
     * @brief Creates a diagnostic representing an error.
     * 
     * @param stage The stage of compilation.
     * @param code The error code.
     * @param message The detailed error message.
     * @return A new Diagnostic instance with Error severity.
     */
    [[nodiscard]] static Diagnostic error(std::string stage, std::string code, std::string message);

    /**
     * @brief Creates a diagnostic representing a warning.
     * 
     * @param stage The stage of compilation.
     * @param code The warning code.
     * @param message The detailed warning message.
     * @return A new Diagnostic instance with Warning severity.
     */
    [[nodiscard]] static Diagnostic warning(std::string stage, std::string code, std::string message);

    /**
     * @brief Creates a diagnostic representing an informational note.
     * 
     * @param stage The stage of compilation.
     * @param code The note code.
     * @param message The detailed note message.
     * @return A new Diagnostic instance with Note severity.
     */
    [[nodiscard]] static Diagnostic note(std::string stage, std::string code, std::string message);

    /**
     * @brief Creates a copy of this diagnostic with a specified source location.
     * 
     * @param line The line number.
     * @param column The column number.
     * @return A new Diagnostic instance with the updated span.
     */
    [[nodiscard]] Diagnostic with_span(std::size_t line, std::size_t column) const;

    /**
     * @brief Creates a copy of this diagnostic with a specified SourceSpan.
     * 
     * @param source_span The SourceSpan to attach.
     * @return A new Diagnostic instance with the updated span.
     */
    [[nodiscard]] Diagnostic with_source_span(SourceSpan source_span) const;

    /**
     * @brief Creates a copy of this diagnostic with a specified help message.
     * 
     * @param help_text The help text to attach.
     * @return A new Diagnostic instance with the updated help text.
     */
    [[nodiscard]] Diagnostic with_help(std::string help_text) const;

    /**
     * @brief Retrieves a human-readable label for the compilation stage.
     * 
     * @return A formatted string representing the stage (e.g., "Lexer", "Parser").
     */
    [[nodiscard]] std::string stage_label() const;

    /**
     * @brief Converts the entire diagnostic to a formatted string.
     * 
     * @return A string containing the stage, severity, message, location, code, and help text.
     */
    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Overloads the stream insertion operator to print a diagnostic.
 * 
 * @param out The output stream.
 * @param diagnostic The diagnostic to print.
 * @return The modified output stream.
 */
std::ostream& operator<<(std::ostream& out, const Diagnostic& diagnostic);
