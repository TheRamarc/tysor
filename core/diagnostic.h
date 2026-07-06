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
    /// A unique code identifying the specific type of diagnostic (e.g., "E001").
    std::string code;
    /// The compilation stage where the diagnostic was generated (e.g., "lexer", "parser").
    std::string stage;
    /// The severity of the diagnostic, defaulting to Error.
    DiagnosticSeverity severity = DiagnosticSeverity::Error;
    /// The detailed diagnostic message.
    std::string message;
    /// An optional source location indicating where the issue occurred.
    std::optional<SourceSpan> span;
    /// An optional help message suggesting a fix or providing more details.
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
