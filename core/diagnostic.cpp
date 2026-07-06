#include "diagnostic.h"

#include <ostream>
#include <sstream>
#include <utility>

namespace {

/**
 * @brief Helper function to format the internal stage name into a human-readable string.
 * 
 * Maps internal stage identifiers to their capitalized display names.
 * Unrecognized stages are returned as-is.
 * 
 * @param stage The internal stage string.
 * @return A human-readable display string for the stage.
 */
std::string display_stage(const std::string& stage) {
    if (stage == "lexer") {
        return "Lexer";
    }
    if (stage == "parser") {
        return "Parser";
    }
    if (stage == "semantic") {
        return "Semantic";
    }
    if (stage == "frontend_ir") {
        return "Frontend IR";
    }
    if (stage == "graph_ir") {
        return "Graph IR";
    }
    if (stage == "backend") {
        return "Backend";
    }
    // Fallback: return the original string if no mapping matches.
    return std::string(stage);
}

/**
 * @brief Internal helper to construct a Diagnostic object.
 * 
 * Initializes a Diagnostic object using the provided attributes,
 * using std::move to efficiently transfer ownership of the strings.
 */
Diagnostic make_diagnostic(
    DiagnosticSeverity severity,
    std::string stage,
    std::string code,
    std::string message
) {
    Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.stage = std::move(stage);
    diagnostic.code = std::move(code);
    diagnostic.message = std::move(message);
    return diagnostic;
}

} // namespace

const char* diagnostic_severity_name(DiagnosticSeverity severity) {
    switch (severity) {
        case DiagnosticSeverity::Error:
            return "Error";
        case DiagnosticSeverity::Warning:
            return "Warning";
        case DiagnosticSeverity::Note:
            return "Note";
    }
    // Default fallback in case an invalid enum value is somehow passed.
    return "Error";
}

Diagnostic Diagnostic::error(std::string stage, std::string code, std::string message) {
    return make_diagnostic(
        DiagnosticSeverity::Error,
        std::move(stage),
        std::move(code),
        std::move(message)
    );
}

Diagnostic Diagnostic::warning(std::string stage, std::string code, std::string message) {
    return make_diagnostic(
        DiagnosticSeverity::Warning,
        std::move(stage),
        std::move(code),
        std::move(message)
    );
}

Diagnostic Diagnostic::note(std::string stage, std::string code, std::string message) {
    return make_diagnostic(
        DiagnosticSeverity::Note,
        std::move(stage),
        std::move(code),
        std::move(message)
    );
}

Diagnostic Diagnostic::with_span(std::size_t line, std::size_t column) const {
    // Copy the current diagnostic object, set the span, and return the copy.
    Diagnostic diagnostic = *this;
    diagnostic.span = SourceSpan{line, column};
    return diagnostic;
}

Diagnostic Diagnostic::with_source_span(SourceSpan source_span) const {
    Diagnostic diagnostic = *this;
    diagnostic.span = source_span;
    return diagnostic;
}

Diagnostic Diagnostic::with_help(std::string help_text) const {
    Diagnostic diagnostic = *this;
    diagnostic.help = std::move(help_text);
    return diagnostic;
}

std::string Diagnostic::stage_label() const {
    return display_stage(stage);
}

std::string Diagnostic::to_string() const {
    // Use an ostringstream and the overloaded operator<< to format the diagnostic.
    std::ostringstream out;
    out << *this;
    return out.str();
}

std::ostream& operator<<(std::ostream& out, const Diagnostic& diagnostic) {
    // Basic format: "Stage Severity: Message"
    out << diagnostic.stage_label()
        << ' '
        << diagnostic_severity_name(diagnostic.severity)
        << ": "
        << diagnostic.message;

    // Append source location if available
    if (diagnostic.span.has_value()) {
        out << " at " << diagnostic.span->line << ':' << diagnostic.span->column;
    }

    // Append error/diagnostic code if provided
    if (!diagnostic.code.empty()) {
        out << " [" << diagnostic.code << ']';
    }

    // Append help message on a new line if available
    if (diagnostic.help.has_value() && !diagnostic.help->empty()) {
        out << "\nhelp: " << *diagnostic.help;
    }

    return out;
}
