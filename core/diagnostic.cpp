#include "diagnostic.h"

#include <ostream>
#include <sstream>
#include <utility>

const char* get_diagnostic_stage_string(DiagnosticCode code) {
    switch (code) {
        case DiagnosticCode::LexerError: return "lexer";
        case DiagnosticCode::ParserError: return "parser";
        case DiagnosticCode::SemanticError: return "semantic";
        case DiagnosticCode::FrontendIrError: return "frontend_ir";
        case DiagnosticCode::GraphIrError: return "graph_ir";
        case DiagnosticCode::BackendError: return "backend";
        case DiagnosticCode::RuntimeError:
        case DiagnosticCode::RuntimeExecutionError:
        case DiagnosticCode::RuntimeBackwardError:
        case DiagnosticCode::RuntimeTrainError: return "runtime";
        case DiagnosticCode::CliError:
        case DiagnosticCode::CliFileError:
        case DiagnosticCode::CliArgsError: return "cli";
        case DiagnosticCode::TestError: return "test";
    }
    return "unknown";
}

const char* get_diagnostic_code_string(DiagnosticCode code) {
    switch (code) {
        case DiagnosticCode::LexerError: return "L0001";
        case DiagnosticCode::ParserError: return "P0001";
        case DiagnosticCode::SemanticError: return "S0001";
        case DiagnosticCode::FrontendIrError: return "F0001";
        case DiagnosticCode::GraphIrError: return "G0001";
        case DiagnosticCode::BackendError: return "B0001";
        case DiagnosticCode::RuntimeError: return "R0001";
        case DiagnosticCode::RuntimeExecutionError: return "R0002";
        case DiagnosticCode::RuntimeBackwardError: return "R0003";
        case DiagnosticCode::RuntimeTrainError: return "R0004";
        case DiagnosticCode::CliError: return "C0001";
        case DiagnosticCode::CliFileError: return "C0002";
        case DiagnosticCode::CliArgsError: return "C0003";
        case DiagnosticCode::TestError: return "T0001";
    }
    return "U0000";
}

namespace {

std::string display_stage(DiagnosticCode code) {
    switch (code) {
        case DiagnosticCode::LexerError: return "Lexer";
        case DiagnosticCode::ParserError: return "Parser";
        case DiagnosticCode::SemanticError: return "Semantic";
        case DiagnosticCode::FrontendIrError: return "Frontend IR";
        case DiagnosticCode::GraphIrError: return "Graph IR";
        case DiagnosticCode::BackendError: return "Backend";
        case DiagnosticCode::RuntimeError:
        case DiagnosticCode::RuntimeExecutionError:
        case DiagnosticCode::RuntimeBackwardError:
        case DiagnosticCode::RuntimeTrainError: return "Runtime";
        case DiagnosticCode::CliError:
        case DiagnosticCode::CliFileError:
        case DiagnosticCode::CliArgsError: return "CLI";
        case DiagnosticCode::TestError: return "Test";
    }
    return "Unknown";
}

/**
 * @brief Internal helper to construct a Diagnostic object.
 * 
 * Initializes a Diagnostic object using the provided attributes,
 * using std::move to efficiently transfer ownership of the strings.
 */
Diagnostic make_diagnostic(
    DiagnosticSeverity severity,
    DiagnosticCode code,
    std::string message
) {
    Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = code;
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

Diagnostic Diagnostic::error(DiagnosticCode code, std::string message) {
    return make_diagnostic(
        DiagnosticSeverity::Error,
        code,
        std::move(message)
    );
}

Diagnostic Diagnostic::warning(DiagnosticCode code, std::string message) {
    return make_diagnostic(
        DiagnosticSeverity::Warning,
        code,
        std::move(message)
    );
}

Diagnostic Diagnostic::note(DiagnosticCode code, std::string message) {
    return make_diagnostic(
        DiagnosticSeverity::Note,
        code,
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
    return display_stage(code);
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

    // Append error/diagnostic code
    out << " [" << get_diagnostic_code_string(diagnostic.code) << ']';

    // Append help message on a new line if available
    if (diagnostic.help.has_value() && !diagnostic.help->empty()) {
        out << "\nhelp: " << *diagnostic.help;
    }

    return out;
}
