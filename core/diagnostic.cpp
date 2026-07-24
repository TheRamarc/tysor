#include "diagnostic.h"

#include <ostream>
#include <sstream>
#include <utility>

const char* getDiagnosticStageString(DiagnosticCode code) {
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

const char* getDiagnosticCodeString(DiagnosticCode code) {
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

std::string displayStage(DiagnosticCode code) {
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

Diagnostic makeDiagnostic(
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

const char* diagnosticSeverityName(DiagnosticSeverity severity) {
    switch (severity) {
        case DiagnosticSeverity::Error:
            return "Error";
        case DiagnosticSeverity::Warning:
            return "Warning";
        case DiagnosticSeverity::Note:
            return "Note";
    }
    return "Error";
}

Diagnostic Diagnostic::error(DiagnosticCode code, std::string message) {
    return makeDiagnostic(
        DiagnosticSeverity::Error,
        code,
        std::move(message)
    );
}

Diagnostic Diagnostic::warning(DiagnosticCode code, std::string message) {
    return makeDiagnostic(
        DiagnosticSeverity::Warning,
        code,
        std::move(message)
    );
}

Diagnostic Diagnostic::note(DiagnosticCode code, std::string message) {
    return makeDiagnostic(
        DiagnosticSeverity::Note,
        code,
        std::move(message)
    );
}

Diagnostic Diagnostic::withSpan(std::size_t line, std::size_t column) const {
    Diagnostic diagnostic = *this;
    diagnostic.span = SourceSpan{line, column};
    return diagnostic;
}

Diagnostic Diagnostic::withSourceSpan(SourceSpan sourceSpan) const {
    Diagnostic diagnostic = *this;
    diagnostic.span = sourceSpan;
    return diagnostic;
}

Diagnostic Diagnostic::withHelp(std::string helpText) const {
    Diagnostic diagnostic = *this;
    diagnostic.help = std::move(helpText);
    return diagnostic;
}

std::string Diagnostic::stageLabel() const {
    return displayStage(code);
}

std::string Diagnostic::toString() const {
    std::ostringstream out;
    out << *this;
    return out.str();
}

std::ostream& operator<<(std::ostream& out, const Diagnostic& diagnostic) {
    out << diagnostic.stageLabel()
        << ' '
        << diagnosticSeverityName(diagnostic.severity)
        << ": "
        << diagnostic.message;

    if (diagnostic.span.has_value()) {
        out << " at " << diagnostic.span->line << ':' << diagnostic.span->column;
    }

    out << " [" << getDiagnosticCodeString(diagnostic.code) << ']';

    if (diagnostic.help.has_value() && !diagnostic.help->empty()) {
        out << "\nhelp: " << *diagnostic.help;
    }

    return out;
}
