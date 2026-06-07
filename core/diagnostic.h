#pragma once

#include "source.h"

#include <iosfwd>
#include <optional>
#include <string>

enum class DiagnosticSeverity {
    Error,
    Warning,
    Note,
};

const char* diagnostic_severity_name(DiagnosticSeverity severity);

struct Diagnostic {
    std::string code;
    std::string stage;
    DiagnosticSeverity severity = DiagnosticSeverity::Error;
    std::string message;
    std::optional<SourceSpan> span;
    std::optional<std::string> help;


    // [[nodiscard]] do not ignore the returned Diagnostic
    [[nodiscard]] static Diagnostic error(std::string stage, std::string code, std::string message);
    [[nodiscard]] static Diagnostic warning(std::string stage, std::string code, std::string message);
    [[nodiscard]] static Diagnostic note(std::string stage, std::string code, std::string message);

    [[nodiscard]] Diagnostic with_span(std::size_t line, std::size_t column) const;
    [[nodiscard]] Diagnostic with_source_span(SourceSpan source_span) const;
    [[nodiscard]] Diagnostic with_help(std::string help_text) const;

    [[nodiscard]] std::string stage_label() const;
    [[nodiscard]] std::string to_string() const;
};

std::ostream& operator<<(std::ostream& out, const Diagnostic& diagnostic);
