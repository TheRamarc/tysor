#pragma once

#include <string>
#include <vector>

/**
 * @brief Main entry point for the Command Line Interface (CLI) driver.
 *
 * Parses the raw command line arguments provided by the user, configures the
 * compilation pipeline, reads the source file, coordinates the compilation steps,
 * and executes requested actions such as running the code or dumping IRs.
 *
 * @param raw_args A vector of strings containing the unparsed command line arguments.
 * @return int The exit status code of the program (0 for success, non-zero for failure).
 */
int run_cli(const std::vector<std::string>& raw_args);
