#include "cli_driver.h"

#include <string>
#include <vector>

/**
 * @brief Entry point for the cpptysor compiler and runtime.
 *
 * Converts raw C-style arguments into a C++ vector of strings
 * and delegates execution to the CLI driver.
 *
 * @param argc The number of command line arguments.
 * @param argv The array of command line argument strings.
 * @return int The program exit code.
 */
int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int index = 1; index < argc; ++index) {
        args.emplace_back(argv[index]);
    }
    return run_cli(args);
}

