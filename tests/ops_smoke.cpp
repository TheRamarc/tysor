#include "ops.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "ops-smoke: failed " << name << '\n';
        return false;
    }
    return true;
}

bool signatures_are_unique_and_complete() {
    std::set<std::string> names;
    for (const auto& signature : all_builtin_signatures()) {
        if (signature.name.empty()) {
            std::cerr << "ops-smoke: empty builtin signature name\n";
            return false;
        }
        if (!names.insert(signature.name).second) {
            std::cerr << "ops-smoke: duplicate builtin signature " << signature.name << '\n';
            return false;
        }
    }

    const std::vector<std::string> required{
        "linear",
        "matmul",
        "relu",
        "scale",
        "Tanh",
        "Sigmoid",
        "rms_norm",
        "sqrt",
        "rsqrt",
        "sum",
        "mean",
    };
    for (const auto& name : required) {
        if (names.find(name) == names.end()) {
            std::cerr << "ops-smoke: missing builtin signature " << name << '\n';
            return false;
        }
    }
    return true;
}

bool classifications_match_compiler_expectations() {
    return expect(is_builtin_op("matmul"), "matmul builtin") &&
           expect(is_primitive_tensor_op("matmul"), "matmul primitive") &&
           expect(is_primitive_tensor_op("relu"), "relu primitive") &&
           expect(is_primitive_tensor_op("scale"), "scale primitive") &&
           expect(!is_primitive_tensor_op("sqrt"), "sqrt non-primitive") &&
           expect(is_library_op("sqrt"), "sqrt library") &&
           expect(is_callable_library_op("linear"), "linear callable") &&
           expect(is_callable_library_op("Tanh"), "Tanh callable") &&
           expect(is_callable_library_op("Sigmoid"), "Sigmoid callable") &&
           expect(preserves_first_tensor_arg("relu"), "relu preserves first tensor arg") &&
           expect(preserves_first_tensor_arg("sqrt"), "sqrt preserves first tensor arg") &&
           expect(!preserves_first_tensor_arg("matmul"), "matmul does not preserve first tensor arg") &&
           expect(runtime_supports_library_op("linear"), "linear runtime library support") &&
           expect(runtime_primitive("matmul") == RuntimePrimitiveKind::Matmul, "matmul runtime primitive") &&
           expect(runtime_primitive("relu") == RuntimePrimitiveKind::Relu, "relu runtime primitive");
}

bool op_ids_are_stable() {
    auto matmul = lookup_op_id("matmul");
    auto tanh = lookup_op_id("Tanh");
    auto rms_norm = lookup_op_id("rms_norm");
    return expect(matmul && *matmul == OpId::Matmul, "matmul op id") &&
           expect(tanh && *tanh == OpId::Tanh, "Tanh op id") &&
           expect(rms_norm && *rms_norm == OpId::RmsNorm, "rms_norm op id") &&
           expect(std::string(op_id_name(OpId::Rsqrt)) == "Rsqrt", "op id name");
}

bool signatures_preserve_arity_and_types() {
    const OpDefinition* reshape = lookup_op("reshape");
    const OpDefinition* linear = lookup_op("linear");
    const OpDefinition* sum = lookup_op("sum");
    return expect(reshape != nullptr, "reshape lookup") &&
           expect(reshape->signature.min_arity == 2 && reshape->signature.max_arity == 8, "reshape arity") &&
           expect(reshape->signature.arg_types.size() == 8, "reshape arg type coverage") &&
           expect(linear != nullptr && linear->signature.return_type.base == TypeBase::Callable, "linear callable return") &&
           expect(sum != nullptr && sum->signature.return_type.tensor_rank == 1, "sum rank-one return");
}

} // namespace

int main() {
    const std::vector<bool> checks{
        signatures_are_unique_and_complete(),
        classifications_match_compiler_expectations(),
        op_ids_are_stable(),
        signatures_preserve_arity_and_types(),
    };

    for (bool check : checks) {
        if (!check) {
            return 1;
        }
    }
    return 0;
}
