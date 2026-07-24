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
    for (const auto& signature : allBuiltinSignatures()) {
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
    return expect(isBuiltinOp("matmul"), "matmul builtin") &&
           expect(isPrimitiveTensorOp("matmul"), "matmul primitive") &&
           expect(isPrimitiveTensorOp("relu"), "relu primitive") &&
           expect(isPrimitiveTensorOp("scale"), "scale primitive") &&
           expect(!isPrimitiveTensorOp("sqrt"), "sqrt non-primitive") &&
           expect(isLibraryOp("sqrt"), "sqrt library") &&
           expect(isCallableLibraryOp("linear"), "linear callable") &&
           expect(isCallableLibraryOp("Tanh"), "Tanh callable") &&
           expect(isCallableLibraryOp("Sigmoid"), "Sigmoid callable") &&
           expect(preservesFirstTensorArg("relu"), "relu preserves first tensor arg") &&
           expect(preservesFirstTensorArg("sqrt"), "sqrt preserves first tensor arg") &&
           expect(!preservesFirstTensorArg("matmul"), "matmul does not preserve first tensor arg") &&
           expect(runtimeSupportsLibraryOp("linear"), "linear runtime library support") &&
           expect(runtimePrimitive("matmul") == RuntimePrimitiveKind::Matmul, "matmul runtime primitive") &&
           expect(runtimePrimitive("relu") == RuntimePrimitiveKind::Relu, "relu runtime primitive");
}

bool op_ids_are_stable() {
    auto matmul = lookupOpId("matmul");
    auto tanh = lookupOpId("Tanh");
    auto rms_norm = lookupOpId("rms_norm");
    return expect(matmul && *matmul == OpId::Matmul, "matmul op id") &&
           expect(tanh && *tanh == OpId::Tanh, "Tanh op id") &&
           expect(rms_norm && *rms_norm == OpId::RmsNorm, "rms_norm op id") &&
           expect(std::string(opIdName(OpId::Rsqrt)) == "Rsqrt", "op id name");
}

bool signatures_preserve_arity_and_types() {
    const OpDefinition* reshape = lookupOp("reshape");
    const OpDefinition* linear = lookupOp("linear");
    const OpDefinition* sum = lookupOp("sum");
    return expect(reshape != nullptr, "reshape lookup") &&
           expect(reshape->signature.minArity == 2 && reshape->signature.maxArity == 8, "reshape arity") &&
           expect(reshape->signature.argTypes.size() == 8, "reshape arg type coverage") &&
           expect(linear != nullptr && linear->signature.returnType.base == TypeBase::Callable, "linear callable return") &&
           expect(sum != nullptr && sum->signature.returnType.tensorRank == 1, "sum rank-one return");
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
