# C++ Coding Style Guide

## Naming

### Classes, Structs, Unions

Use **PascalCase**.

```cpp
class Tensor;
class Graph;
struct Shape;
struct Token;
```

---

### Enums

Use **PascalCase** for enum types and enum values.

```cpp
enum class TokenKind {
    Identifier,
    Number,
    String,
    EndOfFile,
};
```

---

### Functions and Methods

Use **camelCase**.

```cpp
parseProgram();
buildGraph();
computeGradient();
zeroGrad();
```

---

### Variables

Use **camelCase**.

```cpp
learningRate
batchSize
currentToken
executionPlan
```

---

### Member Variables

Use **camelCase** with a trailing underscore (`_`).

```cpp
class Tensor {
private:
    Shape shape_;
    Storage storage_;
    Device device_;
};
```

---

### Constants

Use **kPascalCase**.

```cpp
constexpr int kMaxRank = 8;
constexpr float kEpsilon = 1e-6f;
```

---

### Macros

Use **SCREAMING_SNAKE_CASE**.

```cpp
ASSERT(...)
UNREACHABLE()
MAX_BUFFER_SIZE
```

---

### Namespaces

Use **lowercase**.

```cpp
namespace tysor {}
namespace tysor::frontend {}
namespace tysor::backend {}
```

---

### Files

Use **lowercase** with words separated by underscores if needed.

```
tensor.cpp
execution_plan.cpp
graph_builder.hpp
```

---

### Template Parameters

Use **PascalCase**.

```cpp
template<typename T>
template<typename Elem>
template<typename Storage>
```

---

## Examples

```cpp
namespace tysor {

struct Shape {
    int rows;
    int cols;
};

class Tensor {
public:
    Tensor(Shape shape);

    Tensor matmul(const Tensor& other) const;
    void zeroGrad();
    void reshape();

private:
    Shape shape_;
    Storage storage_;
    Device device_;
};

}
```
