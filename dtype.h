#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>

class DType {
public:
  const int priority;
  const int bitsize;
  const std::string name;
  const std::string fmt; // or std::optional<std::string>

  [[nodiscard]]
  int itemsize() const noexcept {
    return (bitsize + 7) / 8;
  }

  static const DType &new_dtype(int priority, int bitsize, std::string name,
                                std::string fmt);

  // ...

private:
  DType(int priority, int bitsize, std::string name, std::string fmt)
      : priority(priority), bitsize(bitsize), name(std::move(name)),
        fmt(std::move(fmt)) {}

  struct Key {
    int priority;
    int bitsize;
    std::string name;
    std::string fmt;

    bool operator==(const Key &other) const {
      return priority == other.priority && bitsize == other.bitsize &&
             name == other.name && fmt == other.fmt;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key &k) const {
      std::size_t h = std::hash<int>{}(k.priority);

      h ^= std::hash<int>{}(k.bitsize) + 0x9e3779b9 + (h << 6) + (h >> 2);

      h ^= std::hash<std::string>{}(k.name) + 0x9e3779b9 + (h << 6) + (h >> 2);

      h ^= std::hash<std::string>{}(k.fmt) + 0x9e3779b9 + (h << 6) + (h >> 2);

      return h;
    }
  };

  static inline std::unordered_map<Key, std::unique_ptr<DType>, KeyHash> cache;
};

class Buffer{

};