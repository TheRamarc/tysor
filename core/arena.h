#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <utility>

class Arena {
public:
    Arena(std::size_t blockSize = 64 * 1024);
    ~Arena();

    // Prevent copying and moving to keep pointers stable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;

    template <typename T, typename... Args>
    T* allocate(Args&&... args) {
        void* ptr = allocateBytes(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }

    void* allocateBytes(std::size_t size, std::size_t alignment);

private:
    struct Block {
        std::unique_ptr<char[]> data;
        std::size_t size;
        std::size_t used;

        explicit Block(std::size_t size)
            : data(std::make_unique<char[]>(size)), size(size), used(0) {}
    };

    std::vector<Block> blocks_;
    std::size_t blockSize_;
};
