#include "arena.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

Arena::Arena(std::size_t block_size) : block_size_(block_size) {
    blocks_.emplace_back(block_size_);
}

Arena::~Arena() = default;

void* Arena::allocate_bytes(std::size_t size, std::size_t alignment) {
    if (blocks_.empty()) {
        blocks_.emplace_back(std::max(block_size_, size));
    }

    Block& current_block = blocks_.back();
    std::uintptr_t current_ptr = reinterpret_cast<std::uintptr_t>(current_block.data.get() + current_block.used);
    
    // Calculate padding needed for alignment
    std::size_t padding = 0;
    if (current_ptr % alignment != 0) {
        padding = alignment - (current_ptr % alignment);
    }

    if (current_block.used + padding + size > current_block.size) {
        // Doesn't fit in current block, allocate a new one
        std::size_t new_block_size = std::max(block_size_, size + alignment);
        blocks_.emplace_back(new_block_size);
        Block& new_block = blocks_.back();
        
        current_ptr = reinterpret_cast<std::uintptr_t>(new_block.data.get());
        padding = 0;
        if (current_ptr % alignment != 0) {
            padding = alignment - (current_ptr % alignment);
        }
        
        new_block.used = padding + size;
        return new_block.data.get() + padding;
    }

    // Fits in current block
    current_block.used += padding + size;
    return current_block.data.get() + current_block.used - size;
}
