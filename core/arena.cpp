#include "arena.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

Arena::Arena(std::size_t blockSize) : blockSize_(blockSize) {
    blocks_.emplace_back(blockSize_);
}

Arena::~Arena() = default;

void* Arena::allocateBytes(std::size_t size, std::size_t alignment) {
    if (blocks_.empty()) {
        blocks_.emplace_back(std::max(blockSize_, size));
    }

    Block& currentBlock = blocks_.back();
    std::uintptr_t currentPtr = reinterpret_cast<std::uintptr_t>(currentBlock.data.get() + currentBlock.used);
    
    // Calculate padding needed for alignment
    std::size_t padding = 0;
    if (currentPtr % alignment != 0) {
        padding = alignment - (currentPtr % alignment);
    }

    if (currentBlock.used + padding + size > currentBlock.size) {
        // Doesn't fit in current block, allocate a new one
        std::size_t newBlockSize = std::max(blockSize_, size + alignment);
        blocks_.emplace_back(newBlockSize);
        Block& newBlock = blocks_.back();
        
        currentPtr = reinterpret_cast<std::uintptr_t>(newBlock.data.get());
        padding = 0;
        if (currentPtr % alignment != 0) {
            padding = alignment - (currentPtr % alignment);
        }
        
        newBlock.used = padding + size;
        return newBlock.data.get() + padding;
    }

    // Fits in current block
    currentBlock.used += padding + size;
    return currentBlock.data.get() + currentBlock.used - size;
}
