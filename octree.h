#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>

const uint8_t FLAG_LEAF = 1;
const uint8_t FLAG_ROOT = 127;

const size_t MAX_DEPTH = 21;


template<typename T>
struct data_array {
  T *data;

  data_array(size_t capacity) { data = new T[capacity]; }
  ~data_array() { delete[] data; }
};

using idx_t = size_t;
using path_t = size_t;

template <typename... Ts>
class octree {
  size_t capacity;
  std::atomic_size_t next_alloc;

  size_t *child_block_base; // index of the first child of each node
  size_t *parent;    // index of the parent of each node
  uint8_t *flags;    // flags stored for each node
  std::tuple<data_array<Ts>...> data;

  size_t alloc_child_block () {
    size_t block = next_alloc.fetch_add(8); // 8 children per block
    if (block >= capacity) {
      throw std::runtime_error("Out of tree capacity");
    }
    return block;
  }

  public:
  octree(size_t capacity) : capacity(capacity), next_alloc(0), data(std::make_tuple(data_array<Ts>(capacity)...)){
    child_block_base = new size_t[capacity];
    parent = new size_t[capacity];
    flags = new uint8_t[capacity];
  }

  ~octree() {
    delete[] child_block_base;
    delete[] parent;
    delete[] flags;

    std::apply([](auto &...x) { (delete[] x.data, ...); }, data);
  }







};











