#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hnsw::io {

/// A dense row-major matrix loaded from an .fvecs / .ivecs file.
template <typename T>
struct Matrix {
  std::vector<T> data;
  std::size_t rows = 0;
  std::size_t cols = 0;

  const T* row(std::size_t i) const { return data.data() + i * cols; }
  T* row(std::size_t i) { return data.data() + i * cols; }
};

/// Read an .fvecs file (per vector: int32 dim, then dim float32 values).
/// `max_rows` = 0 reads everything; otherwise at most `max_rows` vectors.
/// Throws std::runtime_error on I/O errors, inconsistent dims or truncation.
Matrix<float> read_fvecs(const std::string& path, std::size_t max_rows = 0);
/// Read an .ivecs file (per vector: int32 dim, then dim int32 values).
Matrix<int32_t> read_ivecs(const std::string& path, std::size_t max_rows = 0);

void write_fvecs(const std::string& path, const float* data, std::size_t rows, std::size_t cols);
void write_ivecs(const std::string& path, const int32_t* data, std::size_t rows, std::size_t cols);

}  // namespace hnsw::io
