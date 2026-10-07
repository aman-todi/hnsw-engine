#include "hnsw/io.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace hnsw::io {
namespace {

template <typename T>
Matrix<T> read_vecs(const std::string& path, std::size_t max_rows) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const auto file_size = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  Matrix<T> m;
  if (file_size == 0) return m;

  int32_t dim = 0;
  if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim)) || dim <= 0) {
    throw std::runtime_error(path + ": invalid leading dimension");
  }
  const std::size_t row_bytes = sizeof(int32_t) + static_cast<std::size_t>(dim) * sizeof(T);
  if (file_size % row_bytes != 0) {
    throw std::runtime_error(path + ": file size is not a multiple of the row size (truncated?)");
  }
  std::size_t rows = file_size / row_bytes;
  if (max_rows != 0 && max_rows < rows) rows = max_rows;

  m.rows = rows;
  m.cols = static_cast<std::size_t>(dim);
  m.data.resize(rows * m.cols);
  in.seekg(0);
  for (std::size_t r = 0; r < rows; ++r) {
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!in || d != dim)
      throw std::runtime_error(path + ": inconsistent dimension in row " + std::to_string(r));
    in.read(reinterpret_cast<char*>(m.row(r)), static_cast<std::streamsize>(m.cols * sizeof(T)));
    if (!in) throw std::runtime_error(path + ": unexpected end of file");
  }
  return m;
}

template <typename T>
void write_vecs(const std::string& path, const T* data, std::size_t rows, std::size_t cols) {
  if (cols == 0 || cols > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("write_vecs: invalid number of columns");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open " + path + " for writing");
  const auto dim = static_cast<int32_t>(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(data + r * cols), static_cast<std::streamsize>(cols * sizeof(T)));
  }
  if (!out) throw std::runtime_error("write failed: " + path);
}

}  // namespace

Matrix<float> read_fvecs(const std::string& path, std::size_t max_rows) {
  return read_vecs<float>(path, max_rows);
}
Matrix<int32_t> read_ivecs(const std::string& path, std::size_t max_rows) {
  return read_vecs<int32_t>(path, max_rows);
}
void write_fvecs(const std::string& path, const float* data, std::size_t rows, std::size_t cols) {
  write_vecs(path, data, rows, cols);
}
void write_ivecs(const std::string& path, const int32_t* data, std::size_t rows, std::size_t cols) {
  write_vecs(path, data, rows, cols);
}

}  // namespace hnsw::io
