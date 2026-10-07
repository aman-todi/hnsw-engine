#pragma once

#include <cstddef>
#include <string>

namespace hnsw::detail {

/// RAII read-only memory mapping of a whole file (POSIX).
class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const unsigned char* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool valid() const noexcept { return data_ != nullptr; }

 private:
  void reset() noexcept;
  const unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace hnsw::detail
