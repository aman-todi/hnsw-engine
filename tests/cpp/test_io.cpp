#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "hnsw/io.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

std::string tmp_path(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("hnsw_io_" + name)).string();
}

TEST(Io, FvecsRoundTrip) {
  const auto data = test::random_vectors(37, 13, 5);
  const auto path = tmp_path("a.fvecs");
  io::write_fvecs(path, data.data(), 37, 13);
  const auto m = io::read_fvecs(path);
  EXPECT_EQ(m.rows, 37u);
  EXPECT_EQ(m.cols, 13u);
  EXPECT_EQ(m.data, data);
  const auto sub = io::read_fvecs(path, 10);
  EXPECT_EQ(sub.rows, 10u);
  EXPECT_TRUE(std::equal(sub.data.begin(), sub.data.end(), data.begin()));
  std::remove(path.c_str());
}

TEST(Io, IvecsRoundTrip) {
  std::vector<int32_t> data(5 * 100);
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<int32_t>(i * 7 - 3);
  const auto path = tmp_path("a.ivecs");
  io::write_ivecs(path, data.data(), 5, 100);
  const auto m = io::read_ivecs(path);
  EXPECT_EQ(m.rows, 5u);
  EXPECT_EQ(m.cols, 100u);
  EXPECT_EQ(m.data, data);
  std::remove(path.c_str());
}

TEST(Io, TruncatedFileThrows) {
  const auto data = test::random_vectors(4, 8, 5);
  const auto path = tmp_path("t.fvecs");
  io::write_fvecs(path, data.data(), 4, 8);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 3);
  EXPECT_THROW(io::read_fvecs(path), std::runtime_error);
  std::remove(path.c_str());
  EXPECT_THROW(io::read_fvecs(tmp_path("does_not_exist")), std::runtime_error);
}

}  // namespace
