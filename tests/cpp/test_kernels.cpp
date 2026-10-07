#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "hnsw/metric.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

std::vector<simd::Isa> supported_isas() {
  std::vector<simd::Isa> out;
  for (auto isa : {simd::Isa::Scalar, simd::Isa::AVX2, simd::Isa::AVX512, simd::Isa::NEON}) {
    if (simd::supported(isa)) out.push_back(isa);
  }
  return out;
}

class KernelTest : public ::testing::TestWithParam<simd::Isa> {};

TEST_P(KernelTest, MatchesScalarReferenceForAllDims) {
  const auto& k = simd::get(GetParam());
  // +1 float of slack so we can also test misaligned pointers.
  const auto a_buf = test::random_vectors(1, 1025, 1);
  const auto b_buf = test::random_vectors(1, 1025, 2);
  for (std::size_t dim = 1; dim <= 1024; ++dim) {
    for (std::size_t offset : {0u, 1u}) {
      const float* a = a_buf.data() + offset;
      const float* b = b_buf.data() + offset;
      const float ref_l2 = simd::l2_sq_scalar(a, b, dim);
      const float ref_dot = simd::dot_scalar(a, b, dim);
      double dot_scale = 0.0;
      for (std::size_t i = 0; i < dim; ++i) dot_scale += std::fabs(double(a[i]) * b[i]);
      EXPECT_NEAR(k.l2_sq(a, b, dim), ref_l2, 1e-5 * std::max(1.0f, ref_l2))
          << "isa=" << simd::isa_name(GetParam()) << " dim=" << dim;
      EXPECT_NEAR(k.dot(a, b, dim), ref_dot, 1e-5 * std::max(1.0, dot_scale))
          << "isa=" << simd::isa_name(GetParam()) << " dim=" << dim;
    }
  }
}

TEST_P(KernelTest, ExactOnSmallIntegers) {
  const auto& k = simd::get(GetParam());
  for (std::size_t dim : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 33u, 63u, 64u, 65u, 100u, 128u, 960u}) {
    std::vector<float> a(dim), b(dim);
    double l2 = 0, dp = 0;
    for (std::size_t i = 0; i < dim; ++i) {
      a[i] = static_cast<float>(int(i % 7) - 3);
      b[i] = static_cast<float>(int(i % 5) - 2);
      l2 += (a[i] - b[i]) * (a[i] - b[i]);
      dp += a[i] * b[i];
    }
    EXPECT_EQ(k.l2_sq(a.data(), b.data(), dim), static_cast<float>(l2)) << dim;
    EXPECT_EQ(k.dot(a.data(), b.data(), dim), static_cast<float>(dp)) << dim;
  }
}

TEST_P(KernelTest, ZeroDim) {
  const auto& k = simd::get(GetParam());
  float a = 1.0f, b = 2.0f;
  EXPECT_EQ(k.l2_sq(&a, &b, 0), 0.0f);
  EXPECT_EQ(k.dot(&a, &b, 0), 0.0f);
}

INSTANTIATE_TEST_SUITE_P(AllIsas, KernelTest, ::testing::ValuesIn(supported_isas()),
                         [](const auto& info) { return std::string(simd::isa_name(info.param)); });

TEST(Dispatch, ActiveIsSupported) {
  EXPECT_TRUE(simd::supported(simd::active().isa));
  EXPECT_TRUE(simd::supported(simd::Isa::Scalar));
  EXPECT_NO_THROW(simd::get(simd::Isa::Scalar));
}

TEST(Dispatch, ParseNames) {
  EXPECT_EQ(simd::parse_isa("avx2"), simd::Isa::AVX2);
  EXPECT_THROW(simd::parse_isa("sse9"), std::invalid_argument);
  EXPECT_EQ(parse_metric("l2"), Metric::L2);
  EXPECT_EQ(parse_metric("cosine"), Metric::Cosine);
  EXPECT_EQ(parse_metric("ip"), Metric::InnerProduct);
  EXPECT_THROW(parse_metric("hamming"), std::invalid_argument);
}

}  // namespace
