include(GoogleTest)

add_executable(hnsw_tests
  test_main.cpp
  test_kernels.cpp
  test_io.cpp
  test_brute_force.cpp
  test_index.cpp
  test_concurrency.cpp
  test_persist.cpp
  test_filter.cpp)
target_include_directories(hnsw_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(hnsw_tests PRIVATE hnsw::engine hnsw_warnings GTest::gtest)
gtest_discover_tests(hnsw_tests DISCOVERY_TIMEOUT 120 PROPERTIES TIMEOUT 900)
