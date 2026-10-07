add_executable(bench_main bench_main.cpp)
target_link_libraries(bench_main PRIVATE hnsw::engine hnsw_warnings)

add_executable(micro_bench micro_bench.cpp)
target_link_libraries(micro_bench PRIVATE hnsw::engine benchmark::benchmark)
