# Benchmarks
add_executable(vektordb_bench
    math/bench_distance.cpp
)

target_link_libraries(vektordb_bench
    PRIVATE
        vektordb_core
        benchmark::benchmark
        benchmark::benchmark_main
)
