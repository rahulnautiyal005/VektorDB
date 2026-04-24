# Tests
add_executable(vektordb_tests
    math/test_distance.cpp
    storage/test_storage.cpp
    search/test_flat_search.cpp
    index/test_hnsw.cpp
)

target_link_libraries(vektordb_tests
    PRIVATE
        vektordb_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(vektordb_tests)
