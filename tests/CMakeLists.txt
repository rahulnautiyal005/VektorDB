# Tests
add_executable(vektordb_tests
    math/test_distance.cpp
)

target_link_libraries(vektordb_tests
    PRIVATE
        vektordb_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(vektordb_tests)
