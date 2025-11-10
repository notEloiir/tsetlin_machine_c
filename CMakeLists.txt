# Build instructions:
# 1) Configure:
#    cmake -B build
# 2) Build:
#    cmake --build build
cmake_minimum_required(VERSION 3.23)
project(tsetlin_machine_c
    LANGUAGES C
    VERSION 1.0.0
)

option(ENABLE_FLATCC "Build FlatCC from third_party" ON)
option(BUILD_TESTING "Enable building third_party tests" ON)
option(BUILD_EXAMPLES "Enable building examples" ON)

add_subdirectory(third_party)

if(ENABLE_FLATCC)
    add_subdirectory(schemas)
endif()

add_subdirectory(src/tsetlin_machine_c)

add_subdirectory(examples)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
