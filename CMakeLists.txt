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

option(BUILD_FLATCC "Build FlatCC from third_party" ON)
option(BUILD_EXAMPLES "Enable building examples" ON)
option(BUILD_TESTING "Enable building third_party tests" ON)

add_subdirectory(third_party)

if(BUILD_FLATCC)
    add_compile_definitions(BUILD_FLATCC=1)
    add_subdirectory(schemas)
endif()

add_subdirectory(src/tsetlin_machine_c)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
