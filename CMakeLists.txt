cmake_minimum_required(VERSION 3.23)

# Build instructions:
# 1) Configure:
#    cmake -B build
# 2) Build:
#    cmake --build build

project(green_tsetlin_to_trainable_C
    VERSION 1.0.0
)

option(ENABLE_FLATCC "Build FlatCC from third_party" ON)
option(BUILD_TESTING "Enable building third_party tests" ON)

if(BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(third_party)
