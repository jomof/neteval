cmake_minimum_required(VERSION 3.25.1)
project(neteval)

set(CMAKE_CXX_STANDARD 26)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(APPLE_SILICON TRUE)
endif()
if (APPLE_SILICON)
    add_compile_options(-fopenmp -ffast-math -march=armv8-a+simd -fvectorize -fslp-vectorize)
else()
    add_compile_options(-fopenmp -ffast-math -ftree-vectorize)
endif()

# Add OpenMP linking for all platforms
find_package(OpenMP REQUIRED)
#if (OpenMP_CXX_FOUND)
#    add_compile_options(${OpenMP_CXX_FLAGS})
#    target_link_libraries(neteval PUBLIC OpenMP::OpenMP_CXX)
#endif()

add_executable(neteval main.cpp)
target_link_libraries(neteval PUBLIC OpenMP::OpenMP_CXX)
