cmake_minimum_required(VERSION 3.22)

project(khmh LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(khmh
    src/main.cu
)

target_include_directories(khmh PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(khmh PROPERTIES
    CUDA_ARCHITECTURES 61
)

target_compile_options(khmh PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -lineinfo>
)

find_library(NETCDF_CXX4_LIBRARY netcdf_c++4 REQUIRED)

target_link_libraries(khmh PRIVATE
    ${NETCDF_CXX4_LIBRARY}
)