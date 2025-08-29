cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tesla_nvh_simple VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apple Silicon optimizations
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS " Configuring for Apple Silicon (ARM64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=apple-m1")
else()
    message(STATUS " Configuring for x86_64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

find_package(Torch REQUIRED)

add_executable(tesla_nvh_simple tesla_nvh_simple.cpp)
target_link_libraries(tesla_nvh_simple ${TORCH_LIBRARIES})

set_target_properties(tesla_nvh_simple PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

message(STATUS " Tesla NVH Simple - Apple Silicon Compatible")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
