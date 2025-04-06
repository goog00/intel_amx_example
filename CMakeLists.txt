cmake_minimum_required(VERSION 3.16)
project(IntelAMXExample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 启用 AMX/ 支持的编译器选项
add_compile_options(-O3 -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing)

# 添加可执行文件
add_executable(matrix_mul_amx src/matrix_mul_amx.cpp)
add_executable(matrix_mul_amx_with_policy src/matrix_mul_amx_with_policy.cpp)