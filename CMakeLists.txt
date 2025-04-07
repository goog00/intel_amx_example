cmake_minimum_required(VERSION 3.16)
project(IntelAMXExample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 启用 AMX/ 支持的编译器选项
add_compile_options(-O3 -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing)

# 添加可执行文件
add_executable(matrix_mul_amx_with_policy_1 src/matrix_mul_amx_with_policy_v1.cpp)
add_executable(matrix_mul_amx_with_policy_2 src/matrix_mul_amx_with_policy_v2.cpp)
add_executable(matrix_mul_amx_with_policy_3 src/matrix_mul_amx_with_policy_v3.cpp)
add_executable(matrix_mul_amx_with_policy_4 src/matrix_mul_amx_with_policy_v4.cpp)
add_executable(matrix_mul_amx_with_policy_5 src/matrix_mul_amx_with_policy_v5.cpp)