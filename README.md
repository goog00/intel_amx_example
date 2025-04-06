intel_amx_example

### 编译命令

matrix_mul_amx.cpp
clang++ -O3  -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing matrix_mul_amx.cpp -o ../build/matrix_mul_amx

matrix_mul_amx_with_policy.cpp
clang++ -O3  -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing matrix_mul_amx_with_policy.cpp -o ../build/matrix_mul_amx_with_policy
