intel_amx_example

### 编译命令

clang++ -O3  -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing matrix_mul_amx.cpp -o matrix_mul_amx
