#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>

/*
编译：
clang++ -O0 -g -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing int8mmamx.cpp  -o int8mmamx
执行：./int8mmamx

*/


/*
Two palettes are supported; palette 0 represents the initialized state, and palette 1 consists of
8 KB of storage spread across 8 tile registers named TMM0..TMM7. Each tile has a maximum size of 16 rows x 64
bytes, (1 KB), however the programmer can configure each tile to smaller dimensions appropriate to their algorithm. 
The tile dimensions supplied by the programmer (rows and bytes_per_row, i.e., colsb) are metadata that
drives the execution of tile and accelerator instructions
 */
size_t MAX = 1024;
int MAX_ROWS = 8; // 最大为16， 8x8 矩阵的行数设置为8
int MAX_COLS = 32; // 最大为64，由于_tile_dpbssd 最低计算 4个int_8 ,所以这里设置为32字节
int ARCH_GET_XCOMP_PERM = 0x1022;
int ARCH_REQ_XCOMP_PERM = 0x1023;
int XFEATURE_XTILECFG = 17;
int XFEATURE_XTILEDATA = 18;

// TIlE配置结构体
// AMX（Advanced Matrix Extensions）在硬件上支持 8 个 tile 寄存器，编号从 tmm0 到 tmm7。
struct __tile_config {
  uint8_t palette_id; // 配置模式:0 1
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16]; // 每个 tile 的列字节数
  
  // rows[i] 每个 tile 的行数:
  // rows[i] 的确是第 i 个 tile 的行数，范围是 0 到
  // 255（uint8_t 的取值范围） AMX 当前支持 8 个 tile（tmm0 到
  // tmm7），所以只用前 8 个元素，后 8 个预留。 如，rows[0] = 8 表示 tile 0 有 8 行。
  uint8_t rows[16]; // 每个 tile 的行数
};

bool set_tiledata_use() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    std::cerr << "Failed to enable AMX tile data\n";
    return false;
  } else {
    printf("\n TILE DATA USE SET - OK \n\n");
    return true;
  }
  return true;
}

void init_tile_config(__tile_config *tileinfo) {

  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  /*
  void _tile_loadconfig (const void * mem_addr)
  Load tile configuration from a 64-byte memory location specified by mem_addr. 
  The tile configuration format is specified below, and includes the tile type pallette, 
  the number of bytes per row, and the number of rows.
  */
  for (i = 1; i < 4; ++i) {
    tileinfo->colsb[i] = MAX_COLS; // 由于_tile_dpbssd 最低计算 4个int_8 ,所以这里设置为32字节
    tileinfo->rows[i] = MAX_ROWS;
  }

  _tile_loadconfig(tileinfo);
}

void matrix_multiply_8x8_amx(int8_t *A, int8_t *B, int32_t *C) {

  __tile_config tile_data = {0};
  if (!set_tiledata_use()) {
    exit(-1);
  }

  init_tile_config(&tile_data);
  // 加载矩阵 A B C 到 tile 
  _tile_loadd(2, A, 32);
  _tile_loadd(3, B, 32);
  _tile_loadd(1, C, 32); // 由于每次计算最小单位是 4 个 `int8_t`，为了确保内存访问和硬件计算的对齐和效率，这里将 `stride` 设置为 32 字节，确保每个 tile 行的数据在内存中对齐，以提高计算效率。


  // 打印tile的内容
  // 循环问题
  // 

  // 执行矩阵乘法 
  /*
  void _tile_dpbssd (constexpr int dst, constexpr int a, constexpr int b)

  Compute dot-product of bytes in tiles with a source/destination accumulator. 
  Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit integers in b, 
  producing 4 intermediate 32-bit results. Sum these 4 results with the corresponding 32-bit integer in dst,
  and store the 32-bit result back to tile dst.
  */
  //最小计算单位就是 4 个 int8_t（有符号 8 位整数） 作为一组，进行 dot-product 运算
  _tile_dpbssd(1, 2, 3);
  // 将结果从 tile 中存储到 C 矩阵
  _tile_stored(1, C, 32);

  _tile_release();
}


/* Initialize int8_t buffer */
static void init_buffer(int8_t *buf, int8_t value) {
  int rows, colsb, i, j;
  rows = MAX_ROWS;
  colsb = MAX_COLS;

  for (i = 0; i < rows; i++)
    for (j = 0; j < colsb; j++) {
      buf[i * colsb + j] = value;
    }
}

/* Initialize int32_t buffer */
static void init_buffer32(int32_t *buf, int32_t value) {
  int rows, colsb, i, j;
  rows = MAX_ROWS;
  colsb = MAX_COLS;
  int colsb2 = colsb / 4;

  for (i = 0; i < rows; i++)
    for (j = 0; j < (colsb2); j++) {
      buf[i * colsb2 + j] = value;
    }
}
/* Print int8_t buffer */
static void print_buffer(int8_t *buf, int32_t rows, int32_t colsb) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < (colsb); j++) {
      printf("%d ", buf[i * colsb + j]);
    }
    printf("\n");
  }
  printf("\n");
}

/* Print int32_t buffer */
static void print_buffer32(int32_t *buf, int32_t rows, int32_t colsb) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < (colsb); j++) {
      printf("%d ", buf[i * colsb + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {

  // 存在栈上
  int8_t A[MAX];
  int8_t B[MAX];
  int32_t C[MAX / 4];
  // int rows = 8;
  // int colsb = 8;
  int rows  = MAX_ROWS;
  int colsb = MAX_COLS;

  

  init_buffer(A, 1);
  print_buffer(A, rows, colsb);

  init_buffer(B, 1);
  print_buffer(B, rows, colsb);

  init_buffer32(C, 0);

  matrix_multiply_8x8_amx(A, B, C);

  print_buffer32(C, rows, 32 / 4);
  
  return 0;
}