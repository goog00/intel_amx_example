#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>

template <typename DataType> 
class Matrix {

private:
  int rows;
  int cols;
  DataType *data;

public:
  Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data = new DataType[rows * cols];
  }
  ~Matrix() { delete[] data; }

  size_t Stride() { return this->cols * sizeof(DataType); }
  DataType *Data() const { return data; }
  int Rows() const { return rows; }
  int Cols() const { return cols; }

  // 用于初始化
  void Fill(DataType value) {
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = value;
    }
  }
};

struct __tile_config {
  uint8_t palette_id; // 配置模式:0 1
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16]; // 每个 tile 的列字节数
  uint8_t rows[16];   // 每个 tile 的行数
};

template <typename InputType, typename OutputType> 
class MatrixMultiply {
private:
  int ARCH_REQ_XCOMP_PERM = 0x1023;
  int XFEATURE_XTILEDATA = 18;
  int rows = 16;
  int colsb = 64;

  void init_tile_config(__tile_config *tileinfo) {

    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    // 疑问：这里是否应该是用多少申请多少？
    for (i = 1; i < 4; ++i) {
      tileinfo->colsb[i] = colsb;
      tileinfo->rows[i] = rows;
    }

    _tile_loadconfig(tileinfo);
  }

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

public:
  void matrix_multiply_amx(Matrix<InputType> &A, Matrix<InputType> &B,
                           Matrix<OutputType> &C) {

    __tile_config tile_data = {0};
    if (!set_tiledata_use()) {
      exit(-1);
    }

    init_tile_config(&tile_data);
    _tile_loadd(2, A.Data(), A.Stride());
    _tile_loadd(3, B.Data(), B.Stride());
    _tile_loadd(1, C.Data(), C.Stride());

    _tile_dpbssd(1, 2, 3);
    _tile_stored(1, C.Data(), C.Stride());

    _tile_release();
  }
};

// 测试代码
int main() {
  // 创建矩阵
  Matrix<int8_t> A(16, 64);
  Matrix<int8_t> B(16, 64);
  Matrix<int32_t> C(16, 16);

  // 初始化矩阵
  A.Fill(1);
  B.Fill(1);
  C.Fill(0);

  // 执行乘法
  MatrixMultiply<int8_t, int32_t> multiply;
  multiply.matrix_multiply_amx(A, B, C);

  // 打印结果（简单验证）
  for (int i = 0; i < C.Rows(); ++i) {
    for (int j = 0; j < C.Cols(); ++j) {
      std::cout << C.Data()[i * C.Cols() + j] << " ";
    }
    std::cout << "\n";
  }

  return 0;
}