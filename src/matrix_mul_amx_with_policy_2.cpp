#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <cassert>
#include <memory>
#include <chrono>
#include <vector>

#include <iomanip>



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

  int Size() const { return rows * cols; }

  // 用于初始化
  void Fill(DataType value) {
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = value;
    }
  }

  // 打印矩阵
  void Print_t() const {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        std::cout << static_cast<int>(data[i * cols + j]) << " ";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
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
class IntelAmxMatrixMultiply {

private:

  IntelAmxMatrixMultiply() = default;

  void InitTileConfig() {
    __tile_config tileinfo{};
    tileinfo.palette_id = 1;
    tileinfo.start_row = 0;
    for (int i = 0; i < 8; ++i) {
      tileinfo.colsb[i] = COLSB;
      tileinfo.rows[i] = ROWS;
    }
    _tile_loadconfig(&tileinfo);
  }

  int ARCH_REQ_XCOMP_PERM = 0x1023;
  int XFEATURE_XTILEDATA = 18;
  int ROWS = 16;
  int COLSB = 64;


public:

  static IntelAmxMatrixMultiply Create() {
    IntelAmxMatrixMultiply self;
    self.SetTileDataUse();
    self.InitTileConfig();
    return self;
  }

  void MatrixMultiply(Matrix<InputType> &A0, Matrix<InputType> &A1, Matrix<InputType> &B0, Matrix<InputType> &B1, 
                      Matrix<OutputType> &C00, Matrix<OutputType> &C01, Matrix<OutputType> &C10, Matrix<OutputType> &C11) {


      _tile_loadd(0, A0.Data(), A0.Stride());
      _tile_loadd(1, B0.Data(), B0.Stride());

      _tile_loadd(2, A1.Data(), A1.Stride());
      _tile_loadd(3, B1.Data(), B1.Stride());

      _tile_loadd(4, C00.Data(), C00.Stride());
      _tile_loadd(5, C01.Data(), C01.Stride());
      _tile_loadd(6, C10.Data(), C10.Stride());
      _tile_loadd(7, C11.Data(), C11.Stride());

  
      _tile_dpbssd(4, 0, 1);
      _tile_stored(4, C00.Data(), C00.Stride());
  
      _tile_dpbssd(5, 0, 3);
      _tile_stored(5, C01.Data(), C01.Stride());

      _tile_dpbssd(6, 2, 1);
      _tile_stored(6, C10.Data(), C10.Stride());

      _tile_dpbssd(7, 2, 3);
      _tile_stored(7, C11.Data(), C11.Stride());


  
  }   

  void MatrixMultiply(std::vector<Matrix<InputType>> &VA, std::vector<Matrix<InputType>> &VB, Matrix<OutputType> &C) {
      // todo check VA VB size
      _tile_loadd(0, C.Data(), C.Stride());
      for (int i = 0 ; i < VA.size(); i++) {
        _tile_loadd(1, VA[i].Data(), VA[i].Stride());
        _tile_loadd(2, VB[i].Data(), VB[i].Stride());
        _tile_dpbssd(0, 1, 2);
      }
      _tile_stored(0, C.Data(), C.Stride());

  }
  


  bool SetTileDataUse() {
    auto res = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    assert(!res && "fail:Invoke syscall to set ARCH_SET_STATE_USE ");
    return true;
  }

  void TileRelease() {
    _tile_release();
  }

};

// 测试代码
int main() {


    // 创建输入矩阵向量
    std::vector<Matrix<int8_t>> VA;
    std::vector<Matrix<int8_t>> VB;
    VA.reserve(16);  
    VB.reserve(16);
    Matrix<int32_t> C(16, 16);
    C.Fill(0);

    // 初始化矩阵
    int rows = 16, cols = 64;
    for (int i = 0; i < 8; ++i) {
        VA.emplace_back(rows, cols); 
        VB.emplace_back(rows, cols); 

        VA.back().Fill(2); 
        VB.back().Fill(2); 
      
    }
  

    // 执行乘法
    auto multiply = IntelAmxMatrixMultiply<int8_t, int32_t>::Create();

    int k = 10000000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0 ; i < k; i++){
      multiply.MatrixMultiply(VA, VB, C);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    multiply.TileRelease();

    auto cost_time = static_cast<double>((t1 - t0).count());
    auto ops_per_matmul = int64_t(16) * 64 * 16 * 2;
    auto items  = static_cast<double>(ops_per_matmul * k * 8);
    auto gflops = items / cost_time ;
    std::cout <<"循环次数: " << k << "\n";
    std::cout <<"Intel Amx cost time:" << cost_time / 1e9 <<"s, GFLOPS: " << std::fixed << std::setprecision(4) << gflops << "gflops" << "\n";

    return 0;
}
