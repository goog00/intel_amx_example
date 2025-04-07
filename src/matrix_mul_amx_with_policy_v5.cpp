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
#include <thread>
#include <mutex>


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


  void MatrixMultiply(std::vector<Matrix<InputType>> &VA0, std::vector<Matrix<InputType>> &VA1, std::vector<Matrix<InputType>> &VB0, std::vector<Matrix<InputType>> &VB1, 
                      Matrix<OutputType> &C00, Matrix<OutputType> &C01, Matrix<OutputType> &C10, Matrix<OutputType> &C11) {

                _tile_loadd(4, C00.Data(), C00.Stride());
                _tile_loadd(5, C01.Data(), C01.Stride());
                _tile_loadd(6, C10.Data(), C10.Stride());
                _tile_loadd(7, C11.Data(), C11.Stride());
                        
                for (size_t k = 0; k < VA0.size(); ++k) {
                  _tile_loadd(0, VA0[k].Data(), VA0[k].Stride()); // A00(:,k)
                  _tile_loadd(1, VB0[k].Data(), VB0[k].Stride()); // B00(k,:)

                  _tile_loadd(2, VA1[k].Data(), VA1[k].Stride()); // A10(:,k)
                  _tile_loadd(3, VB1[k].Data(), VB1[k].Stride()); // B01(k,:)

                  // 外积方式进行 accumulate
                  _tile_dpbssd(4, 0, 1); // C00 += A00(:,k) * B00(k,:)
                  _tile_dpbssd(5, 0, 3); // C01 += A00(:,k) * B01(k,:)
                  _tile_dpbssd(6, 2, 1); // C10 += A10(:,k) * B00(k,:)
                  _tile_dpbssd(7, 2, 3); // C11 += A10(:,k) * B01(k,:)
                }

                // 最后一次性存回
                _tile_stored(4, C00.Data(), C00.Stride());
                _tile_stored(5, C01.Data(), C01.Stride());
                _tile_stored(6, C10.Data(), C10.Stride());
                _tile_stored(7, C11.Data(), C11.Stride());
  
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

struct TestData {
  std::vector<Matrix<int8_t>> VA0;
  std::vector<Matrix<int8_t>> VB0;
  std::vector<Matrix<int8_t>> VA1;
  std::vector<Matrix<int8_t>> VB1;
  Matrix<int32_t> C00;
  Matrix<int32_t> C01;
  Matrix<int32_t> C10;
  Matrix<int32_t> C11;
  IntelAmxMatrixMultiply<int8_t, int32_t> multiply;

  TestData() 
      : C00(16, 16), C01(16, 16), C10(16, 16), C11(16, 16),
        multiply(IntelAmxMatrixMultiply<int8_t, int32_t>::Create()) {
      VA0.reserve(16);
      VB0.reserve(16);
      VA1.reserve(16);
      VB1.reserve(16);
  }
};

void run_test(int thread_id, int iterations, TestData& data, double& result_time, double& result_gflops) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
      data.multiply.MatrixMultiply(data.VA0, data.VA1, data.VB0, data.VB1,
                                   data.C00, data.C01, data.C10, data.C11);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  auto cost_time = static_cast<double>((t1 - t0).count());
  auto ops_per_matmul = int64_t(16) * 64 * 16 * 2 * 4 * 16; // 2097152
  auto total_flops = static_cast<double>(ops_per_matmul * iterations);
  auto gflops = total_flops / cost_time;

  result_time = cost_time / 1e9;
  result_gflops = gflops;
}

int main() {
  int total_iterations = 10000000;
  int thread_count = 128; // 按 32 核优化，最高配置
  int iterations_per_thread = total_iterations / thread_count;

  // 初始化测试数据
  std::vector<TestData> thread_data(thread_count);
  int rows = 16, cols = 64;
  for (int t = 0; t < thread_count; ++t) {
      for (int i = 0; i < 16; ++i) {
          thread_data[t].VA0.emplace_back(rows, cols); thread_data[t].VA0.back().Fill(2);
          thread_data[t].VB0.emplace_back(rows, cols); thread_data[t].VB0.back().Fill(2);
          thread_data[t].VA1.emplace_back(rows, cols); thread_data[t].VA1.back().Fill(2);
          thread_data[t].VB1.emplace_back(rows, cols); thread_data[t].VB1.back().Fill(2);
      }
      thread_data[t].C00.Fill(0);
      thread_data[t].C01.Fill(0);
      thread_data[t].C10.Fill(0);
      thread_data[t].C11.Fill(0);
  }

  // 多线程测试
  std::vector<std::thread> threads;
  std::vector<double> times(thread_count);
  std::vector<double> gflops(thread_count);

  auto total_start = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < thread_count; ++t) {
      threads.emplace_back(run_test, t, iterations_per_thread, std::ref(thread_data[t]),
                           std::ref(times[t]), std::ref(gflops[t]));
  }
  for (auto& t : threads) t.join();
  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_time = static_cast<double>((total_end - total_start).count()) / 1e9;

  // 打印结果
  for (int t = 0; t < thread_count; ++t) {
      std::cout << "线程 " << t << " - 循环次数: " << iterations_per_thread << "\n";
      std::cout << "执行时间: " << std::fixed << std::setprecision(4) << times[t]
                << " 秒, 性能: " << std::fixed << std::setprecision(4) << gflops[t] << " GFLOPS\n";
  }
  std::cout << "总执行时间: " << std::fixed << std::setprecision(4) << total_time << " 秒\n";

  // 计算总 GFLOPS
  auto ops_per_matmul = int64_t(16) * 64 * 16 * 2 * 4 * 16; // 2097152
  auto total_flops = static_cast<double>(ops_per_matmul * total_iterations);
  auto total_gflops = total_flops / (total_time * 1e9);
  std::cout << "总性能: " << std::fixed << std::setprecision(4) << total_gflops << " GFLOPS\n";

  for (auto& data : thread_data) data.multiply.TileRelease();
  return 0;
}