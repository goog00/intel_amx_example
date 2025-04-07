### 基于Intel AMX指令的矩阵乘法优化过程

Intel® AMX（Advanced Matrix Extensions）是x86指令集的新扩展，专为矩阵运算设计，能大幅加速AI工作负载中的矩阵乘法。这里简单介绍一下AMX指令的特性,详情参加[官网](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=6865,6866,6859,6872,6873,6863,6864,6863,6863,6873,6863,6873,6874,6861,6862,6874,6873,6876,6872,6875,6875,6866,6877,6877,6866,6877,6883,6881&techs=AMX)：

1. **基于Tile的架构**
   * AMX 引入了tile寄存器（TMM0–TMM7），每个 tile 是一个二维矩阵寄存器（最多 8 个）。
   * 每个 tile 的大小（行数、列数）是可配置的，通过 LDTILECFG 指令进行配置。
   * 支持的数据类型包括：`bf16`、`int8`、`int16`、`fp16`、`fp32`（部分取决于具体的 AMX 扩展版本，如 AMX-BF16、AMX-INT8 等）。
2. **Tile 配置灵活**
   * 可动态配置 tile 的维度与行为，每个 tile 最多可容纳 1024 个元素（例如 16x64）。
   * Tile 配置通过 LDTILECFG 加载，可通过 STTILECFG 存储。

用户可以通过本文提供的[代码](https://github.com/goog00/intel_amx_example)学习使用AMX指令，项目地址：<https://github.com/goog00/intel_amx_example>，也可以通过[官方示例](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)学习。

本文通过不断提高计算访存比（计算强度），一步步优化矩阵乘法实现，最大化AMX硬件的性能来提高处理矩阵乘的效率。

---

#### 第1版：基础实现，单次矩阵乘法

在第1版中(参见matrix\_mul\_amx\_with\_policy\_v1.cpp)，我们实现了一个基本的矩阵乘法，利用AMX的 `_tile_loadd`、`_tile_dpbssd` 和 `_tile_stored` 指令完成计算。

```c++
void MatrixMultiply(Matrix<InputType> &A, Matrix<InputType> &B, Matrix<OutputType> &C) {
    _tile_loadd(2, A.Data(), A.Stride());
    _tile_loadd(3, B.Data(), B.Stride());
    _tile_loadd(1, C.Data(), C.Stride());
    _tile_dpbssd(1, 2, 3);
    _tile_stored(1, C.Data(), C.Stride());
}
```

**硬件配置：** Intel(R) Xeon(R) Platinum 8458P, 4核4G
**特点与瓶颈：**

* **简单直接**：加载矩阵A、B到tile寄存器，执行一次乘加运算（`_tile_dpbssd`），结果存回C。
* **硬件利用不足**：AMX支持8个tile寄存器（0-7），但此实现仅使用了3个（tile 1、2、3），大量计算资源被浪费。
* **访存效率低**：每次只处理一对输入矩阵，访存带宽和计算并行性未被充分利用。

**性能数据：**

```
循环次数: 10000000
Intel Amx cost time:0.479877s, GOPS: 682.8410GOPS
```

这里GOPS表示每秒多少G的int8算数， 对应float的GFLOPS， 本文中的测试数据类型为int8类型。

**优化方向**：如何利用更多tile寄存器并增加输入数据的处理能力？

---

#### 第2版 tile寄存器分块， 提升A和B的计算强度

在第2版中(参见matrix\_mul\_amx\_with\_policy\_v2.cpp)，我们引入了分块矩阵乘法，将输入矩阵分为2×2的子块（A0、A1、B0、B1），输出4个结果子矩阵（C00、C01、C10、C11），充分利用AMX的8个tile寄存器。

```c++
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

    _tile_dpbssd(4, 0, 1); // C00 += A0 * B0
    _tile_stored(4, C00.Data(), C00.Stride());
    _tile_dpbssd(5, 0, 3); // C01 += A0 * B1
    _tile_stored(5, C01.Data(), C01.Stride());
    _tile_dpbssd(6, 2, 1); // C10 += A1 * B0
    _tile_stored(6, C10.Data(), C10.Stride());
    _tile_dpbssd(7, 2, 3); // C11 += A1 * B1
    _tile_stored(7, C11.Data(), C11.Stride());
}
```

**硬件配置：** Intel(R) Xeon(R) Platinum 8458P, 4核4G
**优化亮点：**

* **tile寄存器分块**：A0/1, B0/1四个tile寄存器只读取了一次，且两次参与到了计算中去， 其计算强度得到了提升，

**瓶颈：**

* **C的计算强度低**：每次计算都需要对C进行读取，计算强度太低

**性能数据：**

```terminal
循环次数: 10000000
Intel Amx cost time:1.03217s, GOPS: 1269.8740GOPS
```

**优化方向**：C的计算强度是否也可以得到提升？

---

#### 第3版：增加k纬度的长度， 提升C的计算强度

第3版(参见matrix\_mul\_amx\_with\_policy\_v3.cpp)， 根据矩阵乘法的特性 C[MxN] = A[MxK] * B[KxN]，我们可以知道， 增加K的长度并不会改变C的尺寸， 所以我们在A/B的tile（小矩阵）上增加了K纬度的长度。

```c++
void MatrixMultiply(std::vector<Matrix<InputType>> &VA, std::vector<Matrix<InputType>> &VB, Matrix<OutputType> &C) {
    _tile_loadd(0, C.Data(), C.Stride());
    for (int i = 0; i < VA.size(); i++) {
        _tile_loadd(1, VA[i].Data(), VA[i].Stride());
        _tile_loadd(2, VB[i].Data(), VB[i].Stride());
        _tile_dpbssd(0, 1, 2); // C += VA[i] * VB[i]
    }
    _tile_stored(0, C.Data(), C.Stride());
}
```

**硬件配置：** Intel(R) Xeon(R) Platinum 8458P, 4核4G
**优化亮点：**

* **极大的提升了C的计算强度**：循环开始前读取一次C，每次在k纬度的迭代加载一对输入矩阵（VA[i]、VB[i]），结果累加到tile 0， 循环结束后再取回C， 当K取一个合适的值时，C的计算强度我会得到极大的提升

**瓶颈：**

* 该方案没有考虑到A和B的计算强度

**性能数据：**

```terminal
循环次数: 10000000
Intel Amx cost time:3.94255s, GOPS: 1329.8203GOPS
```

**优化方向**：结合第2版的分块思想和第3版的累加能力，进一步提升性能。

---

#### 第4版：极致优化，最大化硬件利用

第4版，参见matrix\_mul\_amx\_with\_policy\_v4.cpp，该版本我们融合了前两版的策略，使A，B，C的计算强度都得到了提升

```
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
```

**硬件配置：** Intel(R) Xeon(R) Platinum 8458P, 4核4G
**优化亮点：**

* **提升了A, B的计算强度**：使用满了8个tile寄存器，最大化提升A， B的计算强度
* **提升了C的计算强度**：增加了k纬度的长度， 极大提升了C的计算强度

**性能数据：**

```terminal
循环次数: 10000000
Intel Amx cost time:9.76914s, GOPS: 2146.7103GOPS
```

**性能提升：**
相比第1版，第4版通过分块、累加和高效访存，性能从第1版的682 GOPS到2146 GOPS，翻了3倍多。

---

#### 第5版：多线程场景下的进一步优化

在第5版中（参考代码：matrix\_mul\_amx\_with\_policy\_v5.cpp）， 我们在第4版的分块矩阵乘法基础上，通过异步线程并行执行矩阵运算任务。每个线程独立处理部分计算工作，把所有的intel amx运算单元利用起来

**硬件配置：** Intel(R) Xeon(R) Platinum 8458P, 4核4G

**性能数据：**

```terminal
线程 1 - 循环次数: 5000000
执行时间: 5.6920 秒, 性能: 1842.2052 GOPS
线程 2 - 循环次数: 5000000
执行时间: 5.6874 秒, 性能: 1843.6698 GOPS
总执行时间: 5.6921 秒
时间差 (线程1 + 线程2 - 总时间): 5.6873 秒
总性能: 3684.3322 GOPS
```

在64核平台上，我们可以获得40T的int8算力

```terminal
执行时间: 0.5042 秒, 性能: 649.8689 GOPS
线程 59 - 循环次数: 156250
执行时间: 0.4846 秒, 性能: 676.2303 GOPS
线程 60 - 循环次数: 156250
执行时间: 0.4982 秒, 性能: 657.7000 GOPS
线程 61 - 循环次数: 156250
执行时间: 0.5058 秒, 性能: 647.8485 GOPS
线程 62 - 循环次数: 156250
执行时间: 0.5047 秒, 性能: 649.2992 GOPS
线程 63 - 循环次数: 156250
执行时间: 0.5011 秒, 性能: 653.8643 GOPS
总执行时间: 0.5204 秒
总性能: 40302.2399 GOPS
```

#### 总结与展望

从第1版到第5版的优化过程，体现了如何围绕AMX硬件特性逐步提升性能的关键思路：

1. **充分利用tile寄存器**：从3个tile到8个tile，硬件资源利用率大幅提升。
2. **减少访存开销**：从频繁存储到一次性存回，优化了内存带宽。
3. **增加计算并行性**：通过分块和累加，最大化AMX的计算吞吐量。

未来，可以进一步探索动态tile配置、异步访存与计算重叠等技术，继续挖掘AMX的潜力。希望这篇文章能为你的矩阵运算优化提供启发！如果有更多问题，欢迎留言讨论。

### 编译命令

#### matrix_mul_amx.cpp

clang++ -O3  -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing matrix_mul_amx.cpp -o ../build/matrix_mul_amx

#### matrix_mul_amx_with_policy.cpp

clang++ -O3  -march=native -mamx-tile -mamx-int8 -fno-strict-aliasing matrix_mul_amx_with_policy.cpp -o ../build/matrix_mul_amx_with_policy
