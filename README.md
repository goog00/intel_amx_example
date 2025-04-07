### 基于Intel AMX指令的矩阵乘法优化过程

Intel® AMX（Advanced Matrix Extensions）是x86指令集的新扩展，专为矩阵运算设计，能大幅加速AI工作负载中的矩阵乘法。这里简单介绍一下AMX指令的特性,详情参加[官网](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig\_expand=6865,6866,6859,6872,6873,6863,6864,6863,6863,6873,6863,6873,6874,6861,6862,6874,6873,6876,6872,6875,6875,6866,6877,6877,6866,6877,6883,6881&techs=AMX)：

1. \*\*基于Tile的架构\*\*

- AMX 引入了tile寄存器（TMM0–TMM7），每个 tile 是一个二维矩阵寄存器（最多 8 个）。
- 每个 tile 的大小（行数、列数）是可配置的，通过 LDTILECFG 指令进行配置。
- 支持的数据类型包括：\`bf16\`、\`int8\`、\`int16\`、\`fp16\`、\`fp32\`（部分取决于具体的 AMX 扩展版本，如 AMX-BF16、AMX-INT8 等）。

2. \*\*Tile 配置灵活\*\*

- 可动态配置 tile 的维度与行为，每个 tile 最多可容纳 1024 个元素（例如 16x64）。
- Tile 配置通过 LDTILECFG 加载，可通过 STTILECFG 存储。

用户可以通过本文提供的[代码](https://github.com/goog00/intel\\\_amx\\\_example)学习使用AMX指令，项目地址：https://github.com/goog00/intel\_amx\_example，也可以通过[官方示例](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)学习。

本文通过不断提高访存带宽和计算并行性，一步步优化矩阵乘法实现，最大化AMX硬件的性能来提高处理矩阵乘的效率。

---

#### 第1版：基础实现，单次矩阵乘法

在第1版中(参见matrix\_mul\_amx\_with\_policy\_v1.cpp)，我们实现了一个基本的矩阵乘法，利用AMX的 \`\_tile\_loadd\`、\`\_tile\_dpbssd\` 和 \`\_tile\_stored\` 指令完成计算。

\`\`\`cpp

void MatrixMultiply(Matrix<InputType> &A, Matrix<InputType> &B, Matrix<OutputType> &C) {

\_tile\_loadd(2, A.Data(), A.Stride());

\_tile\_loadd(3, B.Data(), B.Stride());

\_tile\_loadd(1, C.Data(), C.Stride());

\_tile\_dpbssd(1, 2, 3);

\_tile\_stored(1, C.Data(), C.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*特点与瓶颈：\*\*

- \*\*简单直接\*\*：加载矩阵A、B到tile寄存器，执行一次乘加运算（\`\_tile\_dpbssd\`），结果存回C。
- \*\*硬件利用不足\*\*：AMX支持8个tile寄存器（0-7），但此实现仅使用了3个（tile 1、2、3），大量计算资源被浪费。
- \*\*访存效率低\*\*：每次只处理一对输入矩阵，访存带宽和计算并行性未被充分利用。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:0.479877s, GFLOPs: 682.8410gflops

\`\`\`

\*\*优化方向\*\*：如何利用更多tile寄存器并增加输入数据的处理能力？

---

#### 第2版：分块计算，利用更多tile寄存器

在第2版中(参见matrix\_mul\_amx\_with\_policy\_v2.cpp)，我们引入了分块矩阵乘法，将输入矩阵分为2×2的子块（A0、A1、B0、B1），输出4个结果子矩阵（C00、C01、C10、C11），充分利用AMX的8个tile寄存器。

\`\`\`cpp

void MatrixMultiply(Matrix<InputType> &A0, Matrix<InputType> &A1, Matrix<InputType> &B0, Matrix<InputType> &B1,

Matrix<OutputType> &C00, Matrix<OutputType> &C01, Matrix<OutputType> &C10, Matrix<OutputType> &C11) {

\_tile\_loadd(0, A0.Data(), A0.Stride());

\_tile\_loadd(1, B0.Data(), B0.Stride());

\_tile\_loadd(2, A1.Data(), A1.Stride());

\_tile\_loadd(3, B1.Data(), B1.Stride());

\_tile\_loadd(4, C00.Data(), C00.Stride());

\_tile\_loadd(5, C01.Data(), C01.Stride());

\_tile\_loadd(6, C10.Data(), C10.Stride());

\_tile\_loadd(7, C11.Data(), C11.Stride());

\_tile\_dpbssd(4, 0, 1); // C00 += A0 \* B0

\_tile\_stored(4, C00.Data(), C00.Stride());

\_tile\_dpbssd(5, 0, 3); // C01 += A0 \* B1

\_tile\_stored(5, C01.Data(), C01.Stride());

\_tile\_dpbssd(6, 2, 1); // C10 += A1 \* B0

\_tile\_stored(6, C10.Data(), C10.Stride());

\_tile\_dpbssd(7, 2, 3); // C11 += A1 \* B1

\_tile\_stored(7, C11.Data(), C11.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*优化亮点：\*\*

- \*\*tile寄存器全利用\*\*：使用了全部8个tile（0-7），分别存储A0、A1、B0、B1和C的4个子块。
- \*\*分块并行\*\*：通过分块计算，同时处理多个子矩阵乘法，提升了计算并行性。

\*\*瓶颈：\*\*

- \*\*单次计算\*\*：每个子矩阵仅执行一次乘加运算，无法累加多次计算结果。
- \*\*访存开销高\*\*：每次计算后立即存储结果（\`\_tile\_stored\`），增加了访存次数。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:1.03217s, GFLOPS: 1269.8740gflops

\`\`\`

\*\*优化方向\*\*：能否支持多次累加计算，减少存储开销？

---

#### 第3版：支持累加，减少访存

第3版(参见matrix\_mul\_amx\_with\_policy\_v3.cpp)，引入了输入矩阵的向量（\`std::vector<Matrix>\`），支持多次矩阵乘法累加到同一结果矩阵C上。

\`\`\`cpp

void MatrixMultiply(std::vector<Matrix<InputType>> &VA, std::vector<Matrix<InputType>> &VB, Matrix<OutputType> &C) {

\_tile\_loadd(0, C.Data(), C.Stride());

for (int i = 0; i < VA.size(); i++) {

\_tile\_loadd(1, VA[i].Data(), VA[i].Stride());

\_tile\_loadd(2, VB[i].Data(), VB[i].Stride());

\_tile\_dpbssd(0, 1, 2); // C += VA[i] \* VB[i]

}

\_tile\_stored(0, C.Data(), C.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*优化亮点：\*\*

- \*\*累加计算\*\*：通过循环，每次加载一对输入矩阵（VA[i]、VB[i]），结果累加到tile 0，最终一次性存回。
- \*\*访存优化\*\*：减少了中间结果的存储操作，仅在最后执行一次\`\_tile\_stored\`。

\*\*瓶颈：\*\*

- \*\*tile利用率下降\*\*：仅使用了3个tile寄存器（0-2），未充分利用AMX的硬件资源。
- \*\*单输出限制\*\*：只支持一个输出矩阵C，无法处理分块输出。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:3.94255s, GFLOPS: 1329.8203gflops

\`\`\`

\*\*优化方向\*\*：结合第2版的分块思想和第3版的累加能力，进一步提升性能。

---

#### 第4版：极致优化，最大化硬件利用

第4版，参见matrix\_mul\_amx\_with\_policy\_v4.cpp，融合了前几版的优点，支持分块输入矩阵的向量（VA0、VA1、VB0、VB1），输出4个子矩阵，同时实现累加计算和最大化的tile利用率。

\`\`\`cpp

void MatrixMultiply(std::vector<Matrix<InputType>> &VA0, std::vector<Matrix<InputType>> &VA1, std::vector<Matrix<InputType>> &VB0, std::vector<Matrix<InputType>> &VB1,

Matrix<OutputType> &C00, Matrix<OutputType> &C01, Matrix<OutputType> &C10, Matrix<OutputType> &C11) {

\_tile\_loadd(4, C00.Data(), C00.Stride());

\_tile\_loadd(5, C01.Data(), C01.Stride());

\_tile\_loadd(6, C10.Data(), C10.Stride());

\_tile\_loadd(7, C11.Data(), C11.Stride());

for (size\_t k = 0; k < VA0.size(); ++k) {

\_tile\_loadd(0, VA0[k].Data(), VA0[k].Stride()); // A00(:,k)

\_tile\_loadd(1, VB0[k].Data(), VB0[k].Stride()); // B00(k,:)

\_tile\_loadd(2, VA1[k].Data(), VA1[k].Stride()); // A10(:,k)

\_tile\_loadd(3, VB1[k].Data(), VB1[k].Stride()); // B01(k,:)

// 外积方式进行 accumulate

\_tile\_dpbssd(4, 0, 1); // C00 += A00(:,k) \* B00(k,:)

\_tile\_dpbssd(5, 0, 3); // C01 += A00(:,k) \* B01(k,:)

\_tile\_dpbssd(6, 2, 1); // C10 += A10(:,k) \* B00(k,:)

\_tile\_dpbssd(7, 2, 3); // C11 += A10(:,k) \* B01(k,:)

}

// 最后一次性存回

\_tile\_stored(4, C00.Data(), C00.Stride());

\_tile\_stored(5, C01.Data(), C01.Stride());

\_tile\_stored(6, C10.Data(), C10.Stride());

\_tile\_stored(7, C11.Data(), C11.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*优化亮点：\*\*

- \*\*全tile利用\*\*：使用8个tile寄存器，分别存储4个输入子矩阵和4个输出子矩阵。
- \*\*累加计算\*\*：支持多次外积累加，充分利用AMX的乘加能力。
- \*\*访存带宽最大化\*\*：通过批量加载输入矩阵向量，减少访存瓶颈，提升数据吞吐量。
- \*\*分块并行\*\*：同时计算4个输出子矩阵，最大化计算并行性。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:9.76914s, GFLOPS: 2146.7103gflops

\`\`\`

\*\*性能提升：\*\*

相比第1版，第4版通过分块、累加和高效访存，性能从第1版的682 GFLOPS到2146 GFLOPS，翻了3倍多。

---

#### 第5版：多线程场景下的进一步优化

在第5版中（参考代码：matrix\_mul\_amx\_with\_policy\_v5.cpp），我们在第4版的分块矩阵乘法基础上，通过异步线程并行执行矩阵运算任务。每个线程独立处理部分计算工作，充分利用CPU的多核特性。核心代码逻辑保持不变，但通过线程划分提升了整体吞吐量。

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*性能数据：\*\*

\`\`\`

线程 1 - 循环次数: 5000000

执行时间: 5.6920 秒, 性能: 1842.2052 GFLOPS

线程 2 - 循环次数: 5000000

执行时间: 5.6874 秒, 性能: 1843.6698 GFLOPS

总执行时间: 5.6921 秒

时间差 (线程1 + 线程2 - 总时间): 5.6873 秒

总性能: 3684.3322 GFLOPS

\`\`\`

\*\*优化亮点：\*\*

1. \*\*多线程并行\*\*：

- 通过引入两个异步线程，充分利用4核CPU的并行能力，实现任务分担。
- 线程1和线程2分别独立执行5,000,000次矩阵乘法循环，计算负载均衡。

2. \*\*高效重叠执行\*\*：

- 总执行时间（5.5788秒）接近单个线程的最长执行时间（5.5787秒），表明线程间计算与访存操作实现了高效重叠，几乎无显著的同步开销。

3. \*\*性能翻倍\*\*：

- 单线程性能分别为1932.90 GFLOPS和1879.60 GFLOPS，总性能达到3759.12 GFLOPS，接近两线程性能之和，显示出良好的线性扩展性，同时性能接近两倍第4版。

#### 总结与展望

从第1版到第5版的优化过程，体现了如何围绕AMX硬件特性逐步提升性能的关键思路：

1. \*\*充分利用tile寄存器\*\*：从3个tile到8个tile，硬件资源利用率大幅提升。
2. \*\*减少访存开销\*\*：从频繁存储到一次性存回，优化了内存带宽。
3. \*\*增加计算并行性\*\*：通过分块和累加，最大化AMX的计算吞吐量。

未来，可以进一步探索动态tile配置、异步访存与计算重叠等技术，继续挖掘AMX的潜力。希望这篇文章能为你的矩阵运算优化提供启发！如果有更多问题，欢迎留言讨论。

### 基于Intel AMX指令的矩阵乘法优化过程

Intel® AMX（Advanced Matrix Extensions）是x86指令集的新扩展，专为矩阵运算设计，能大幅加速AI工作负载中的矩阵乘法。这里简单介绍一下AMX指令的特性,详情参加[官网](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig\_expand=6865,6866,6859,6872,6873,6863,6864,6863,6863,6873,6863,6873,6874,6861,6862,6874,6873,6876,6872,6875,6875,6866,6877,6877,6866,6877,6883,6881&techs=AMX)：

1. \*\*基于Tile的架构\*\*

- AMX 引入了tile寄存器（TMM0–TMM7），每个 tile 是一个二维矩阵寄存器（最多 8 个）。
- 每个 tile 的大小（行数、列数）是可配置的，通过 LDTILECFG 指令进行配置。
- 支持的数据类型包括：\`bf16\`、\`int8\`、\`int16\`、\`fp16\`、\`fp32\`（部分取决于具体的 AMX 扩展版本，如 AMX-BF16、AMX-INT8 等）。

2. \*\*Tile 配置灵活\*\*

- 可动态配置 tile 的维度与行为，每个 tile 最多可容纳 1024 个元素（例如 16x64）。
- Tile 配置通过 LDTILECFG 加载，可通过 STTILECFG 存储。

用户可以通过本文提供的[代码](https://github.com/goog00/intel\_amx\_example)学习使用AMX指令，项目地址：https://github.com/goog00/intel_amx_example，也可以通过[官方示例](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)学习。

本文通过不断提高访存带宽和计算并行性，一步步优化矩阵乘法实现，最大化AMX硬件的性能来提高处理矩阵乘的效率。

---

#### 第1版：基础实现，单次矩阵乘法

在第1版中(参见matrix\_mul\_amx\_with\_policy\_v1.cpp)，我们实现了一个基本的矩阵乘法，利用AMX的 \`\_tile\_loadd\`、\`\_tile\_dpbssd\` 和 \`\_tile\_stored\` 指令完成计算。

\`\`\`cpp

void MatrixMultiply(Matrix<InputType> &A, Matrix<InputType> &B, Matrix<OutputType> &C) {

\_tile\_loadd(2, A.Data(), A.Stride());

\_tile\_loadd(3, B.Data(), B.Stride());

\_tile\_loadd(1, C.Data(), C.Stride());

\_tile\_dpbssd(1, 2, 3);

\_tile\_stored(1, C.Data(), C.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*特点与瓶颈：\*\*

- \*\*简单直接\*\*：加载矩阵A、B到tile寄存器，执行一次乘加运算（\`\_tile\_dpbssd\`），结果存回C。
- \*\*硬件利用不足\*\*：AMX支持8个tile寄存器（0-7），但此实现仅使用了3个（tile 1、2、3），大量计算资源被浪费。
- \*\*访存效率低\*\*：每次只处理一对输入矩阵，访存带宽和计算并行性未被充分利用。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:0.479877s, GFLOPs: 682.8410gflops

\`\`\`

\*\*优化方向\*\*：如何利用更多tile寄存器并增加输入数据的处理能力？

---

#### 第2版：分块计算，利用更多tile寄存器

在第2版中(参见matrix\_mul\_amx\_with\_policy\_v2.cpp)，我们引入了分块矩阵乘法，将输入矩阵分为2×2的子块（A0、A1、B0、B1），输出4个结果子矩阵（C00、C01、C10、C11），充分利用AMX的8个tile寄存器。

\`\`\`cpp

void MatrixMultiply(Matrix<InputType> &A0, Matrix<InputType> &A1, Matrix<InputType> &B0, Matrix<InputType> &B1,

Matrix<OutputType> &C00, Matrix<OutputType> &C01, Matrix<OutputType> &C10, Matrix<OutputType> &C11) {

\_tile\_loadd(0, A0.Data(), A0.Stride());

\_tile\_loadd(1, B0.Data(), B0.Stride());

\_tile\_loadd(2, A1.Data(), A1.Stride());

\_tile\_loadd(3, B1.Data(), B1.Stride());

\_tile\_loadd(4, C00.Data(), C00.Stride());

\_tile\_loadd(5, C01.Data(), C01.Stride());

\_tile\_loadd(6, C10.Data(), C10.Stride());

\_tile\_loadd(7, C11.Data(), C11.Stride());

\_tile\_dpbssd(4, 0, 1); // C00 += A0 \* B0

\_tile\_stored(4, C00.Data(), C00.Stride());

\_tile\_dpbssd(5, 0, 3); // C01 += A0 \* B1

\_tile\_stored(5, C01.Data(), C01.Stride());

\_tile\_dpbssd(6, 2, 1); // C10 += A1 \* B0

\_tile\_stored(6, C10.Data(), C10.Stride());

\_tile\_dpbssd(7, 2, 3); // C11 += A1 \* B1

\_tile\_stored(7, C11.Data(), C11.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*优化亮点：\*\*

- \*\*tile寄存器全利用\*\*：使用了全部8个tile（0-7），分别存储A0、A1、B0、B1和C的4个子块。
- \*\*分块并行\*\*：通过分块计算，同时处理多个子矩阵乘法，提升了计算并行性。

\*\*瓶颈：\*\*

- \*\*单次计算\*\*：每个子矩阵仅执行一次乘加运算，无法累加多次计算结果。
- \*\*访存开销高\*\*：每次计算后立即存储结果（\`\_tile\_stored\`），增加了访存次数。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:1.03217s, GFLOPS: 1269.8740gflops

\`\`\`

\*\*优化方向\*\*：能否支持多次累加计算，减少存储开销？

---

#### 第3版：支持累加，减少访存

第3版(参见matrix\_mul\_amx\_with\_policy\_v3.cpp)，引入了输入矩阵的向量（\`std::vector<Matrix>\`），支持多次矩阵乘法累加到同一结果矩阵C上。

\`\`\`cpp

void MatrixMultiply(std::vector<Matrix<InputType>> &VA, std::vector<Matrix<InputType>> &VB, Matrix<OutputType> &C) {

\_tile\_loadd(0, C.Data(), C.Stride());

for (int i = 0; i < VA.size(); i++) {

\_tile\_loadd(1, VA[i].Data(), VA[i].Stride());

\_tile\_loadd(2, VB[i].Data(), VB[i].Stride());

\_tile\_dpbssd(0, 1, 2); // C += VA[i] \* VB[i]

}

\_tile\_stored(0, C.Data(), C.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*优化亮点：\*\*

- \*\*累加计算\*\*：通过循环，每次加载一对输入矩阵（VA[i]、VB[i]），结果累加到tile 0，最终一次性存回。
- \*\*访存优化\*\*：减少了中间结果的存储操作，仅在最后执行一次\`\_tile\_stored\`。

\*\*瓶颈：\*\*

- \*\*tile利用率下降\*\*：仅使用了3个tile寄存器（0-2），未充分利用AMX的硬件资源。
- \*\*单输出限制\*\*：只支持一个输出矩阵C，无法处理分块输出。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:3.94255s, GFLOPS: 1329.8203gflops

\`\`\`

\*\*优化方向\*\*：结合第2版的分块思想和第3版的累加能力，进一步提升性能。

---

#### 第4版：极致优化，最大化硬件利用

第4版，参见matrix\_mul\_amx\_with\_policy\_v4.cpp，融合了前几版的优点，支持分块输入矩阵的向量（VA0、VA1、VB0、VB1），输出4个子矩阵，同时实现累加计算和最大化的tile利用率。

\`\`\`cpp

void MatrixMultiply(std::vector<Matrix<InputType>> &VA0, std::vector<Matrix<InputType>> &VA1, std::vector<Matrix<InputType>> &VB0, std::vector<Matrix<InputType>> &VB1,

Matrix<OutputType> &C00, Matrix<OutputType> &C01, Matrix<OutputType> &C10, Matrix<OutputType> &C11) {

\_tile\_loadd(4, C00.Data(), C00.Stride());

\_tile\_loadd(5, C01.Data(), C01.Stride());

\_tile\_loadd(6, C10.Data(), C10.Stride());

\_tile\_loadd(7, C11.Data(), C11.Stride());

for (size\_t k = 0; k < VA0.size(); ++k) {

\_tile\_loadd(0, VA0[k].Data(), VA0[k].Stride()); // A00(:,k)

\_tile\_loadd(1, VB0[k].Data(), VB0[k].Stride()); // B00(k,:)

\_tile\_loadd(2, VA1[k].Data(), VA1[k].Stride()); // A10(:,k)

\_tile\_loadd(3, VB1[k].Data(), VB1[k].Stride()); // B01(k,:)

// 外积方式进行 accumulate

\_tile\_dpbssd(4, 0, 1); // C00 += A00(:,k) \* B00(k,:)

\_tile\_dpbssd(5, 0, 3); // C01 += A00(:,k) \* B01(k,:)

\_tile\_dpbssd(6, 2, 1); // C10 += A10(:,k) \* B00(k,:)

\_tile\_dpbssd(7, 2, 3); // C11 += A10(:,k) \* B01(k,:)

}

// 最后一次性存回

\_tile\_stored(4, C00.Data(), C00.Stride());

\_tile\_stored(5, C01.Data(), C01.Stride());

\_tile\_stored(6, C10.Data(), C10.Stride());

\_tile\_stored(7, C11.Data(), C11.Stride());

}

\`\`\`

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*优化亮点：\*\*

- \*\*全tile利用\*\*：使用8个tile寄存器，分别存储4个输入子矩阵和4个输出子矩阵。
- \*\*累加计算\*\*：支持多次外积累加，充分利用AMX的乘加能力。
- \*\*访存带宽最大化\*\*：通过批量加载输入矩阵向量，减少访存瓶颈，提升数据吞吐量。
- \*\*分块并行\*\*：同时计算4个输出子矩阵，最大化计算并行性。

\*\*性能数据：\*\*

\`\`\`

循环次数: 10000000

Intel Amx cost time:9.76914s, GFLOPS: 2146.7103gflops

\`\`\`

\*\*性能提升：\*\*

相比第1版，第4版通过分块、累加和高效访存，性能从第1版的682 GFLOPS到2146 GFLOPS，翻了3倍多。

---

#### 第5版：多线程场景下的进一步优化

在第5版中（参考代码：matrix\_mul\_amx\_with\_policy\_v5.cpp），我们在第4版的分块矩阵乘法基础上，通过异步线程并行执行矩阵运算任务。每个线程独立处理部分计算工作，充分利用CPU的多核特性。核心代码逻辑保持不变，但通过线程划分提升了整体吞吐量。

\*\*硬件配置：\*\* Intel(R) Xeon(R) Platinum 8458P, 4核4G

\*\*性能数据：\*\*

\`\`\`

线程 1 - 循环次数: 5000000

执行时间: 5.6920 秒, 性能: 1842.2052 GFLOPS

线程 2 - 循环次数: 5000000

执行时间: 5.6874 秒, 性能: 1843.6698 GFLOPS

总执行时间: 5.6921 秒

时间差 (线程1 + 线程2 - 总时间): 5.6873 秒

总性能: 3684.3322 GFLOPS

\`\`\`

\*\*优化亮点：\*\*

1. \*\*多线程并行\*\*：

- 通过引入两个异步线程，充分利用4核CPU的并行能力，实现任务分担。
- 线程1和线程2分别独立执行5,000,000次矩阵乘法循环，计算负载均衡。

2. \*\*高效重叠执行\*\*：

- 总执行时间（5.5788秒）接近单个线程的最长执行时间（5.5787秒），表明线程间计算与访存操作实现了高效重叠，几乎无显著的同步开销。

3. \*\*性能翻倍\*\*：

- 单线程性能分别为1932.90 GFLOPS和1879.60 GFLOPS，总性能达到3759.12 GFLOPS，接近两线程性能之和，显示出良好的线性扩展性，同时性能接近两倍第4版。

#### 总结与展望

从第1版到第5版的优化过程，体现了如何围绕AMX硬件特性逐步提升性能的关键思路：

1. \*\*充分利用tile寄存器\*\*：从3个tile到8个tile，硬件资源利用率大幅提升。
2. \*\*减少访存开销\*\*：从频繁存储到一次性存回，优化了内存带宽。
3. \*\*增加计算并行性\*\*：通过分块和累加，最大化AMX的计算吞吐量。

未来，可以进一步探索动态tile配置、异步访存与计算重叠等技术，继续挖掘AMX的潜力。希望这篇文章能为你的矩阵运算优化提供启发！如果有更多问题，欢迎留言讨论。
