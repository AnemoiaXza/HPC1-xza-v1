=======
# 实验一：基于 MPI 的矩阵运算并行化

## 实验要求

掌握 MPI 并行编程的基本方法，完成**矩阵相关运算**任务。可选内容：

- 矩阵加法
- 矩阵乘法
- 矩阵转置
- 分块矩阵乘法

**核心要求**：

1. 将矩阵数据划分到多个进程中并行计算
2. 理解数据分发、局部计算、结果汇总的完整过程
3. 实验报告中需说明：矩阵划分方式、通信方式、核心代码逻辑
4. **对比串行计算与并行计算的运行时间**

---

## 项目目录结构

```
实验一/
├── mpich-3.3.2.tar/            # MPICH 3.3.2 源码（已解压）
│   └── mpich-3.3.2/
│       ├── src/                # 核心源码（mpi, mpid, mpl, pmi 等）
│       ├── examples/           # MPI 示例（hellow.c, cpi.c 等）
│       ├── test/               # 测试套件
│       ├── doc/                # 安装与用户文档
│       └── configure           # autotools 构建脚本
├── OpenBLAS-0.3.8/             # OpenBLAS 0.3.8 源码（已完全解压）
│       ├── kernel/             # 各架构优化内核（x86_64, arm64 等）
│       ├── driver/             # 驱动层（level2/level3 BLAS）
│       ├── interface/          # 接口层（BLAS/LAPACK）
│       ├── lapack/             # LAPACK 线性代数例程
│       ├── benchmark/          # 性能测试脚本
│       └── Makefile            # 构建入口
└── README.md
```

---

## 软件说明

### MPICH 3.3.2 — MPI 并行通信库

MPICH 是阿贡国家实验室开发的高性能 MPI（Message Passing Interface）实现，支持 **MPI-3.1 标准**。

**功能**：让多个进程之间高效传递消息和协调工作，是 HPC 分布式并行计算的基础设施。本实验中所有进程间的数据分发、结果汇总都通过 MPICH 提供的通信函数完成。

核心 src 子模块：

| 目录 | 作用 |
|------|------|
| `mpi/` | MPI 标准接口实现 |
| `mpid/` | 设备抽象层（ch3 设备） |
| `pmi/` | 进程管理接口（与 Hydra 进程管理器配合） |
| `mpl/` | 可移植工具库 |

### OpenBLAS 0.3.8 — 高性能线性代数库

OpenBLAS 是优化的 BLAS（Basic Linear Algebra Subprograms）实现，为矩阵乘法等基础线性代数操作提供极致性能（利用 SIMD 指令集）。

**功能**：为单节点上的矩阵计算提供加速。在本实验中，每个进程内部的矩阵局部计算可受益于 OpenBLAS 的优化。

### 二者在本实验中的角色

- **MPICH**：负责跨进程的数据划分、通信与同步（进程间协作）
- **OpenBLAS**：负责单进程内的矩阵运算加速（进程内计算）
- 二者结合构成 **通信 + 计算** 的完整 HPC 并行计算环境

---

## 推荐学习顺序

### 第一阶段：环境搭建

1. **编译安装 MPICH**
   ```bash
   cd mpich-3.3.2.tar/mpich-3.3.2/
   ./configure --prefix=$HOME/mpich-install
   make -j$(nproc) && make install
   ```
   验证：`$HOME/mpich-install/bin/mpicc --version`

2. **编译安装 OpenBLAS**
   ```bash
   cd OpenBLAS-0.3.8/
   make -j$(nproc)
   ```

### 第二阶段：MPI 基础入门

3. **运行 MPICH 自带的 hellow 示例**，理解 `mpirun -np N` 的多进程执行模型
4. **学习 MPI 核心概念**：
   - `MPI_Init` / `MPI_Finalize`：环境初始化与退出
   - `MPI_Comm_rank` / `MPI_Comm_size`：获取进程编号与总数
   - `MPI_Send` / `MPI_Recv`：点对点通信
   - `MPI_Bcast` / `MPI_Scatter` / `MPI_Gather`：集合通信
   - `MPI_Barrier`：同步屏障
5. **自行编写一个简单的数据分发-汇总程序**，比如将一个数组 scatter 到各进程做局部加和再 gather 回来

### 第三阶段：矩阵运算实现

6. **选择矩阵运算题目**（建议先行后列拆分矩阵乘法）
7. **实现步骤**：
   - 分析矩阵如何按行/按列/按块划分到各进程
   - 选择通信模式（Scatter/Gather 或点对点）
   - 编写局部计算逻辑
   - 汇总结果并验证正确性
8. **对比实验**：
   - 编写等价的串行版本
   - 用 `MPI_Wtime()` 计时，对比不同进程数下的加速比
   - 记录不同矩阵规模下的性能变化

### 第四阶段：实验报告

9. 整理矩阵划分方式、通信方式、核心代码逻辑
10. 给出串行 vs 并行的运行时间对比表及分析

---

## 相关 MPI 参考文档

- MPICH 自带示例：`mpich-3.3.2.tar/mpich-3.3.2/examples/`
- 安装指南：`mpich-3.3.2.tar/mpich-3.3.2/doc/installguide/install.pdf`
- 用户指南：`mpich-3.3.2.tar/mpich-3.3.2/doc/userguide/user.pdf`

## 环境依赖（已验证）

| 依赖 | 状态 |
|------|------|
| gcc 11.4.0 | 已安装 |
| g++ 11.4.0 | 已安装 |
| gfortran | 已安装 |
| make 4.3 | 已安装 |
| x86_64 / 28核 / 15GB | WSL2 |

## 当前进度

- [x] MPICH 已编译安装（`$HOME/mpich-install/bin/`，已加入 PATH）
- [x] OpenBLAS 已编译安装（`$HOME/openblas-install/`，TARGET=HASWELL, AVX2）
- [x] 矩阵并行运算程序已编写并编译（`matrix/` 目录下 gemm, conv, pooling）

---

## 运行指令

```bash
# 先加载环境
export PATH=$HOME/mpich-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openblas-install/lib:$LD_LIBRARY_PATH
cd /home/xizian/WorkSpace/HPC/实验一/matrix

# 矩阵乘法 (M N K use-blas)
bash run.sh gemm 4          # np=4, 1024x1024
bash run.sh gemm 9          # np=9
bash run.sh gemm 16         # np=16

# 也可直接指定矩阵大小
mpirun -np 4 ./gemm 4096 4096 4096 0

# 卷积 (M N img2col)  0=naive, 1=img2col
bash run.sh conv 4 0        # np=4, naive 卷积
bash run.sh conv 4 1        # np=4, img2col 卷积

# 池化 (M N)
bash run.sh pooling 4       # np=4
```

### 扩进程测试结果（单机 WSL2）

| 进程数 | 1024×1024 gemm | 4096×4096 gemm |
|--------|---------------|---------------|
| np=1 | 88.8 ms | 3687 ms |
| np=4 | 132.3 ms | 3726 ms |
| np=9 | 233.5 ms | 4041 ms |
| np=16 | 372.0 ms | 4482 ms |

> 单机上进程越多反而越慢，因为 MPI 进程间通信开销（内存拷贝+同步）超过了并行计算收益。矩阵增大后差距缩小（O(n³) 计算 vs O(n²) 通信），说明加速比受问题规模和硬件拓扑共同影响。报告可据此分析通信-计算比与并行效率的关系。

---

## 优化版本：单机并行方案对比 (`matrix-opt/`)

针对原始 MPI 版本在单机上通信开销过大的问题，实现了三种优化方案：

| 文件 | 方法 | 原理 |
|------|------|------|
| `serial.cpp` | 纯串行 | 性能基线，单线程三重循环 |
| `openmp.cpp` | OpenMP 多线程 | 线程共享数据，零通信开销 |
| `mpi-shm.cpp` | MPI 共享内存 | 进程间共享物理内存，消除数据拷贝 |

### 运行方式

```bash
cd /home/xizian/WorkSpace/HPC/实验一/matrix-opt
export PATH=$HOME/mpich-install/bin:$PATH

# 串行基线
./serial 1024 1024 1024

# OpenMP（控制线程数）
OMP_NUM_THREADS=4 ./openmp 1024 1024 1024

# MPI 共享内存
mpirun -np 4 ./mpi-shm 1024 1024 1024

# 完整基准测试（生成 bench_result.csv）
bash benchmark.sh

# 可视化（需要 pandas + matplotlib）
python3 plot.py
```

### 4096×4096 性能对比

| 方法 | 最佳配置 | 时间 | 加速比 |
|------|---------|------|--------|
| serial | 1 | 47292 ms | 1.00× |
| mpi-original | np=4 | 2366 ms | 20.0× |
| mpi-shm | np=16 | 4509 ms | 10.5× |
| **openmp** | **16 threads** | **3815 ms** | **12.4×** |

> mpi-original 的 "最快" 实际来自 OpenBLAS + OpenMP 的混合加速（每个 MPI 进程内部 OpenMP 用了全部 28 核），并非 MPI 通信的功劳。真正的单机最优是纯 OpenMP。

### 生成图表

- `speed_comparison.png` — 各方法 2048 矩阵下的耗时与加速比
- `scaling.png` — 不同矩阵规模的性能缩放 (log-log)
- `efficiency.png` — 512 vs 4096 小/大矩阵的并行效率对比
>>>>>>> 5f24a5a (首次提交：初始化项目)
