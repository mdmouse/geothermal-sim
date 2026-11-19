# 架构说明 / Architecture

本文档描述阶段 1 的代码结构与数值方法，并给出后续阶段的前后端拆分规划。

## 1. 模块布局

```
src/geothermal_sim/
├── __init__.py      # 公共 API 导出
├── __main__.py      # python -m geothermal_sim
├── grid.py          # Grid：均匀单元中心网格（x 水平，z 向下为正）
├── materials.py     # Layer / MaterialFields：分层岩石属性映射到网格
├── solver.py        # HeatConductionSolver：算子组装、稳态求解、θ 法时间推进、能量诊断
├── analytic.py      # 一维稳态地温解析解（分段常数 k、Q），用于验证
├── plotting.py      # matplotlib：热力图、温度–深度剖面
└── cli.py           # 命令行入口 geothermal-sim
tests/               # pytest：稳态验证、能量守恒、合理性与 CLI 冒烟测试
examples/            # Python API 示例
docs/                # 本文档
```

数据流：`Grid` + `[Layer]` → `layered_fields()` → `MaterialFields`（`k`、`ρc_p`、`Q`
三个 `(nz, nx)` 数组）→ `HeatConductionSolver` → `steady_state()` 或 `run()` →
`SimulationResult` → `plotting`。

## 2. 控制方程

\[
\rho c_p \frac{\partial T}{\partial t} = \nabla\cdot(k\nabla T) + Q
\]

- `T` 温度（°C），`k` 热导率（W/m/K），`ρ c_p` 体积热容（J/m³/K），`Q` 体积生热率（W/m³）。
- 二维 x–z 剖面，沿走向（y 方向）取单位长度，因此能量量纲为 J/m、热流量为 W/m。

## 3. 空间离散：单元中心有限体积

- 均匀网格，`nx × nz` 个单元，单元中心 `x_i = (i+½)Δx`，`z_j = (j+½)Δz`。
- 对单元积分并以面通量表达 `∇·(k∇T)`。相邻单元间的面通量为
  `k_f (T_B − T_A)/h`，其中面热导率取**调和平均** `k_f = 2 k_A k_B /(k_A + k_B)`。
  当层界面落在单元面上时，调和平均使稳态通量精确，离散稳态解在单元中心与
  解析分段线性地温曲线一致到机器精度（见 `tests/test_steady_state.py`）。
- 离散后得到线性算子 `A`（`scipy.sparse.csr_matrix`，五点格式）和边界源项 `b`：
  `∇·(k∇T) ≈ A T + b`（单位 W/m³）。
- 组装是向量化的（`numpy` 索引数组 + COO 构造），不含 Python 级双重循环。

### 边界条件

| 边界 | 类型 | 离散处理 |
| --- | --- | --- |
| 地表 `z = 0` | Dirichlet，`T = T_s`（默认 20 °C） | 顶层单元中心距地表 `Δz/2`，通量 `2k₀(T_s − T₀)/Δz`，对应 `A` 的对角项 `−2k₀/Δz²` 与源项 `2k₀T_s/Δz²` |
| 底部 `z = L` | Neumann，热流 `q_b`（默认 80 mW/m²，向上进入计算域） | 底层单元源项 `q_b/Δz` |
| 两侧 `x = 0, W` | 绝热（零通量） | 不添加面通量项 |

地表热流诊断 `surface_heat_flux()` 使用与算子完全相同的表达式，保证能量诊断与
离散方程一致。

## 4. 时间推进：θ 法（Crank–Nicolson / 隐式 Euler）

记 `M = diag(1/ρc_p) A`，`s = (b + Q)/ρc_p`，则

\[
(I - \theta\,\Delta t\,M)\,T^{n+1} = (I + (1-\theta)\,\Delta t\,M)\,T^{n} + \Delta t\, s
\]

- `θ = 0.5`（默认）为 Crank–Nicolson，二阶精度；`θ = 1` 为隐式 Euler，一阶、强阻尼，
  适合极大时间步直接逼近稳态。两者对任意 `Δt` 均无条件稳定，因此时间步只受精度约束，
  不受显式格式的 `Δt < h²/(2κ)` 限制（对 100 m 网格约为 50 年）。
- 系数矩阵在整个模拟中不变，`run()` 对左端矩阵做一次 `scipy.sparse.linalg.splu`
  分解，之后每步只做一次稀疏矩阵–向量乘与一次三角回代。50 × 100 网格、200 步的
  典型运行耗时不到 1 s。
- `steady_state()` 直接求解 `A T = −(b + Q)`（`spsolve`）。

### 能量守恒

对所有单元求和时，内部面通量成对抵消（`A` 的内部部分列和为零），因此离散能量方程为

\[
E^{n+1} - E^{n} = \Delta t \left[\,Q_{\text{bottom}} + Q_{\text{gen}} - \theta\, Q_{\text{top}}(T^{n+1}) - (1-\theta)\, Q_{\text{top}}(T^{n})\right]
\]

`run()` 以同样的 θ 加权累计 `heat_in_bottom`、`heat_out_top`、`heat_generated`，
`SimulationResult.energy_balance_residual` 应为舍入误差量级（测试要求相对残差 < 1e-9）。

## 5. 验证（tests/）

| 测试 | 内容 |
| --- | --- |
| `test_steady_state.py` | 均质线性地温（底温 180 °C）；三层分段线性地温（机器精度）；含生热的二次地温（二阶误差）；瞬态长时间收敛到稳态 |
| `test_energy_balance.py` | CN 与隐式 Euler 的离散能量平衡精确成立；含生热的能量平衡；稳态下净热输入为零且时间推进不改变解；早期热量几乎全部进入储存 |
| `test_sanity.py` | 网格几何、分层赋值、非法输入、温度随深度单调且有界、超大时间步稳定、快照记录、CLI 输出文件 |

## 6. 前后端拆分规划（阶段 2+）

```
┌──────────────────────────┐       JSON / WebSocket        ┌──────────────────────────────┐
│  前端 React + Three.js    │  ◄──────────────────────────► │  后端 FastAPI (Python)         │
│  - 参数面板（地层、边界、时间）│                               │  - /simulate：接收参数，调用     │
│  - 三维温度场体渲染 / 切片    │                               │    geothermal_sim 求解器        │
│  - 井位拖拽、时间轴播放       │                               │  - 任务队列 + 进度推送           │
│  - 经济指标仪表盘            │                               │  - 结果缓存（npz / parquet）     │
└──────────────────────────┘                               └──────────────────────────────┘
```

- **后端**：`geothermal_sim` 保持为纯计算库，不依赖 Web 框架；在其上新增 `api/`
  包提供 REST / WebSocket 接口，返回温度场数组（可下采样）、剖面、能量平衡和后续的
  产能 / 经济指标。长时间模拟使用后台任务并按快照推送进度。
- **前端**：React 负责参数输入与状态管理，Three.js 负责三维温度场可视化（体渲染、
  等值面、剖面切片），时间轴回放快照序列。
- **求解器扩展路径**：
  - 三维：`Grid` 增加 y 方向，算子组装由五点变为七点格式，线性求解器切换为迭代法
    （CG / AMG 预条件）。
  - 开采模拟（阶段 3）：在 `Q` 项中加入井的汇 / 源，或引入对流项与储层流动耦合。
  - 经济评价（阶段 4）：独立 `economics` 模块，输入产能时间序列，输出回收期 / LCOE。
