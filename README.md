# geothermal-sim — 交互式地热资源模拟系统

> **English summary.** `geothermal-sim` is an interactive geothermal resource
> simulator. Phase 1 (this repo) is a Python + NumPy + SciPy solver for 2D
> transient heat conduction, `ρ c_p ∂T/∂t = ∇·(k∇T) + Q`, on a 5 km deep
> cross-section with a fixed 20 °C surface (Dirichlet), an 80 mW/m² basal
> heat flux (Neumann), layered rock conductivity and adjustable simulation
> time. It uses a conservative cell-centred finite-volume discretisation
> with Crank–Nicolson time stepping on `scipy.sparse` matrices, outputs a
> temperature heatmap and a temperature–depth profile, and ships pytest
> tests for energy conservation and agreement with the analytic steady
> state. Later phases (3D visualisation, wells, reservoir depletion,
> economics) are documented in the roadmap but not yet implemented.

[![tests](https://github.com/mdmouse/geothermal-sim/actions/workflows/tests.yml/badge.svg)](https://github.com/mdmouse/geothermal-sim/actions/workflows/tests.yml)

## 项目简介

本项目旨在构建一个**交互式地热资源模拟系统**：从地下温度场的物理模拟出发，
逐步扩展到地热开采过程模拟和工程经济评价，最终提供可在浏览器中交互操作的
三维可视化界面。

第一阶段（当前仓库）实现核心数值内核：二维瞬态热传导求解器。

## 系统分层

| 层 | 内容 | 状态 |
| --- | --- | --- |
| **1. 地质 / 温度场层** | 地层结构（分层热导率、密度、比热、放射性生热），热传导方程求解，稳态与瞬态温度场 | 阶段 1，本仓库已实现 |
| **2. 开采模拟层** | 生产井 / 回灌井布置、井筒与储层换热、储层热量衰竭、产能随时间变化 | 规划中 |
| **3. 经济评价层** | 钻井与地面设施投资、运营成本、售热/售电收入、投资回收期与 LCOE | 规划中 |

## 阶段 1 功能

- 控制方程：`ρ c_p ∂T/∂t = ∇·(k ∇T) + Q`（二维 x–z 剖面）
- 计算域：默认宽 5 km、深 5 km，网格分辨率可调
- 边界条件：地表 20 °C（Dirichlet）；底部热流 80 mW/m²（Neumann）；两侧绝热
- 岩石属性：水平分层，每层独立的 `k`、`ρ`、`c_p` 和体积生热率 `Q`
- 数值方法：单元中心有限体积 + θ 法时间推进（默认 Crank–Nicolson），
  `scipy.sparse` 稀疏矩阵与 LU 分解，无条件稳定，严格守恒
- 输出：温度热力图、温度–深度剖面（PNG），以及含能量平衡的 `summary.json`
- 测试：稳态解与解析（分段线性 / 二次）地温曲线对比、能量守恒、基本合理性检查

## 快速开始

```bash
git clone https://github.com/mdmouse/geothermal-sim.git
cd geothermal-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 运行测试
pytest

# 三层地壳，模拟 20 万年，输出到 output/
geothermal-sim --layers "1000:2.0,3000:2.5,5000:3.0" --years 200000 --out output

# 直接求稳态
geothermal-sim --layers "1000:2.0,5000:3.0" --steady-only --out output_steady
```

`--layers` 的格式为逗号分隔的 `底深_m:k[:ρ[:c_p[:Q]]]`，单位均为 SI
（W/m/K、kg/m³、J/kg/K、W/m³）。其它常用参数：`--heat-flux`（mW/m²）、
`--surface-temp`（°C）、`--nx/--nz`、`--dt-years`、`--theta`（0.5 为
Crank–Nicolson，1 为隐式 Euler）、`--initial steady`（以稳态作为初值）。
完整列表见 `geothermal-sim --help`。

Python API 示例见 [`examples/basic_run.py`](examples/basic_run.py)：

```python
from geothermal_sim import *

grid = Grid(width=5000, depth=5000, nx=50, nz=100)
layers = [Layer(1000, k=2.0), Layer(3000, k=2.5), Layer(5000, k=3.0)]
solver = HeatConductionSolver(grid, layered_fields(grid, layers),
                              BoundaryConditions(20.0, 0.080))
result = solver.run(solver.uniform_initial(),
                    total_time=2e5 * SECONDS_PER_YEAR, dt=1e3 * SECONDS_PER_YEAR)
plot_heatmap(grid, result.temperature, "heatmap.png")
```

## 路线图

- [x] **阶段 1** — 二维热传导求解器（Python / NumPy / SciPy），CLI，单元测试
- [ ] **阶段 2** — 前后端分离：FastAPI 服务封装求解器；React + Three.js 三维温度场可视化与参数交互
- [ ] **阶段 3** — 生产井 / 回灌井设计：井位、流量、回灌温度；井—储层换热与储层热衰竭模拟
- [ ] **阶段 4** — 工程经济：投资与运营成本模型、产能—收益曲线、投资回收期、LCOE 与敏感性分析

详细的模块划分、数值方法和前后端规划见 [`docs/architecture.md`](docs/architecture.md)。

## 许可证

[MIT](LICENSE)
