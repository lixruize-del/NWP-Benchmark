# NWP 预测评估 Benchmark 方案（中期 3–15 天）

> 本文档将当前项目的 Benchmark 需求整理为可执行规范，便于后续实现统一数据流程、模型推理、指标评估与报告产出。

## 1. 目标与范围

### 1.1 目标
- 定量评估 AI 模型在中期（3–15 天）天气预报中的可用性与性能。
- 与传统数值模式（NWP）及主流气象大模型进行可比评估。
- 提供统一评估流程、统一验证数据、统一基线实现与可复现代码。

### 1.2 对比模型
- **确定性模型（优先）**
  - Pangu-Weather
  - GraphCast
  - Aurora
  - FourCastNet（待确认版本）
  - NeuralGCM
  - FuXi
  - FengWu
  - Stormer
  - AIFS
- **集合模型（可并行推进）**
  - GenCast
  - FourCastNet-v3（ensemble 配置）
  - Fuxi-ENS

### 1.3 评估范围
- **变量**：
  - 2m 气温（t2m）
  - 降水（16h/24h 累计，按模型可用性）
  - 海表温度（SST）
  - 海平面气压（MSLP）
  - 1000/500 hPa 位势高度
  - 10m 风场（u10/v10）
  - 相对湿度 / 比湿
- **空间尺度**：
  - 全球格点（0.25° / 0.5° / 1°）
  - 区域（中国/东亚/北半球）
  - 台站点观测验证
- **时间尺度**：
  - 预报时效 15 天
  - 评估步长 6 小时
- **输出类型**：
  - 确定性预报场
  - 概率/集合分布

## 2. 数据集与验证资料

### 2.1 评估时间范围
- 2024–2025（确定性）
- 集合评估优先 2024 年 7–12 月

### 2.2 真值与再分析
- ERA5：主验证基准（高分辨率再分析）
- Analysis（业务运行分析场，用于 contribution 分析）

### 2.3 降水评估数据
- GPM IMERG（如模型输出可比降水）
- ERA5 precipitation
- Analysis precipitation

### 2.4 地面观测
- Weather-10K / Weather-5K（t2m、msl、u10/v10 或 windspeed）

### 2.5 NWP 业务基线
- ECMWF IFS（2024 预测数据）
- GFS（次优先级）

### 2.6 数据格式建议
- NetCDF（CF conventions）或 Zarr
- 统一时间坐标、变量单位、经纬坐标与元数据
- 数据必须记录来源、处理步骤、重网格信息

## 3. 基线模型（Baseline）

### 3.1 简单基准
- Climatology（按日历日气候态或季节性气候场）
- Persistence（建议加入，便于低门槛 sanity check）

### 3.2 NWP 基线
- ECMWF IFS
- Bias-corrected NWP（基于 reforecast 的订正）

### 3.3 AI 基线
- 使用上述可运行 AI 模型作为统一对比组

### 3.4 实施约束
- 所有基线在同一验证集、同一目标网格、同一起报规则下运行
- 对齐初始化时次、预报步长与 lead time

## 4. 评估指标（参考 WeatherBench）

### 4.1 确定性指标
- WRMSE（按 lead time 与空间平均，同时可按经纬分解）
- ACC（Anomaly Correlation Coefficient）
- Bias / Mean Error
- Activity（建议定义为方差或扰动活动度量）
- Power Spectrum（空间尺度误差诊断）

### 4.2 概率/集合指标
- CRPS / CRPSS
- Brier Score / Brier Skill Score（阈值事件）
- Spread-skill relationship

### 4.3 物理一致性检查（可选增强）
- 区域/全球水量、能量、动量守恒误差
- 若模型具备守恒约束，报告守恒残差

### 4.4 指标展示形式
- Lead-time score curve
- 季节分组统计
- 区域分组统计
- 空间误差分布图 / 差值图 / score card

## 5. 实验设计与验证协议

### 5.1 起报与时间对齐
- 固定 00/12 UTC 起报（与 IFS 可用时次对齐）
- 统一 6 小时步进到 15 天

### 5.2 集合预测
- 统一 50 members（可用模型按近似配置映射）

### 5.3 极端事件评估（Contribution）
- 总体事件评分：SEDI（如 >99th percentile 降水、极端高温等）
- 事件驱动评估：
  - 热带气旋：生成命中率、路径/强度误差
  - 热浪：覆盖区域内 t2m 对台站 RMSE
  - 寒潮/冬季风暴：关键区域 t2m RMSE
  - 强降水过程：累计降水 RMSE / ACC

### 5.4 可视化要求
- 同一起报时刻多模型对比图
- 关键时效建议：1/3/5/7/10 天

## 6. 可重复性与工程规范

### 6.1 代码组织
- 数据下载与处理脚本
- baseline 与 AI 推理脚本
- 评估脚本
- 可视化脚本
- YAML/JSON 实验配置

### 6.2 环境与容器
- 提供 Docker/Singularity 描述
- 固定 Python/conda/关键依赖版本

### 6.3 实验记录
- 建议维护 experiment registry：
  - commit id
  - 运行参数
  - seed
  - 节点/硬件信息

### 6.4 报告产出
- 自动生成 HTML/PDF 报告
- 输出 lead-time 曲线、空间图、表格并归档到 artifact

## 7. 资源估算与运行安排
- 首阶段：A100-40GB 验证单模型闭环
- 二阶段：迁移 H200 做大规模并行推理评估
- 关注点：
  - TB 级存储
  - I/O 吞吐
  - 并行重网格与评估 CPU/GPU 配比
- 优化建议：
  - 并行 I/O
  - Zarr chunk 优化
  - 分区域并行

## 8. 交付物定义
- 标准化验证数据包 + 重网格脚本
- Baseline 实现（Climatology / Persistence / NWP / AI）
- 统一评估库 + 自动化评估 pipeline
- 完整报告（总体性能、分变量/区域/季节、极端事件）
- 可视化 dashboard
- 图表/原始结果/评估结果归档下载包

## 9. 建议时间表（示例）
- Week 0–2：需求确认、数据可用性评估、验证协议冻结
- Week 3–6：数据获取、预处理、重网格、baseline 落地
- Week 7–10：AI 推理与初步评估
- Week 11–14：深度诊断、极端事件与后处理
- Week 15–16：报告整理、可视化交付、code freeze

## 10. 推荐工具栈
- 数据与科学计算：xarray, dask, cfgrib, netCDF4, zarr
- 指标与诊断：xskillscore, properscoring, MetPy
- 可选：climetlab, pygrib, wrf-python（按场景）

---

## 11. 与当前仓库的对齐建议（Next Actions）
1. 在 `configs/` 新增统一评估配置（变量、区域、lead times、指标开关）。
2. 在 `src/common/` 增加标准化重网格与单位归一模块。
3. 在 `src/common/evaluate.py` 中补齐 CRPS/CRPSS、Brier、SEDI、分组统计接口。
4. 新增 `scripts/benchmark_run.sh` 串联下载、推理、评估、作图、报告导出。
5. 将实验 registry（例如 JSONL）写入 `outputs/registry/`。
